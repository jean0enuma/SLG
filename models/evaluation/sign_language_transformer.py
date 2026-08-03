# -*- coding: utf-8 -*-
"""
Sign Language Transformers (Camgoz et al., CVPR 2020) の再実装
"Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation"

公式実装 https://github.com/neccam/slt (JoeyNMT ベース) を参考に，
CNN 特徴の代わりに入力を (B, T, 150) の骨格特徴ベクトル系列とした版．

構成 (論文 Fig.2 / Sec.3 に対応):
  - SpatialEmbedding : Linear(150→D) + Masked BatchNorm + ReLU  (論文 Table 2 の "+ BN & ReLU")
  - WordEmbedding    : nn.Embedding (論文では one-hot → Linear と等価)
  - PositionalEncoding: 正弦波 PE (Vaswani et al.)
  - SLRT (encoder)   : Pre-LN Transformer Encoder → Linear → CTC で p(G|V)   (式 2-4)
  - SLTT (decoder)   : 自己回帰 Transformer Decoder → p(S|V) = Π p(w_u|h_u)  (式 5-7)
  - Joint loss       : L = λ_R * L_R(CTC) + λ_T * L_T(XEnt)                  (式 8)

論文設定: hidden=512, heads=8, layers=3, ff=2048, dropout=0.1, Xavier init,
          Adam(lr=1e-3, β=(0.9,0.998), wd=1e-3), plateau scheduler (factor 0.7),
          λ_R=5.0, λ_T=1.0 が最良 (Table 4)
"""

import math
from itertools import groupby
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
def make_pad_mask(lengths: Tensor, max_len: int) -> Tensor:
    """長さ配列から key_padding_mask を作る． True = パディング位置．
    lengths: (B,)  →  (B, max_len) bool
    """
    ar = torch.arange(max_len, device=lengths.device)[None, :]  # (1, T)
    return ar >= lengths[:, None]  # (B, T)


def subsequent_mask(size: int, device=None) -> Tensor:
    """SLTT 用 causal mask． True = attention を禁止する位置．"""
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


NEG_INF = float("-inf")


def _logsumexp(*xs: float) -> float:
    """log(Σ exp(x_i)) を数値安定に計算 (スカラー版)"""
    m = max(xs)
    if m == NEG_INF:
        return NEG_INF
    return m + math.log(sum(math.exp(x - m) for x in xs))


def ctc_prefix_beam_search(
    log_probs: Tensor,      # (T, C)  1 サンプル分のフレーム毎 log softmax
    blank: int = 0,
    beam_size: int = 5,
    prune_topk: int = 30,
) -> list:
    """CTC prefix beam search (Hannun et al. 2014, "First-Pass Large Vocabulary
    Continuous Speech Recognition using Bi-Directional Recurrent DNNs").

    greedy (best path) 復号と異なり，同一ラベル系列に潰れる複数のアライメント
    経路 π ∈ B^{-1}(G) の確率を周辺化しながら探索する (論文 式 3 の
    p(G|V) = Σ_π p(π|V) を近似的に最大化する系列を返す)．

    各 prefix について 2 つの log 確率を保持する:
      p_b  : その prefix で終わり，最後のフレームが blank
      p_nb : その prefix で終わり，最後のフレームが非 blank
    区別が必要なのは，同一ラベルの連続 (例 "AA") を生成できるのが
    blank を挟んだ経路 (A, _, A) のみであるため．

    prune_topk: 各フレームで展開する候補ラベルを上位 k 個に制限 (語彙が
    大きい場合の高速化．k=C で厳密な prefix beam search)．
    """
    T, C = log_probs.shape
    lp = log_probs.cpu().tolist()

    # beams: prefix(tuple) -> [log p_b, log p_nb]
    beams = {(): [0.0, NEG_INF]}

    for t in range(T):
        frame = lp[t]
        # フレーム毎の候補プルーニング (blank は常に含める)
        if prune_topk < C:
            cand = sorted(range(C), key=lambda c: frame[c], reverse=True)[:prune_topk]
            if blank not in cand:
                cand.append(blank)
        else:
            cand = range(C)

        new_beams: dict = {}

        def _get(prefix):
            if prefix not in new_beams:
                new_beams[prefix] = [NEG_INF, NEG_INF]
            return new_beams[prefix]

        for prefix, (p_b, p_nb) in beams.items():
            p_total = _logsumexp(p_b, p_nb)
            last = prefix[-1] if prefix else None

            for c in cand:
                p_c = frame[c]
                if c == blank:
                    # blank: prefix は変化せず p_b を更新
                    e = _get(prefix)
                    e[0] = _logsumexp(e[0], p_total + p_c)
                elif c == last:
                    # 同一ラベルの繰り返しフレーム → prefix 不変 (p_nb 側)
                    e = _get(prefix)
                    e[1] = _logsumexp(e[1], p_nb + p_c)
                    # blank を挟んで同一ラベルを新たに出力 → prefix 拡張
                    e2 = _get(prefix + (c,))
                    e2[1] = _logsumexp(e2[1], p_b + p_c)
                else:
                    # 新しいラベルで prefix を拡張
                    e = _get(prefix + (c,))
                    e[1] = _logsumexp(e[1], p_total + p_c)

        # 上位 beam_size 本に枝刈り
        beams = dict(
            sorted(
                new_beams.items(),
                key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_size]
        )

    best = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))
    return list(best[0])


# ---------------------------------------------------------------------------
# Positional Encoding (Vaswani et al. 2017, 論文 Sec.3.1)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: Tensor) -> Tensor:  # (B, T, D)
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Masked BatchNorm (公式実装 signjoey/embeddings.py の MaskedNorm 相当)
#   パディングフレームを統計量計算から除外する BatchNorm1d
# ---------------------------------------------------------------------------
class MaskedBatchNorm(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features)
        self.num_features = num_features

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        """x: (B, T, D), pad_mask: (B, T) True=pad"""
        if self.training:
            B, T, D = x.shape
            reshaped = x.reshape(-1, D)                       # (B*T, D)
            valid = (~pad_mask).reshape(-1, 1)                # (B*T, 1)
            selected = torch.masked_select(reshaped, valid).reshape(-1, D)
            normed = self.norm(selected)
            out = reshaped.masked_scatter(valid, normed)
            return out.reshape(B, T, D)
        else:
            B, T, D = x.shape
            return self.norm(x.reshape(-1, D)).reshape(B, T, D)


# ---------------------------------------------------------------------------
# 埋め込み層 (論文 式 1)
# ---------------------------------------------------------------------------
class SpatialEmbedding(nn.Module):
    """f_t = SpatialEmbedding(x_t)
    論文では CNN 特徴 → (BN+ReLU) だが，ここでは骨格特徴 (B,T,150) を
    Linear で D 次元へ射影し，同じく Masked BN + ReLU を適用する．
    """

    def __init__(self, input_size: int, d_model: int, scale: bool = True,activation: str = "softsign"):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.norm = MaskedBatchNorm(d_model)
        self.scale_factor = math.sqrt(d_model) if scale else 1.0
        self.activation=nn.Softsign() if activation=="softsign" else nn.ReLU()

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        x = self.proj(x)
        x = self.norm(x, pad_mask)
        x = self.activation(x)
        return x * self.scale_factor


class WordEmbedding(nn.Module):
    """m_u = WordEmbedding(w_u)  (one-hot → Linear は nn.Embedding と等価)"""

    def __init__(self, vocab_size: int, d_model: int, pad_idx: int, scale: bool = True):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.scale_factor = math.sqrt(d_model) if scale else 1.0

    def forward(self, x: Tensor) -> Tensor:
        return self.lut(x) * self.scale_factor


# ---------------------------------------------------------------------------
# Transformer layers (JoeyNMT と同じ Pre-LN 構成)
# ---------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(self.layer_norm(x)) + x


class EncoderLayer(nn.Module):
    """SLRT の 1 層: Self-Attention → FF (残差 + LN 付き，式 2)"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        x_norm = self.layer_norm(x)
        h, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=pad_mask, need_weights=False)
        x = x + self.dropout(h)
        return self.ff(x)


class DecoderLayer(nn.Module):
    """SLTT の 1 層: Masked Self-Attention → Encoder-Decoder Attention → FF (式 5)"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.x_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dec_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: Tensor,               # (B, U, D)  デコーダ入力
        memory: Tensor,          # (B, T, D)  SLRT 出力 z_{1:T}
        tgt_causal_mask: Tensor, # (U, U)     True=禁止
        tgt_pad_mask: Tensor,    # (B, U)     True=pad
        src_pad_mask: Tensor,    # (B, T)     True=pad
    ) -> Tensor:
        x_norm = self.x_layer_norm(x)
        h, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=tgt_causal_mask,
            key_padding_mask=tgt_pad_mask,
            need_weights=False,
        )
        x = x + self.dropout(h)

        x_norm = self.dec_layer_norm(x)
        h, _ = self.cross_attn(
            x_norm, memory, memory,
            key_padding_mask=src_pad_mask,
            need_weights=False,
        )
        x = x + self.dropout(h)
        return self.ff(x)


# ---------------------------------------------------------------------------
# SLRT: Sign Language Recognition Transformer (論文 Sec.3.2)
# ---------------------------------------------------------------------------
class SLRT(nn.Module):
    def __init__(self, d_model=512, n_heads=8, num_layers=3, d_ff=2048, dropout=0.1):
        super().__init__()
        self.pe = PositionalEncoding(d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, f: Tensor, pad_mask: Tensor) -> Tensor:
        """f: (B, T, D) SpatialEmbedding 済み特徴 → z: (B, T, D)"""
        x = self.emb_dropout(self.pe(f))  # f̂_t = f_t + PE(t)
        for layer in self.layers:
            x = layer(x, pad_mask)
        return self.layer_norm(x)


# ---------------------------------------------------------------------------
# SLTT: Sign Language Translation Transformer (論文 Sec.3.3)
# ---------------------------------------------------------------------------
class SLTT(nn.Module):
    def __init__(self, d_model=512, n_heads=8, num_layers=3, d_ff=2048, dropout=0.1):
        super().__init__()
        self.pe = PositionalEncoding(d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, m: Tensor, memory: Tensor, tgt_pad_mask: Tensor, src_pad_mask: Tensor) -> Tensor:
        """m: (B, U, D) WordEmbedding 済み系列 → h: (B, U, D)"""
        U = m.size(1)
        x = self.emb_dropout(self.pe(m))  # m̂_u = m_u + PE(u)
        causal = subsequent_mask(U, device=m.device)
        for layer in self.layers:
            x = layer(x, memory, causal, tgt_pad_mask, src_pad_mask)
        return self.layer_norm(x)


# ---------------------------------------------------------------------------
# 本体: Sign Language Transformer
# ---------------------------------------------------------------------------
class SignLanguageTransformer(nn.Module):
    """
    入力:  骨格特徴系列 (B, T, 150)
    出力:  gloss_log_probs (T, B, |G|+1)  … CTC 用 (blank=0)
           word_logits     (B, U, |W|)    … 翻訳用
    """

    def __init__(
        self,
        input_size: int = 150,
        gloss_vocab_size: int = 1066 + 1,   # +1: CTC blank (index 0)
        text_vocab_size: int = 2887 + 4,    # +4: special tokens
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.3,
        txt_pad_idx: int = 0,
        txt_bos_idx: int = 1,
        txt_eos_idx: int = 2,
        ctc_blank_idx: int = 0,
    ):
        super().__init__()
        self.txt_pad_idx = txt_pad_idx
        self.txt_bos_idx = txt_bos_idx
        self.txt_eos_idx = txt_eos_idx
        self.ctc_blank_idx = ctc_blank_idx

        self.sgn_embed = SpatialEmbedding(input_size, d_model,scale=False)
        self.txt_embed = WordEmbedding(text_vocab_size, d_model, pad_idx=txt_pad_idx)

        self.encoder = SLRT(d_model, n_heads, num_layers, d_ff, dropout)
        self.decoder = SLTT(d_model, n_heads, num_layers, d_ff, dropout)

        self.gloss_output_layer = nn.Linear(d_model, gloss_vocab_size)  # → CTC
        self.word_output_layer = nn.Linear(d_model, text_vocab_size)

        self._init_xavier()

    def _init_xavier(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ---- encode / decode -------------------------------------------------
    def encode(self, sgn: Tensor, sgn_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        src_pad_mask = make_pad_mask(sgn_lengths, sgn.size(1))       # (B, T)
        f = self.sgn_embed(sgn, src_pad_mask)                        # (B, T, D)
        z = self.encoder(f, src_pad_mask)                            # (B, T, D)
        return z, src_pad_mask

    def decode(self, txt_input: Tensor, memory: Tensor, src_pad_mask: Tensor) -> Tensor:
        tgt_pad_mask = txt_input.eq(self.txt_pad_idx)                # (B, U)
        m = self.txt_embed(txt_input)                                # (B, U, D)
        h = self.decoder(m, memory, tgt_pad_mask, src_pad_mask)      # (B, U, D)
        return self.word_output_layer(h)                             # (B, U, |W|)

    # ---- forward ----------------------------------------------------------
    def forward(
        self,
        sgn: Tensor,          # (B, T, 150)
        sgn_lengths: Tensor,  # (B,)
        txt_input: Tensor,    # (B, U)  <bos> w_1 ... w_{U-1}
    ) -> Tuple[Tensor, Tensor]:
        z, src_pad_mask = self.encode(sgn, sgn_lengths)

        # 認識ヘッド (SLRT → CTC): (B,T,C) → log_softmax → (T,B,C)
        gloss_scores = self.gloss_output_layer(z)
        gloss_log_probs = gloss_scores.log_softmax(-1).permute(1, 0, 2)

        # 翻訳ヘッド (SLTT)
        word_logits = self.decode(txt_input, z, src_pad_mask)
        return gloss_log_probs, word_logits

    # ---- 推論: CTC greedy decode (Sign2Gloss) ------------------------------
    @torch.no_grad()
    def recognize_greedy(self, sgn: Tensor, sgn_lengths: Tensor):
        z, _ = self.encode(sgn, sgn_lengths)
        pred = self.gloss_output_layer(z).argmax(-1)  # (B, T)
        results = []
        for b in range(pred.size(0)):
            seq = pred[b, : sgn_lengths[b]].tolist()
            collapsed = [k for k, _ in groupby(seq) if k != self.ctc_blank_idx]
            results.append(collapsed)
        return results

    # ---- 推論: CTC beam search decode (Sign2Gloss) --------------------------
    @torch.no_grad()
    def recognize_beam(
        self, sgn: Tensor, sgn_lengths: Tensor,
        beam_size: int = 5, prune_topk: int = 30,
    ):
        """CTC prefix beam search (Hannun et al. 2014) による gloss 系列復号．
        公式実装では TensorFlow の ctc_beam_search_decoder を使用しているが，
        ここでは pure PyTorch/Python で同等の prefix beam search を実装する．
        """
        z, _ = self.encode(sgn, sgn_lengths)
        log_probs = self.gloss_output_layer(z).log_softmax(-1)  # (B, T, C)
        results = []
        for b in range(log_probs.size(0)):
            lp = log_probs[b, : sgn_lengths[b]]  # (T_b, C)
            results.append(
                ctc_prefix_beam_search(
                    lp, blank=self.ctc_blank_idx,
                    beam_size=beam_size, prune_topk=prune_topk,
                )
            )
        return results

    # ---- 推論: CTC beam search decode (torchaudio / Flashlight 版) ----------
    @torch.no_grad()
    def recognize_beam_torchaudio(
        self, sgn: Tensor, sgn_lengths: Tensor,
        beam_size: int = 5, prune_topk: int = 30,
    ):
        """torchaudio.models.decoder.ctc_decoder (Flashlight C++ 実装) による
        CTC beam search．自前の ctc_prefix_beam_search と同じ復号を C++ で
        高速に行う (dev/test 全体の評価はこちらを推奨)．

        - lexicon=None の lexicon-free モード
        - log_add=True で同一 prefix に潰れる経路を logsumexp で周辺化
          (prefix beam search と等価; False だと max 近似になるので注意)
        - beam_size_token が自前実装の prune_topk に対応

        要: pip install torchaudio flashlight-text
        """
        from torchaudio.models.decoder import ctc_decoder

        C = self.gloss_output_layer.out_features
        if getattr(self, "_ta_decoder_cfg", None) != (beam_size, prune_topk, C):
            tokens = [f"g{i}" for i in range(C)]
            tokens[self.ctc_blank_idx] = "<blank>"
            self._ta_decoder = ctc_decoder(
                lexicon=None,
                tokens=tokens,
                blank_token="<blank>",
                sil_token="<blank>",   # 無音トークンは使わないので blank に割当
                beam_size=beam_size,
                beam_size_token=min(prune_topk, C),
                nbest=1,
                log_add=True,
            )
            self._ta_decoder_cfg = (beam_size, prune_topk, C)

        z, _ = self.encode(sgn, sgn_lengths)
        emissions = self.gloss_output_layer(z).log_softmax(-1)  # (B, T, C)
        hyps = self._ta_decoder(
            emissions.float().cpu().contiguous(), sgn_lengths.cpu().to(torch.int32)
        )
        return [h[0].tokens.tolist() for h in hyps]

    # ---- 推論: greedy 翻訳 (Sign2Text) --------------------------------------
    @torch.no_grad()
    def translate_greedy(self, sgn: Tensor, sgn_lengths: Tensor, max_len: int = 30):
        B = sgn.size(0)
        z, src_pad_mask = self.encode(sgn, sgn_lengths)
        ys = torch.full((B, 1), self.txt_bos_idx, dtype=torch.long, device=sgn.device)
        finished = torch.zeros(B, dtype=torch.bool, device=sgn.device)
        for _ in range(max_len):
            logits = self.decode(ys, z, src_pad_mask)           # (B, u, |W|)
            next_tok = logits[:, -1].argmax(-1, keepdim=True)   # (B, 1)
            next_tok = next_tok.masked_fill(finished.unsqueeze(1), self.txt_pad_idx)
            ys = torch.cat([ys, next_tok], dim=1)
            finished |= next_tok.squeeze(1).eq(self.txt_eos_idx)
            if finished.all():
                break
        return ys[:, 1:].cpu().tolist() # <bos> を除く

    # ---- 推論: beam search 翻訳 --------------------------------------------
    @torch.no_grad()
    def translate_beam(
        self, sgn: Tensor, sgn_lengths: Tensor,
        beam_size: int = 5, max_len: int = 30, length_penalty_alpha: float = 1.0,
    ):
        """簡易 beam search (バッチ内 1 サンプルずつ処理)．
        論文 Sec.5.1: beam width 0-10, length penalty α 0-2 を dev で探索．
        score = log p / lp,  lp = ((5+|Y|)/6)^α  (Wu et al. 2016)
        """
        device = sgn.device
        outputs = []
        for b in range(sgn.size(0)):
            z, src_pad_mask = self.encode(
                sgn[b : b + 1, : sgn_lengths[b]], sgn_lengths[b : b + 1]
            )
            beams = [(0.0, [self.txt_bos_idx], False)]  # (logp, tokens, finished)
            for _ in range(max_len):
                candidates = []
                for logp, toks, fin in beams:
                    if fin:
                        candidates.append((logp, toks, fin))
                        continue
                    ys = torch.tensor([toks], device=device)
                    logits = self.decode(ys, z, src_pad_mask)[0, -1]
                    log_probs = logits.log_softmax(-1)
                    topv, topi = log_probs.topk(beam_size)
                    for v, i in zip(topv.tolist(), topi.tolist()):
                        candidates.append((logp + v, toks + [i], i == self.txt_eos_idx))

                def lp_score(c):
                    length = len(c[1]) - 1
                    lp = ((5.0 + length) / 6.0) ** length_penalty_alpha
                    return c[0] / lp

                beams = sorted(candidates, key=lp_score, reverse=True)[:beam_size]
                if all(f for _, _, f in beams):
                    break
            best = max(beams, key=lp_score)[1][1:]  # <bos> 除去
            if best and best[-1] == self.txt_eos_idx:
                best = best[:-1]
            outputs.append(best)
        return outputs


# ---------------------------------------------------------------------------
# 損失関数 (論文 式 4, 7, 8)
# ---------------------------------------------------------------------------
class XentLoss(nn.Module):
    """label smoothing 付き cross-entropy (公式実装 loss.py 相当，pad 無視)"""

    def __init__(self, pad_idx: int, smoothing: float = 0.0):
        super().__init__()
        self.pad_idx = pad_idx
        self.smoothing = smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """logits: (B, U, V), targets: (B, U) → sum-reduced loss"""
        V = logits.size(-1)
        log_probs = logits.log_softmax(-1).reshape(-1, V)
        targets = targets.reshape(-1)
        if self.smoothing <= 0.0:
            return F.nll_loss(log_probs, targets, ignore_index=self.pad_idx, reduction="sum")
        # smoothed KL
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.smoothing / (V - 2))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            smooth[:, self.pad_idx] = 0.0
            smooth[targets == self.pad_idx] = 0.0
        return F.kl_div(log_probs, smooth, reduction="sum")


class JointLoss(nn.Module):
    """L = λ_R * L_R + λ_T * L_T  (論文 式 8; Table 4 最良は λ_R=5.0, λ_T=1.0)"""

    def __init__(self, txt_pad_idx: int, blank_idx: int = 0,
                 lambda_r: float = 5.0, lambda_t: float = 1.0, smoothing: float = 0.0):
        super().__init__()
        self.lambda_r = lambda_r
        self.lambda_t = lambda_t
        self.ctc = nn.CTCLoss(blank=blank_idx, zero_infinity=False, reduction="none")
        self.xent = XentLoss(pad_idx=txt_pad_idx, smoothing=smoothing)

    def forward(
        self,
        gloss_log_probs: Tensor,  # (T, B, |G|)
        sgn_lengths: Tensor,      # (B,)
        gloss_targets: Tensor,    # (B, N)  pad は何でも良い (lengths で切る)
        gloss_lengths: Tensor,    # (B,)
        word_logits: Tensor,      # (B, U, |W|)
        txt_targets: Tensor,      # (B, U)  w_1 ... w_U <eos> (pad 埋め)
        batch_size: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        loss_r_batch = self.ctc(gloss_log_probs, gloss_targets, sgn_lengths, gloss_lengths)#(B,)
        loss_r=torch.tensor(0.0, device=loss_r_batch.device)
        for b in range(batch_size):
            loss_r=loss_r_batch[b]/gloss_lengths[b]+loss_r
        loss_r/=batch_size
        loss_t = self.xent(word_logits, txt_targets)/batch_size
        total = self.lambda_r * loss_r + self.lambda_t * loss_t
        return total, loss_r, loss_t


# ---------------------------------------------------------------------------
# 動作確認 (ダミーデータ)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, F_IN = 4, 120, 150
    GLS_V, TXT_V = 1067, 2891  # blank / special 込み
    U, N = 12, 8

    model = SignLanguageTransformer(input_size=F_IN,
                                    gloss_vocab_size=GLS_V,
                                    text_vocab_size=TXT_V)
    criterion = JointLoss(txt_pad_idx=0, blank_idx=0, lambda_r=5.0, lambda_t=1.0, smoothing=0.0)

    sgn = torch.randn(B, T, F_IN)
    sgn_lengths = torch.tensor([120, 100, 90, 60])
    # テキスト: 入力は <bos> w_1..w_{U-1}, ターゲットは w_1..w_{U-1} <eos>
    txt = torch.randint(4, TXT_V, (B, U))
    txt_input = torch.cat([torch.full((B, 1), 1), txt[:, :-1]], dim=1)  # bos=1
    txt_target = torch.cat([txt[:, :-1], torch.full((B, 1), 2)], dim=1)  # eos=2
    gls = torch.randint(1, GLS_V, (B, N))
    gls_lengths = torch.tensor([8, 7, 6, 4])

    gloss_log_probs, word_logits = model(sgn, sgn_lengths, txt_input)
    print("gloss_log_probs:", tuple(gloss_log_probs.shape))  # (T, B, GLS_V)
    print("word_logits    :", tuple(word_logits.shape))      # (B, U, TXT_V)

    total, lr_, lt_ = criterion(gloss_log_probs, sgn_lengths, gls, gls_lengths,
                                word_logits, txt_target, batch_size=B)
    print(f"loss: total={total.item():.3f}  L_R={lr_.item():.3f}  L_T={lt_.item():.3f}")
    total.backward()
    print("backward OK")

    model.eval()

    # --- ctc_prefix_beam_search の正当性チェック ---------------------------
    # 古典的な例: 2 フレーム，語彙 {blank, A}, 各フレーム p=[0.6, 0.4]
    #   greedy (best path) → blank,blank → "" (P=0.36)
    #   周辺化: P("A") = 0.4*0.4 + 0.4*0.6 + 0.6*0.4 = 0.64 > P("")
    toy = torch.tensor([[0.6, 0.4], [0.6, 0.4]]).log()
    from sign_language_transformer import ctc_prefix_beam_search as _cbs  # noqa
    assert ctc_prefix_beam_search(toy, blank=0, beam_size=4, prune_topk=2) == [1], \
        "beam search は周辺化により 'A' を返すべき"
    print("toy beam test: OK (greedy='' に対し beam=[A] を正しく選択)")

    print("CTC greedy   :", model.recognize_greedy(sgn, sgn_lengths)[0][:10])
    print("CTC beam     :", model.recognize_beam(sgn, sgn_lengths, beam_size=5)[0][:10])
    try:
        ta = model.recognize_beam_torchaudio(sgn, sgn_lengths, beam_size=5)
        print("CTC beam (ta):", ta[0][:10])
    except ImportError:
        print("CTC beam (ta): torchaudio/flashlight-text 未インストールのためスキップ")
    print("greedy trans :", model.translate_greedy(sgn, sgn_lengths, max_len=8)[0].tolist())
    print("beam trans   :", model.translate_beam(sgn[:1], sgn_lengths[:1], beam_size=3, max_len=8)[0])