"""
Neural Sign Language Translation (Camgoz et al., CVPR 2018) の PyTorch 実装
==========================================================================
入力を CNN 特徴ではなく骨格座標 (B, T, 150) とした Sign2Text (S2T) 版。

論文の設定を再現:
  - Spatial Embedding: 論文では AlexNet だが、ここでは骨格座標 150 次元を
    線形射影で埋め込む (式(1) の SpatialEmbedding に相当)
  - Word Embedding: one-hot からの線形射影 = nn.Embedding (式(2))
  - Encoder/Decoder: 4 層スタック residual GRU、隠れ 1000 次元
    (論文 5 節: "four stacked layers of residual recurrent units",
     GRU が LSTM より良好: Table 2)
  - 入力系列の時間反転 (Sutskever et al. に従う、3.3 節)
  - Attention: Luong (乗算型) / Bahdanau (連結型)、式(7)-(10)
  - Input feeding: attention ベクトル a_{u-1} を次ステップへ入力 (式(11))
  - 学習: Adam lr=1e-5, gradient clipping 5.0, dropout 0.2
  - 推論: ビームサーチ (ビーム幅 3 が最適: Figure 3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """(B,) の系列長から (B, max_len) の bool マスク (True = 有効フレーム)。"""
    ar = torch.arange(max_len, device=lengths.device)
    return ar.unsqueeze(0) < lengths.unsqueeze(1)


def reverse_padded_sequence(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """パディングを保ったまま各系列の有効部分だけを時間反転する。

    論文 3.3 節: "we first reverse its order in the temporal domain,
    as suggested by [Sutskever et al.]"
    x: (B, T, D), lengths: (B,)
    """
    B, T, _ = x.shape
    idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # (B, T)
    rev = lengths.unsqueeze(1) - 1 - idx          # 有効部分の反転 index
    rev = torch.where(idx < lengths.unsqueeze(1), rev, idx)  # パディングはそのまま
    return x.gather(1, rev.unsqueeze(-1).expand_as(x))


# ---------------------------------------------------------------------------
# 埋め込み層 (3.1 節)
# ---------------------------------------------------------------------------

class SkeletonEmbedding(nn.Module):
    """式(1): f_t = SpatialEmbedding(x_t)

    論文では 2D CNN (AlexNet) だが、入力が骨格座標 (150 = 50 関節 × xyz など)
    のため、フレーム毎の線形射影 + 非線形で置き換える。
    """

    def __init__(self, in_dim: int = 150, embed_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, 150) -> (B, T, E)
        return self.net(x)


class WordEmbedding(nn.Module):
    """式(2): g_u = WordEmbedding(y_u)  (one-hot への線形射影 = Embedding)"""

    def __init__(self, vocab_size: int, embed_dim: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)

    def forward(self, y: torch.Tensor) -> torch.Tensor:  # (B, U) -> (B, U, E)
        return self.embed(y)


# ---------------------------------------------------------------------------
# Attention (3.3 節, 式(7)-(10))
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Luong (乗算型) / Bahdanau (連結型) attention。

    score(h_u, o_n) =
        h_u^T W o_n                      [Luong / multiplicative]
        V^T tanh(W [h_u; o_n])           [Bahdanau / concat]
    """

    def __init__(self, hidden_dim: int, mode: str = "luong"):
        super().__init__()
        assert mode in ("luong", "bahdanau", "none")
        self.mode = mode
        if mode == "luong":
            self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif mode == "bahdanau":
            self.W = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
            self.V = nn.Linear(hidden_dim, 1, bias=False)
        # 式(10): a_u = tanh(W_c [c_u; h_u])
        self.Wc = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        h: torch.Tensor,            # (B, H)   デコーダ隠れ状態 h_u
        enc_out: torch.Tensor,      # (B, N, H) エンコーダ出力 o_{1:N}
        enc_mask: torch.Tensor,     # (B, N)   True = 有効
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "none":
            # attention なし: 文脈ベクトルの代わりに h をそのまま使う
            return torch.tanh(self.Wc(torch.cat([h, h], dim=-1))), None

        if self.mode == "luong":
            score = torch.bmm(self.W(enc_out), h.unsqueeze(-1)).squeeze(-1)  # (B, N)
        else:  # bahdanau
            N = enc_out.size(1)
            hx = h.unsqueeze(1).expand(-1, N, -1)                 # (B, N, H)
            score = self.V(torch.tanh(self.W(torch.cat([hx, enc_out], -1)))).squeeze(-1)

        score = score.masked_fill(~enc_mask, float("-inf"))
        gamma = torch.softmax(score, dim=-1)                       # 式(8)
        c = torch.bmm(gamma.unsqueeze(1), enc_out).squeeze(1)      # 式(7) (B, H)
        a = torch.tanh(self.Wc(torch.cat([c, h], dim=-1)))         # 式(10)
        return a, gamma


# ---------------------------------------------------------------------------
# Encoder: 4 層 residual GRU (5 節)
# ---------------------------------------------------------------------------

class ResidualGRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1000,
                 num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.GRU(in_dim, hidden_dim, batch_first=True))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, z: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """z: (B, N, E) [反転済み] -> (enc_out (B,N,H), h_n (L,B,H))"""
        packed_lengths = lengths.cpu()
        out = z
        finals = []
        for i, gru in enumerate(self.layers):
            packed = nn.utils.rnn.pack_padded_sequence(
                out, packed_lengths, batch_first=True, enforce_sorted=False)
            y, h_n = gru(packed)
            y, _ = nn.utils.rnn.pad_packed_sequence(
                y, batch_first=True, total_length=z.size(1))
            # residual 接続 (次元が一致する 2 層目以降)
            out = y + out if (i > 0) else y
            out = self.dropout(out)
            finals.append(h_n)  # (1, B, H): 各系列の最終有効ステップ = h_sign
        return out, torch.cat(finals, dim=0)  # (L, B, H)


# ---------------------------------------------------------------------------
# Decoder: 4 層 residual GRUCell + attention + input feeding
# ---------------------------------------------------------------------------

class AttnResidualGRUDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 512,
                 hidden_dim: int = 1000, num_layers: int = 4,
                 dropout: float = 0.2, attention: str = "luong"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.word_embed = WordEmbedding(vocab_size, embed_dim)
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            # 式(11): 入力は [g_{u-1}; a_{u-1}] (input feeding)
            in_dim = embed_dim + hidden_dim if i == 0 else hidden_dim
            self.cells.append(nn.GRUCell(in_dim, hidden_dim))
        self.attn = Attention(hidden_dim, attention)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)  # 式(6) を出す FC
        self.dropout = nn.Dropout(dropout)

    def step(
        self,
        y_prev: torch.Tensor,                 # (B,)
        h: list[torch.Tensor],                # 各層 (B, H)
        a_prev: torch.Tensor,                 # (B, H)
        enc_out: torch.Tensor,
        enc_mask: torch.Tensor,
    ):
        g = self.word_embed(y_prev)                        # (B, E)
        x = torch.cat([g, a_prev], dim=-1)                 # input feeding
        new_h = []
        for i, cell in enumerate(self.cells):
            hi = cell(x, h[i])
            hi = self.dropout(hi)
            x = (hi + x) if (i > 0) else hi                # residual (2 層目以降)
            new_h.append(hi)
        a, gamma = self.attn(x, enc_out, enc_mask)         # h_u として最上層出力
        logits = self.out_proj(a)                          # (B, V)
        return logits, new_h, a, gamma

    def init_state(self, h_enc: torch.Tensor):
        """h_0 = h_sign (3.3 節)。各層の最終隠れ状態でデコーダを初期化。"""
        h = [h_enc[i] for i in range(self.num_layers)]     # list of (B, H)
        B = h_enc.size(1)
        a0 = h_enc.new_zeros(B, self.hidden_dim)
        return h, a0


# ---------------------------------------------------------------------------
# 損失関数 (式(6) の順序付き条件付き確率に対する単語毎 cross entropy)
# ---------------------------------------------------------------------------

class Seq2SeqCrossEntropyLoss(nn.Module):
    """Sign2Text / Gloss2Text 用の系列 cross entropy。

    論文 3.3 節: "which is used to calculate the errors by applying
    cross entropy loss for each word"

    - PAD 位置をマスクし、有効トークン数で正規化 (バッチ内の系列長差に不変)
    - label_smoothing > 0 でラベル平滑化 (論文にはないが一般的な拡張)
    - reduction="sum" にすると系列毎の合計 (perplexity 計算などに)
    """

    def __init__(self, pad_id: int = PAD_ID, label_smoothing: float = 0.0,
                 reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.pad_id = pad_id
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,    # (B, U, V)
        targets: torch.Tensor,   # (B, U)
    ) -> torch.Tensor:
        B, U, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            ignore_index=self.pad_id,
            label_smoothing=self.label_smoothing,
            reduction="none",
        ).view(B, U)                                     # (B, U)

        mask = targets.ne(self.pad_id).float()           # (B, U)
        loss = loss * mask
        if self.reduction == "mean":
            return loss.sum() / mask.sum().clamp(min=1.0)
        if self.reduction == "sum":
            return loss.sum(dim=1)                       # (B,) 系列毎の NLL
        return loss                                      # (B, U)

    @torch.no_grad()
    def perplexity(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """訓練の収束判定用 (論文 5 節: "trained until the training
        perplexity is converged")。トークン平均 NLL の exp。"""
        mask = targets.ne(self.pad_id).float()
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.pad_id,
            reduction="sum",
        )
        return torch.exp(nll / mask.sum().clamp(min=1.0))


# ---------------------------------------------------------------------------
# Sign2Text モデル本体
# ---------------------------------------------------------------------------

@dataclass
class NSLTConfig:
    input_dim: int = 144         # 骨格座標次元 (例: 50 keypoints × (x,y,z))
    embed_dim: int = 512
    hidden_dim: int = 1000        # 論文: 1000 hidden units
    num_layers: int = 4           # 論文: 4 stacked residual layers
    dropout: float = 0.2          # 論文: drop probability 0.2
    attention: str = "luong"      # 論文: Luong が最良 (Table 3/5)
    vocab_size: int = 2890        # PHOENIX14T ドイツ語語彙 2887 + 特殊トークン
    label_smoothing: float = 0.0  # 論文は 0 (plain cross entropy)


class Sign2Text(nn.Module):
    """骨格系列 (B, T, 150) → 音声言語文の seq2seq 翻訳モデル。

    src_embed を差し替えることで G2T (グロス ID 列 → 文) にも流用できる。
    """

    def __init__(self, cfg: NSLTConfig, src_embed: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.spatial_embed = src_embed if src_embed is not None else \
            SkeletonEmbedding(cfg.input_dim, cfg.embed_dim, cfg.dropout)
        self.encoder = ResidualGRUEncoder(cfg.embed_dim, cfg.hidden_dim,
                                          cfg.num_layers, cfg.dropout)
        self.decoder = AttnResidualGRUDecoder(cfg.vocab_size, cfg.embed_dim,
                                              cfg.hidden_dim, cfg.num_layers,
                                              cfg.dropout, cfg.attention)
        self.criterion = Seq2SeqCrossEntropyLoss(
            pad_id=PAD_ID, label_smoothing=cfg.label_smoothing)

    # ---- 共通: エンコード ----
    def encode(self, x: torch.Tensor, x_len: torch.Tensor):
        z = self.spatial_embed(x)                       # 式(1)
        z = reverse_padded_sequence(z, x_len)           # 入力反転 (3.3 節)
        enc_out, h_enc = self.encoder(z, x_len)         # 式(4)
        enc_mask = make_pad_mask(x_len, x.size(1))
        return enc_out, enc_mask, h_enc

    # ---- 学習: teacher forcing + cross entropy (式(6)) ----
    def forward(
        self,
        x: torch.Tensor,        # (B, T, 150)
        x_len: torch.Tensor,    # (B,)
        y: torch.Tensor,        # (B, U+2)  [BOS, w1..wU, EOS] (PAD 埋め)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc_out, enc_mask, h_enc = self.encode(x, x_len)
        h, a = self.decoder.init_state(h_enc)

        y_in, y_out = y[:, :-1], y[:, 1:]
        logits_all = []
        for u in range(y_in.size(1)):
            logits, h, a, _ = self.decoder.step(y_in[:, u], h, a, enc_out, enc_mask)
            logits_all.append(logits)
        logits_all = torch.stack(logits_all, dim=1)     # (B, U+1, V)

        loss = self.criterion(logits_all, y_out)
        return loss, logits_all

    # ---- 推論: greedy ----
    @torch.no_grad()
    def greedy_decode(self, x, x_len, max_len: int = 50):
        self.eval()
        enc_out, enc_mask, h_enc = self.encode(x, x_len)
        h, a = self.decoder.init_state(h_enc)
        B = x.size(0)
        y = torch.full((B,), BOS_ID, dtype=torch.long, device=x.device)
        done = torch.zeros(B, dtype=torch.bool, device=x.device)
        outputs, attentions = [], []
        for _ in range(max_len):
            logits, h, a, gamma = self.decoder.step(y, h, a, enc_out, enc_mask)
            y = logits.argmax(-1)
            y = torch.where(done, torch.full_like(y, PAD_ID), y)
            outputs.append(y)
            attentions.append(gamma)
            done |= y.eq(EOS_ID)
            if done.all():
                break
        return torch.stack(outputs, dim=1), torch.stack(attentions, dim=1)

    # ---- 推論: ビームサーチ (ビーム幅 3、Figure 3) ----
    @torch.no_grad()
    def beam_search(self, x, x_len, beam_width: int = 3, max_len: int = 50,
                    length_norm: bool = True):
        """バッチサイズ 1 のビームサーチ。戻り値: 最良仮説のトークン列 (list[int])"""
        assert x.size(0) == 1, "beam_search はバッチサイズ 1 のみ対応"
        self.eval()
        enc_out, enc_mask, h_enc = self.encode(x, x_len)

        # ビーム分に複製
        enc_out = enc_out.expand(beam_width, -1, -1).contiguous()
        enc_mask = enc_mask.expand(beam_width, -1).contiguous()
        h_enc = h_enc.expand(-1, beam_width, -1).contiguous()
        h, a = self.decoder.init_state(h_enc)

        seqs = torch.full((beam_width, 1), BOS_ID, dtype=torch.long, device=x.device)
        scores = torch.full((beam_width,), float("-inf"), device=x.device)
        scores[0] = 0.0  # 初期は同一状態なので 1 本のみ有効
        finished: list[tuple[float, list[int]]] = []

        for _ in range(max_len):
            logits, h_new, a_new, _ = self.decoder.step(
                seqs[:, -1], h, a, enc_out, enc_mask)
            logp = F.log_softmax(logits, dim=-1)               # (K, V)
            V = logp.size(-1)
            total = scores.unsqueeze(1) + logp                 # (K, V)
            flat = total.view(-1)
            top_scores, top_idx = flat.topk(beam_width)
            beam_idx, tok_idx = top_idx // V, top_idx % V

            seqs = torch.cat([seqs[beam_idx], tok_idx.unsqueeze(1)], dim=1)
            scores = top_scores
            h = [hi[beam_idx] for hi in h_new]
            a = a_new[beam_idx]

            # EOS に達したビームを取り出す
            alive = tok_idx.ne(EOS_ID)
            for k in range(beam_width):
                if not alive[k]:
                    s = scores[k].item()
                    hyp = seqs[k, 1:-1].tolist()  # BOS/EOS を除去
                    norm = s / max(len(hyp), 1) if length_norm else s
                    finished.append((norm, hyp))
                    scores[k] = float("-inf")
            if len(finished) >= beam_width:
                break

        if not finished:  # EOS 未到達仮説からフォールバック
            k = scores.argmax().item()
            finished.append((scores[k].item(), seqs[k, 1:].tolist()))
        finished.sort(key=lambda t: t[0], reverse=True)
        return finished[0][1]


# ---------------------------------------------------------------------------
# Sign2Gloss (S2G): CTC ベースの連続手話認識器
# ---------------------------------------------------------------------------
#
# 論文の S2G2T では tokenizer に Koller et al. [36] の CNN-RNN-HMM
# (Re-Sign, WER 25.7%/26.6%) を使用しているが、これは外部システムであり
# 公式コードにも含まれない。ここでは HMM の現代的等価物である CTC で
# gloss 認識器を実装する (gloss はフレームと単調に整列するため CTC が適合。
# 論文 2 節でも CSLR における CTC の利用 [6, 17] に言及)。
#
# gloss 語彙の ID 0 は CTC blank として予約する。

GLOSS_BLANK_ID = 0


class Sign2Gloss(nn.Module):
    """骨格系列 (B, T, 150) → gloss 列。エンコーダ + CTC ヘッド。

    S2T と同じ SkeletonEmbedding + ResidualGRUEncoder を再利用。
    CTC は単調整列を仮定するため、seq2seq と違い入力反転は行わない。
    """

    def __init__(self, cfg: NSLTConfig, gloss_vocab_size: int = 1070):
        super().__init__()
        self.spatial_embed = SkeletonEmbedding(cfg.input_dim, cfg.embed_dim, cfg.dropout)
        self.encoder = ResidualGRUEncoder(cfg.embed_dim, cfg.hidden_dim,
                                          cfg.num_layers, cfg.dropout)
        self.gloss_head = nn.Linear(cfg.hidden_dim, gloss_vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=GLOSS_BLANK_ID, zero_infinity=True)

    def encode(self, x: torch.Tensor, x_len: torch.Tensor):
        z = self.spatial_embed(x)                        # 反転なし
        enc_out, _ = self.encoder(z, x_len)              # (B, T, H)
        return self.gloss_head(enc_out)                  # (B, T, Vg)

    def forward(
        self,
        x: torch.Tensor,            # (B, T, 150)
        x_len: torch.Tensor,        # (B,)
        gloss: torch.Tensor,        # (B, Lg)  gloss ID (PAD なし前提の連結 or PAD 埋め)
        gloss_len: torch.Tensor,    # (B,)
    ) -> torch.Tensor:
        logits = self.encode(x, x_len)                   # (B, T, Vg)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, Vg)
        return self.ctc_loss(log_probs, gloss, x_len, gloss_len)

    @torch.no_grad()
    def greedy_ctc_decode(self, x, x_len) -> list[list[int]]:
        """best-path デコード: 連続重複の縮約 + blank 除去。"""
        self.eval()
        logits = self.encode(x, x_len)                   # (B, T, Vg)
        paths = logits.argmax(-1)                        # (B, T)
        results = []
        for b in range(x.size(0)):
            path = paths[b, : x_len[b]].tolist()
            out, prev = [], None
            for p in path:
                if p != prev and p != GLOSS_BLANK_ID:
                    out.append(p)
                prev = p
            results.append(out)
        return results


class Gloss2Text(Sign2Text):
    """G2T: gloss ID 列 → 音声言語文。

    Sign2Text のソース埋め込みを nn.Embedding に差し替えただけの構成
    (論文 5.1 節の G2T / 5.3 節の S2G→G2T で使用)。
    入力 x は (B, N) の gloss ID 列。
    """

    def __init__(self, cfg: NSLTConfig, gloss_vocab_size: int = 1070):
        src_embed = nn.Embedding(gloss_vocab_size, cfg.embed_dim,
                                 padding_idx=GLOSS_BLANK_ID)
        super().__init__(cfg, src_embed=src_embed)

    def encode(self, x: torch.Tensor, x_len: torch.Tensor):
        z = self.spatial_embed(x)                        # (B, N, E)
        z = reverse_padded_sequence(z, x_len)
        enc_out, h_enc = self.encoder(z, x_len)
        enc_mask = make_pad_mask(x_len, x.size(1))
        return enc_out, enc_mask, h_enc


class Sign2Gloss2Text(nn.Module):
    """S2G2T パイプライン: S2G の認識結果を G2T に渡す 2 段構成。

    論文 5.3 節に対応。2 つのサブモデルは個別に学習し
    (S2G は CTC、G2T は gloss→文で cross entropy)、推論時に連結する。
    論文の S2G2T 設定 (推定 gloss で G2T を再学習) を再現するには、
    学習時に S2G の予測 gloss を G2T の入力として与えればよい。
    """

    def __init__(self, s2g: Sign2Gloss, g2t: Gloss2Text):
        super().__init__()
        self.s2g = s2g
        self.g2t = g2t

    @torch.no_grad()
    def translate(self, x, x_len, beam_width: int = 3, max_len: int = 50):
        gloss_seqs = self.s2g.greedy_ctc_decode(x, x_len)
        results = []
        for g in gloss_seqs:
            if len(g) == 0:          # 認識結果が空の場合のフォールバック
                results.append([])
                continue
            g_t = torch.tensor(g, dtype=torch.long, device=x.device).unsqueeze(0)
            g_len = torch.tensor([len(g)], device=x.device)
            results.append(self.g2t.beam_search(g_t, g_len,
                                                beam_width=beam_width,
                                                max_len=max_len))
        return results, gloss_seqs


# ---------------------------------------------------------------------------
# 学習ループの雛形 (5 節のハイパーパラメータ)
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, lr: float = 1e-5):
    """論文: Adam, lr=1e-5 (収束が停滞したら 1e-6 に減衰)"""
    return torch.optim.Adam(model.parameters(), lr=lr)


def train_step(model: Sign2Text, optimizer, batch, clip: float = 5.0):
    model.train()
    x, x_len, y = batch
    loss, _ = model(x, x_len, y)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip)  # 論文: threshold 5
    optimizer.step()
    return loss.item()
def eval_step(model: Sign2Text, batch):
    model.eval()
    x, x_len, y = batch
    with torch.no_grad():
        loss, _ = model(x, x_len, y)
    return loss.item()


# ---------------------------------------------------------------------------
# 動作確認
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = NSLTConfig(vocab_size=200, hidden_dim=256, embed_dim=128)  # 小型で確認
    model = Sign2Text(cfg)
    print(f"#params = {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    B, T = 2, 60
    x = torch.randn(B, T, 150)
    x_len = torch.tensor([60, 45])
    # y: [BOS, ..., EOS, PAD...]
    y = torch.tensor([
        [BOS_ID, 10, 11, 12, 13, EOS_ID, PAD_ID, PAD_ID],
        [BOS_ID, 20, 21, EOS_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID],
    ])

    opt = build_optimizer(model, lr=1e-3)  # 動作確認用に大きめ lr
    for step in range(3):
        loss = train_step(model, opt, (x, x_len, y))
        print(f"step {step}: loss = {loss:.4f}")

    with torch.no_grad():
        _, logits = model(x, x_len, y)
        ppl = model.criterion.perplexity(logits, y[:, 1:])
        print(f"train perplexity = {ppl.item():.2f}")

    hyps, attn = model.greedy_decode(x, x_len, max_len=10)
    print("greedy:", hyps.tolist())
    print("attention shape:", attn.shape)  # (B, U, N)

    best = model.beam_search(x[:1], x_len[:1], beam_width=3, max_len=10)
    print("beam(width=3):", best)

    # ---- S2G / S2G2T の動作確認 ----
    print("\n--- Sign2Gloss (CTC) ---")
    Vg = 50  # gloss 語彙 (ID 0 = blank)
    s2g = Sign2Gloss(cfg, gloss_vocab_size=Vg)
    gloss = torch.tensor([[5, 8, 3, 0], [7, 2, 0, 0]])   # PAD は blank=0 で埋め
    gloss_len = torch.tensor([3, 2])
    opt_g = build_optimizer(s2g, lr=1e-3)
    for step in range(3):
        s2g.train()
        ctc = s2g(x, x_len, gloss, gloss_len)
        opt_g.zero_grad(); ctc.backward()
        nn.utils.clip_grad_norm_(s2g.parameters(), 5.0)
        opt_g.step()
        print(f"step {step}: ctc_loss = {ctc.item():.4f}")
    print("greedy CTC decode:", s2g.greedy_ctc_decode(x, x_len))

    print("\n--- Sign2Gloss2Text ---")
    g2t = Gloss2Text(cfg, gloss_vocab_size=Vg)
    g_loss, _ = g2t(gloss, gloss_len, y)   # G2T も同じ Seq2SeqCrossEntropyLoss を使用
    print(f"G2T loss = {g_loss.item():.4f}")
    s2g2t = Sign2Gloss2Text(s2g, g2t)
    texts, glosses = s2g2t.translate(x, x_len, beam_width=3, max_len=10)
    print("recognized glosses:", glosses)
    print("translations:", texts)