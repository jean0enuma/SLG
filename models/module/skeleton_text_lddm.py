"""
Text-conditioned Latent Denoising Diffusion Model (LDDM)
========================================================

config.mode によって 2 つの学習を切り替える:

  - "vae"  : SkeletonVAE / SkeletonGCNVAE の学習。skeleton (B, T, F) を潜在空間に圧縮する。
             損失 = 再構成損失 (masked MSE) + KL(q(z|x) || N(0, I))

  - "lddm" : 学習済み VAE の潜在空間上での text 条件付き Latent Diffusion。
             text の入れ方は config.conditioning_type で切り替える。損失は潜在トークン
             位置のノイズ MSE のみ。

VAE バックボーンは config.vae_backbone で切り替えられる:
  - "transformer" : Transformer ベースの系列 VAE
  - "gcn"         : ST-GCN ベースの VAE (骨格のグラフ構造を利用)

text encoder は config.text_encoder_type で切り替えられる:
  - "pretrained" : HuggingFace の学習済みエンコーダ (AutoModel)。隠れ次元を d_model へ射影。
  - "scratch"    : 語彙埋め込み + Transformer をスクラッチ構成。最初から d_model で出力。

text 条件付け (件付け) の入れ方は config.conditioning_type で切り替えられる:
  - "in_context"      : text トークンを潜在(motion)トークンに連結し、self-attention
                        (TransformerEncoder) で条件付け。motion 位置の出力のみ取り出す。
  - "cross_attention" : motion を query、text を key/value とする cross-attention
                        (TransformerDecoder) で条件付け。

すべてのサブモデルは Transformer / GCN をベースに構成している。
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    # ---- データ次元 ----
    skeleton_dim: int = 150      # F  : skeleton 特徴次元 (= num_joints * joint_coord_dim)
    skeleton_hand_dim: int =126
    skeleton_body_dim: int =24
    max_seq_len: int = 196       # T  : skeleton の最大系列長

    # ---- VAE バックボーン ----
    vae_backbone: str = "transformer"   # "transformer" | "gcn"
    joint_coord_dim: int = 3            # GCN: 1関節あたりの座標次元 (F = num_joints * coord)
    num_joints: int = None              # None なら skeleton_dim // joint_coord_dim から決定
    gcn_temporal_kernel: int = 5        # GCN の時間畳み込みカーネル長

    # ---- Text encoder ----
    condition_on_text: bool = True                # False で無条件 LDDM (text encoder 不要)
    text_encoder_type: str = "pretrained"         # "pretrained" | "scratch"

    # pretrained (HuggingFace) 用
    text_encoder_name: str = "bert-base-uncased"  # AutoModel.from_pretrained に渡す名前
    freeze_text_encoder: bool = True              # 事前学習済みエンコーダを凍結するか

    # scratch (自前 Transformer) 用
    text_vocab_size: int = 30522                  # tokenizer の語彙数に合わせる (重要)
    text_num_layers: int = 4                      # text encoder の層数
    text_nhead: int = 8                           # text encoder のヘッド数
    text_dim_feedforward: int = 1024              # text encoder の FFN 次元
    text_max_len: int = 64                        # text トークンの最大長
    text_pad_token_id: int = 0                    # 埋め込みの padding_idx

    # ---- 件付け (text conditioning) の入れ方 ----
    conditioning_type: str = "in_context"         # "in_context" | "cross_attention"

    # ---- 潜在空間 ----
    latent_dim: int = 256

    # ---- Transformer ----
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # ---- Diffusion ----
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # ---- 学習モード ----
    mode: str = "vae"            # "vae" | "lddm"

    # ---- 損失 / 生成 ----
    kl_weight: float = 1e-4      # VAE の KL 項
    use_latent_mean: bool = True # diffusion の教師潜在に mu を使う (Trueで分散低減)
    cfg_dropout: float = 0.1     # 学習時に text 条件を落とす確率 (classifier-free guidance)
    guidance_scale: float = 1.0  # 生成時の guidance 係数 (1.0 = guidance なし)


# --------------------------------------------------------------------------- #
#  共通モジュール
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """正弦波の位置エンコーディング。"""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TimestepEmbedding(nn.Module):
    """diffusion timestep t (B,) -> (B, d_model) の埋め込み。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)  # (B, d_model)


def length_to_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """skeleton_length (B,) -> padding mask (B, T)。True がパディング位置。"""
    idx = torch.arange(max_len, device=lengths.device)[None, :]
    return idx >= lengths[:, None]


def _valid_mask(pad_mask: torch.Tensor) -> torch.Tensor:
    return (~pad_mask).unsqueeze(-1).float()


def masked_mse(pred, target, pad_mask) -> torch.Tensor:
    valid = _valid_mask(pad_mask)
    se = ((pred - target) ** 2) * valid
    denom = valid.sum().clamp(min=1.0) * pred.size(-1)
    return se.sum() / denom


def masked_kl_to_standard_normal(mu, logvar, pad_mask) -> torch.Tensor:
    """KL(N(mu, var) || N(0, I))。系列のパディングをマスクして平均。"""
    valid = _valid_mask(pad_mask)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = (kl * valid).sum() / (valid.sum().clamp(min=1.0) * mu.size(-1))
    return kl


def build_transformer_encoder(cfg: ModelConfig) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=cfg.num_layers)


def build_transformer_decoder(cfg: ModelConfig) -> nn.TransformerDecoder:
    """cross-attention 条件付け用。各層 = self-attn + cross-attn + FFN を num_layers 層。"""
    layer = nn.TransformerDecoderLayer(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerDecoder(layer, num_layers=cfg.num_layers)


# --------------------------------------------------------------------------- #
#  Text encoder (pretrained / scratch 切り替え)
# --------------------------------------------------------------------------- #
class BaseTextEncoder(nn.Module):
    """全 text encoder が満たす統一インターフェース。

    forward(input_ids, attention_mask)
        input_ids:      (B, L) long   トークン ID
        attention_mask: (B, L) {0,1}  1=実トークン, 0=pad  (HF の規約)
      -> mem:      (B, L, d_model)  text トークン系列
         text_pad: (B, L) bool      True=pad  (PyTorch attention の規約)
    """

    def forward(self, input_ids, attention_mask):
        raise NotImplementedError


class PretrainedTextEncoder(BaseTextEncoder):
    """HuggingFace の学習済みエンコーダ。隠れ次元 -> d_model へ射影する。"""

    def __init__(self, cfg: ModelConfig, hf_encoder: nn.Module | None = None):
        super().__init__()
        if hf_encoder is None:
            from transformers import AutoModel  # 遅延 import
            hf_encoder = AutoModel.from_pretrained(cfg.text_encoder_name)
        self.encoder = hf_encoder
        hidden = self.encoder.config.hidden_size
        self.mem_proj = (nn.Identity() if hidden == cfg.d_model
                         else nn.Linear(hidden, cfg.d_model))

        self.frozen = cfg.freeze_text_encoder
        if self.frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, input_ids, attention_mask):
        ctx = torch.no_grad() if self.frozen else contextlib.nullcontext()
        with ctx:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mem = self.mem_proj(out.last_hidden_state)   # (B, L, d_model)
        text_pad = attention_mask == 0               # True=pad
        return mem, text_pad


class ScratchTextEncoder(BaseTextEncoder):
    """語彙埋め込み + 位置エンコーディング + TransformerEncoder をスクラッチ構成。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.text_vocab_size, cfg.d_model,
                                  padding_idx=cfg.text_pad_token_id)
        self.scale = math.sqrt(cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, cfg.text_max_len, cfg.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.text_nhead,
            dim_feedforward=cfg.text_dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.text_num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, input_ids, attention_mask):
        text_pad = attention_mask == 0                       # True=pad
        x = self.embed(input_ids) * self.scale               # (B, L, d_model)
        x = self.pos(x)
        mem = self.encoder(x, src_key_padding_mask=text_pad)
        return self.norm(mem), text_pad


def build_text_encoder(cfg: ModelConfig, hf_encoder: nn.Module | None = None):
    """config に従い text encoder を構築。condition_on_text=False なら None。"""
    if not cfg.condition_on_text:
        return None
    if isinstance(hf_encoder, BaseTextEncoder):
        return hf_encoder
    if cfg.text_encoder_type == "pretrained":
        return PretrainedTextEncoder(cfg, hf_encoder)
    elif cfg.text_encoder_type == "scratch":
        if hf_encoder is not None:
            raise ValueError(
                "text_encoder_type='scratch' では hf_encoder を注入できません。None を渡してください。"
            )
        return ScratchTextEncoder(cfg)
    else:
        raise ValueError(
            f"unknown text_encoder_type: {cfg.text_encoder_type!r} "
            f'(expected "pretrained" or "scratch")'
        )


# --------------------------------------------------------------------------- #
#  Skeleton VAE (Transformer)
# --------------------------------------------------------------------------- #
class SkeletonVAE(nn.Module):
    """Transformer ベースの系列 VAE。skeleton (B, T, F) <-> 潜在 z (B, T, latent_dim)。"""

    def __init__(self, cfg: ModelConfig,parts="all"):
        super().__init__()
        self.cfg = cfg
        if parts=="all":
            skeleton_dim=cfg.skeleton_dim
        elif parts=="body":
            skeleton_dim=cfg.skeleton_body_dim
        elif parts=="hand":
            skeleton_dim=cfg.skeleton_hand_dim
        self.in_proj = nn.Linear(cfg.skeleton_dim, cfg.d_model)
        self.enc_pos = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.encoder = build_transformer_encoder(cfg)
        self.to_mu = nn.Linear(cfg.d_model, cfg.latent_dim)
        self.to_logvar = nn.Linear(cfg.d_model, cfg.latent_dim)
        self.lat_proj = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.dec_pos = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.decoder = build_transformer_encoder(cfg)
        self.out_proj = nn.Linear(cfg.d_model, cfg.skeleton_dim)

    def encode(self, x, pad_mask):
        h = self.enc_pos(self.in_proj(x))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.to_mu(h), self.to_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, pad_mask):
        h = self.dec_pos(self.lat_proj(z))
        h = self.decoder(h, src_key_padding_mask=pad_mask)
        return self.out_proj(h)

    def forward(self, x, pad_mask):
        mu, logvar = self.encode(x, pad_mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, pad_mask)
        return recon, mu, logvar, z


# --------------------------------------------------------------------------- #
#  Skeleton GCN VAE (ST-GCN ベース)
# --------------------------------------------------------------------------- #
def _raw_sign_connections():
    """提示された接続をそのまま構築 (番号は飛び飛び)。"""
    connections = [(0, 10), (10, 11), (10, 12), (11, 13), (12, 14)]
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20), # Little finger
        (5, 9), (9, 13), (13, 17),
    ]
    return (
        connections + [(13, 17)]
        + [(i + 17, j + 17) for i, j in hand_connections] + [(14, 38)]
        + [(i + 38, j + 38) for i, j in hand_connections]
    )


def relabel_connections(edges):
    """使われている関節 index を昇順に 0..K-1 へ振り直し、連番化する。

    返り値: (連番化したエッジ(ソート済み, 重複なし), 関節数 K, 旧index->新index の対応)
    元の飛び飛び番号 (1-9,15,16 が欠番) のままだと孤立ノードができるため、
    実際に接続に現れる関節だけを連番に詰める。データの関節並びもこの順
    (body {0,10,11,12,13,14} -> 0..5, left hand 17..37 -> 6..26,
     right hand 38..58 -> 27..47) に合わせること。
    """
    nodes = sorted({n for e in edges for n in e})
    remap = {old: new for new, old in enumerate(nodes)}
    new_edges = sorted({tuple(sorted((remap[i], remap[j]))) for i, j in edges})
    return new_edges, len(nodes), remap


def default_sign_connections(return_mapping=False):
    """手話骨格の接続を連番化して返す (入力の関節順に合わせるため)。"""
    edges, num_nodes, remap = relabel_connections(_raw_sign_connections())
    if return_mapping:
        return edges, num_nodes, remap
    return edges


def sign_graph_num_joints():
    """連番化後の関節数 K を返す (config.num_joints / skeleton_dim 設定用)。"""
    return default_sign_connections(return_mapping=True)[1]


def build_normalized_adjacency(num_joints, edges):
    """対称正規化隣接行列 D^-1/2 (A+I) D^-1/2 を返す。"""
    A = torch.eye(num_joints)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    deg = A.sum(dim=1)
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0.0
    return dinv[:, None] * A * dinv[None, :]


class GraphConv(nn.Module):
    """空間グラフ畳み込み: 1x1 conv でチャネル変換し、隣接行列で関節集約。"""

    def __init__(self, cin, cout):
        super().__init__()
        self.fc = nn.Conv2d(cin, cout, kernel_size=1)

    def forward(self, x, A):
        # x: (B, Cin, T, N) -> (B, Cout, T, N)
        x = self.fc(x)
        return torch.einsum("nm,bctm->bctn", A, x)


class STGCNBlock(nn.Module):
    """空間グラフ畳み込み + 時間方向畳み込み + 残差。"""

    def __init__(self, cin, cout, kt, dropout):
        super().__init__()
        self.gcn = GraphConv(cin, cout)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=(kt, 1), padding=(kt // 2, 0)),
            nn.BatchNorm2d(cout), nn.Dropout(dropout),
        )
        self.residual = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        return self.relu(x + res)


class SkeletonGCNVAE(nn.Module):
    """ST-GCN ベースの系列 VAE。SkeletonVAE と同じインターフェース。

    入力 skeleton (B, T, F) を (B, T, N, C)=(関節, 座標) に並べ替えて
    空間(グラフ)+時間の畳み込みで符号化し、フレームごとの潜在 (B, T, latent_dim) を得る。
    F = num_joints * joint_coord_dim で、関節 major のレイアウト
    (j0c0, j0c1, ..., j1c0, ...) を仮定する (forward の reshape(B,T,J,C) と整合)。
    """

    def __init__(self, cfg: ModelConfig, edges=None):
        super().__init__()
        self.cfg = cfg
        C = cfg.joint_coord_dim
        N = cfg.num_joints if cfg.num_joints is not None else cfg.skeleton_dim // C
        assert cfg.skeleton_dim == N * C, (
            f"skeleton_dim({cfg.skeleton_dim}) != num_joints({N}) * joint_coord_dim({C})。"
            " num_joints / joint_coord_dim をデータに合わせて設定してください。"
        )
        self.N, self.C = N, C

        edges = default_sign_connections() if edges is None else sorted(edges)
        max_idx = max(max(e) for e in edges)
        assert max_idx < N, (
            f"グラフは関節 index {max_idx} までを要求しますが num_joints={N} です。"
            f" この接続には num_joints>={max_idx + 1} が必要です。"
        )
        self.register_buffer("A", build_normalized_adjacency(N, edges))

        h = cfg.d_model
        kt = cfg.gcn_temporal_kernel
        L = cfg.num_layers

        # encoder
        self.enc_in = nn.Conv2d(C, h, kernel_size=1)
        self.enc_blocks = nn.ModuleList([STGCNBlock(h, h, kt, cfg.dropout) for _ in range(L)])
        self.to_mu = nn.Linear(h, cfg.latent_dim)
        self.to_logvar = nn.Linear(h, cfg.latent_dim)

        # decoder
        self.dec_lat = nn.Linear(cfg.latent_dim, h)
        self.joint_emb = nn.Parameter(torch.randn(N, h) * 0.02)  # 関節ごとの埋め込み
        self.dec_blocks = nn.ModuleList([STGCNBlock(h, h, kt, cfg.dropout) for _ in range(L)])
        self.dec_out = nn.Conv2d(h, C, kernel_size=1)

    def _to_grid(self, x):
        # (B,T,F) -> (B,C,T,N)
        B, T, _ = x.shape
        return x.view(B, T, self.N, self.C).permute(0, 3, 1, 2).contiguous()

    def encode(self, x, pad_mask=None):
        g = self._to_grid(x)                       # (B,C,T,N)
        if pad_mask is not None:                   # パディングフレームを0に
            g = g * (~pad_mask).float()[:, None, :, None]
        h = self.enc_in(g)
        for blk in self.enc_blocks:
            h = blk(h, self.A)
        h = h.mean(dim=-1)                         # 関節方向に平均プール -> (B,h,T)
        h = h.permute(0, 2, 1)                      # (B,T,h)
        return self.to_mu(h), self.to_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, pad_mask=None):
        B, T, _ = z.shape
        f = self.dec_lat(z)                         # (B,T,h)
        h = f.permute(0, 2, 1).unsqueeze(-1)        # (B,h,T,1)
        h = h.expand(-1, -1, -1, self.N).contiguous()           # (B,h,T,N)
        h = h + self.joint_emb.t()[None, :, None, :]            # 関節埋め込み
        for blk in self.dec_blocks:
            h = blk(h, self.A)
        out = self.dec_out(h)                       # (B,C,T,N)
        out = out.permute(0, 2, 3, 1).contiguous().view(B, T, self.N * self.C)
        return out                                  # (B,T,F)

    def forward(self, x, pad_mask=None):
        mu, logvar = self.encode(x, pad_mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, pad_mask)
        return recon, mu, logvar, z


# --------------------------------------------------------------------------- #
#  Gaussian Diffusion (schedule)
# --------------------------------------------------------------------------- #
class GaussianDiffusion(nn.Module):
    """前向き過程の係数を保持する。buffer なので .to(device) で一緒に移動する。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_timesteps = cfg.num_timesteps
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def _gather(self, buf, t):
        return buf[t][:, None, None]

    def q_sample(self, z0, t, noise):
        return self._gather(self.sqrt_alpha_bars, t) * z0 + \
            self._gather(self.sqrt_one_minus_alpha_bars, t) * noise


# --------------------------------------------------------------------------- #
#  Text-conditioned Latent Denoiser (in-context / cross-attention 切り替え)
# --------------------------------------------------------------------------- #
class TextConditionedDenoiser(nn.Module):
    """ノイズ付き潜在 z_t と timestep t を受け取り、ノイズ epsilon を予測する Transformer。

    text 条件 (件付け) の入れ方は cfg.conditioning_type で切り替える:
      - "in_context"      : [text; motion] を連結し self-attention (TransformerEncoder)。
                            motion 位置の出力のみ取り出す。
      - "cross_attention" : motion を query、text を key/value とする cross-attention
                            (TransformerDecoder)。学習可能な null トークンを memory に
                            常に1つ加えることで、text を全て落とした場合 (無条件 / CFG)
                            でも memory が空にならず、attention の NaN を避ける。
    """

    def __init__(self, cfg: ModelConfig, encoder: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        self.condition_on_text = cfg.condition_on_text
        self.conditioning_type = cfg.conditioning_type

        # ---- 潜在 (motion) 側 ----
        self.lat_in = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.lat_pos = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.t_emb = TimestepEmbedding(cfg.d_model)
        self.type_emb = nn.Embedding(2, cfg.d_model)  # 0=text, 1=motion

        # ---- text 側 (pretrained / scratch を config で切り替え) ----
        if self.condition_on_text:
            self.text_encoder = build_text_encoder(cfg, hf_encoder=encoder)

        # ---- 本体 (in_context / cross_attention を config で切り替え) ----
        if self.conditioning_type == "in_context":
            self.net = build_transformer_encoder(cfg)
        elif self.conditioning_type == "cross_attention":
            self.net = build_transformer_decoder(cfg)
            # cross-attention 用の学習可能な null memory トークン。
            # text を全マスクしたとき (無条件 / CFG) の拠り所になり、
            # memory が空になって softmax が NaN になるのを防ぐ。
            self.null_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        else:
            raise ValueError(
                f"unknown conditioning_type: {cfg.conditioning_type!r} "
                f'(expected "in_context" or "cross_attention")'
            )

        self.out_proj = nn.Linear(cfg.d_model, cfg.latent_dim)

    def _encode_text(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)

    def _embed_motion(self, z_t, t):
        """潜在 (motion) 側の入力埋め込み。timestep / 位置 / type を加える。"""
        B, T, _ = z_t.shape
        dev = z_t.device
        m = self.lat_in(z_t)
        m = m + self.t_emb(t).unsqueeze(1)
        m = self.lat_pos(m)
        m = m + self.type_emb(torch.ones(B, T, dtype=torch.long, device=dev))
        return m

    def forward(self, z_t, t, input_ids=None, attention_mask=None,
                skel_pad_mask=None, drop_text=None):
        B, T, _ = z_t.shape
        dev = z_t.device
        if skel_pad_mask is None:
            skel_pad_mask = torch.zeros(B, T, dtype=torch.bool, device=dev)

        m = self._embed_motion(z_t, t)
        use_text = self.condition_on_text and (input_ids is not None)

        if self.conditioning_type == "in_context":
            h_motion = self._forward_in_context(
                m, T, use_text, input_ids, attention_mask, skel_pad_mask, drop_text)
        else:  # "cross_attention"
            h_motion = self._forward_cross_attention(
                m, use_text, input_ids, attention_mask, skel_pad_mask, drop_text)
        return self.out_proj(h_motion)

    # ---- in-context (連結 + self-attention) ----
    def _forward_in_context(self, m, T, use_text, input_ids, attention_mask,
                            skel_pad_mask, drop_text):
        if not use_text:
            return self.net(m, src_key_padding_mask=skel_pad_mask)

        B, dev = m.size(0), m.device
        text, text_pad = self._encode_text(input_ids, attention_mask)
        text = text + self.type_emb(
            torch.zeros(B, text.size(1), dtype=torch.long, device=dev))
        if drop_text is not None:
            text_pad = text_pad | drop_text[:, None]

        tokens = torch.cat([text, m], dim=1)
        pad = torch.cat([text_pad, skel_pad_mask], dim=1)
        h = self.net(tokens, src_key_padding_mask=pad)
        return h[:, -T:]

    # ---- cross-attention (motion=query, text=key/value) ----
    def _forward_cross_attention(self, m, use_text, input_ids, attention_mask,
                                 skel_pad_mask, drop_text):
        B, dev = m.size(0), m.device

        # memory には常に学習可能な null トークンを1つ加える。
        # これにより text を全マスク (無条件 / CFG) しても memory が空にならず、
        # cross-attention の softmax が NaN になるのを防ぐ。
        null = self.null_token.expand(B, 1, -1)               # (B, 1, d)
        null_pad = torch.zeros(B, 1, dtype=torch.bool, device=dev)

        if use_text:
            text, text_pad = self._encode_text(input_ids, attention_mask)
            text = text + self.type_emb(
                torch.zeros(B, text.size(1), dtype=torch.long, device=dev))
            if drop_text is not None:
                text_pad = text_pad | drop_text[:, None]
            memory = torch.cat([null, text], dim=1)           # (B, 1+L, d)
            mem_pad = torch.cat([null_pad, text_pad], dim=1)  # (B, 1+L)
        else:
            memory, mem_pad = null, null_pad

        return self.net(
            tgt=m,
            memory=memory,
            tgt_key_padding_mask=skel_pad_mask,
            memory_key_padding_mask=mem_pad,
        )


# --------------------------------------------------------------------------- #
#  統合モデル
# --------------------------------------------------------------------------- #
class SkeletonTextDiffusion(nn.Module):
    """config を引数に取り、mode に応じてサブモデルを構築・学習を切り替える。"""

    def __init__(self, config: ModelConfig, text_encoder: nn.Module | None = None):
        super().__init__()
        assert config.mode in ("vae", "lddm"), f"unknown mode: {config.mode}"
        self.cfg = config
        self.mode = config.mode

        self.vae = (
            SkeletonVAE(config)
        )
        self.hand_vae=(
            SkeletonVAE(config)
        )
        self.body_vae=(
            SkeletonVAE(config)
        )
        self.diffusion = GaussianDiffusion(config)
        self.denoiser = (
            TextConditionedDenoiser(config, encoder=text_encoder)
            if self.mode == "lddm" else None
        )

        if self.mode == "lddm":
            self.freeze_vae()

        self.register_buffer("latent_shift", torch.zeros(1, 1, config.latent_dim))
        self.register_buffer("latent_scale", torch.ones(1, 1, config.latent_dim))
        self.register_buffer("latent_normalized", torch.tensor(False))

    # ---------------- VAE ユーティリティ ---------------- #
    def freeze_vae(self):
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

    def load_vae(self, state_dict_path: str, map_location="cpu"):
        """SkeletonTextDiffusion *全体* の重みパスを受け取り、VAE 部分だけ self.vae に読み込む。"""
        ckpt = torch.load(state_dict_path, map_location=map_location)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            full_sd = ckpt["model_state_dict"]
        else:
            full_sd = ckpt
        full_sd = {k[len("module."):] if k.startswith("module.") else k: v
                   for k, v in full_sd.items()}
        vae_sd = {k[len("vae."):]: v for k, v in full_sd.items() if k.startswith("vae.")}
        if not vae_sd:
            raise KeyError(
                f"'vae.' で始まる重みが {state_dict_path} に見つかりません。"
                " SkeletonTextDiffusion 全体の重みを指定してください。"
            )
        self.vae.load_state_dict(vae_sd, strict=True)
        if self.mode == "lddm":
            self.freeze_vae()

    # ---------------- 潜在の分散1正規化 ---------------- #
    @torch.no_grad()
    def _raw_latent(self, skeleton, pad_mask):
        mu, logvar = self.vae.encode(skeleton, pad_mask)
        return mu if self.cfg.use_latent_mean else self.vae.reparameterize(mu, logvar)

    def _normalize_latent(self, z):
        return (z - self.latent_shift) / self.latent_scale

    def _denormalize_latent(self, z):
        return z * self.latent_scale + self.latent_shift

    @torch.no_grad()
    def compute_latent_stats(self, dataloader, device="cpu", per_dim=False,
                             max_batches=None, standardize=True, set_buffers=True):
        """学習集合で VAE 潜在 (mu) の平均・標準偏差を測り、表示し、正規化バッファを設定する。"""
        self.vae.eval()
        D = self.cfg.latent_dim
        n = torch.zeros(1, device=device)
        s = torch.zeros(D, device=device)
        ss = torch.zeros(D, device=device)

        for bi, batch in enumerate(dataloader):
            if max_batches is not None and bi >= max_batches:
                break
            if isinstance(batch, dict):
                skel = batch["skeleton"].float().to(device)
                length = batch["skeleton_length"].to(device)
            else:
                skel = batch[0].float().to(device)
                length = batch[2].to(device)
            B, T = skel.shape[0], skel.shape[1]
            skel = skel.reshape(B, T, -1)          # (B,T,J,C) でも (B,T,F) でも可
            pad_mask = length_to_padding_mask(length, T)
            z = self._raw_latent(skel, pad_mask)
            valid = (~pad_mask).unsqueeze(-1).float()
            zb = z * valid
            n += valid.sum()
            s += zb.sum(dim=(0, 1))
            ss += (zb * z).sum(dim=(0, 1))

        mean = s / n.clamp(min=1.0)
        var = (ss / n.clamp(min=1.0)) - mean ** 2
        std = var.clamp(min=1e-12).sqrt()

        print(f"[latent stats] valid tokens = {int(n.item())}, dim = {D}")
        print(f"  per-dim std : min={std.min():.4f}  median={std.median():.4f}  "
              f"mean={std.mean():.4f}  max={std.max():.4f}")
        print(f"  per-dim mean: min={mean.min():.4f}  median={mean.median():.4f}  "
              f"max={mean.max():.4f}")
        global_std = std.pow(2).mean().sqrt()
        print(f"  global (scalar) std = {global_std:.4f}  -> "
              f"{'1 から大きく外れている (正規化推奨)' if abs(global_std-1) > 0.3 else 'ほぼ 1'}")

        if set_buffers:
            if per_dim:
                scale = std.view(1, 1, D)
                shift = mean.view(1, 1, D) if standardize else torch.zeros(1, 1, D, device=device)
            else:
                scale = global_std.view(1, 1, 1).expand(1, 1, D).contiguous()
                shift = (mean.mean().view(1, 1, 1).expand(1, 1, D).contiguous()
                         if standardize else torch.zeros(1, 1, D, device=device))
            self.latent_shift.copy_(shift.to(self.latent_shift.device))
            self.latent_scale.copy_(scale.to(self.latent_scale.device))
            self.latent_normalized.copy_(torch.tensor(True))
            print(f"  -> 正規化バッファを設定 (per_dim={per_dim}, standardize={standardize})")
        return mean, std

    @torch.no_grad()
    def _encode_latent(self, skeleton, pad_mask):
        z = self._raw_latent(skeleton, pad_mask)
        return self._normalize_latent(z)

    # ---------------- 損失計算 ---------------- #
    def compute_loss(self, skeleton,hand_skeleton,body_skeleton, input_ids=None, attention_mask=None, skeleton_length=None):
        B, T = skeleton.shape[0], skeleton.shape[1]
        if skeleton_length is None:
            skeleton_length = torch.full((B,), T, device=skeleton.device, dtype=torch.long)
        pad_mask = length_to_padding_mask(skeleton_length, T)
        left_hand_skeleton,right_hand_skeleton=torch.chunk(hand_skeleton,2,dim=-1)

        if self.mode == "vae":
            recon, mu, logvar, _ = self.vae(skeleton, pad_mask)
            recon_left_hand,mu_left_hand,logvar_left_hand=self.hand_vae(left_hand_skeleton, pad_mask)
            recon_right_hand,mu_right_hand,logvar_right_hand=self.hand_vae(right_hand_skeleton, pad_mask)
            recon_body,mu_body,logvar_body=self.body_vae(body_skeleton, pad_mask)
            recon_loss = masked_mse(recon, skeleton, pad_mask)
            recon_hand_loss=(masked_mse(recon_right_hand,hand_skeleton,pad_mask)+masked_mse(recon_left_hand,hand_skeleton,pad_mask))/2
            recon_body_loss=masked_mse(recon_body,body_skeleton,pad_mask)
            kl = masked_kl_to_standard_normal(mu, logvar, pad_mask)
            kl_hand=(masked_kl_to_standard_normal(mu_left_hand,logvar_left_hand,pad_mask)+masked_kl_to_standard_normal(mu_right_hand,logvar_right_hand,pad_mask))/2
            kl_body=masked_kl_to_standard_normal(mu_body,logvar_body,pad_mask)
            loss = (recon_loss + recon_hand_loss + recon_body_loss) + self.cfg.kl_weight * (kl + kl_hand + kl_body)
            return {"loss": loss, "recon": recon_loss.detach(), "kl": kl.detach()}

        z0 = self._encode_latent(skeleton, pad_mask)
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=skeleton.device)
        noise = torch.randn_like(z0)
        z_t = self.diffusion.q_sample(z0, t, noise)

        drop_text = None
        if self.cfg.condition_on_text and self.cfg.cfg_dropout > 0 and input_ids is not None:
            drop_text = torch.rand(B, device=skeleton.device) < self.cfg.cfg_dropout

        pred_noise = self.denoiser(z_t, t, input_ids, attention_mask, pad_mask, drop_text)
        loss = masked_mse(pred_noise, noise, pad_mask)
        return {"loss": loss}

    def forward(self, skeleton, input_ids=None, attention_mask=None, skeleton_length=None):
        # (B,T,J,C) なら (B,T,J*C) に平坦化。(B,T,F) はそのまま。
        if skeleton.dim() == 4:
            B, T, J, C = skeleton.shape
            skeleton = skeleton.reshape(B, T, J * C)
        return self.compute_loss(skeleton, input_ids, attention_mask, skeleton_length)

    # ---------------- 生成 ---------------- #
    def _predict_eps(self, z, t, input_ids, attention_mask, pad_mask, guidance_scale, do_cfg):
        eps = self.denoiser(z, t, input_ids, attention_mask, pad_mask)
        if do_cfg:
            drop = torch.ones(z.size(0), dtype=torch.bool, device=z.device)
            eps_u = self.denoiser(z, t, input_ids, attention_mask, pad_mask, drop)
            eps = eps_u + guidance_scale * (eps - eps_u)
        return eps

    def _init_sampling(self, seq_len, input_ids, skeleton_length, guidance_scale, device):
        assert self.denoiser is not None, "sample は mode='lddm' でのみ使用可能"
        if guidance_scale is None:
            guidance_scale = self.cfg.guidance_scale
        B = input_ids.size(0) if input_ids is not None else 1
        if skeleton_length is None:
            skeleton_length = torch.full((B,), seq_len, device=device, dtype=torch.long)
        pad_mask = length_to_padding_mask(skeleton_length, seq_len)
        z = torch.randn(B, seq_len, self.cfg.latent_dim, device=device)
        do_cfg = (input_ids is not None) and (guidance_scale != 1.0)
        return z, pad_mask, guidance_scale, do_cfg

    @torch.no_grad()
    def sample(self, seq_len, input_ids=None, attention_mask=None,
               skeleton_length=None, guidance_scale=None, device="cpu"):
        """DDPM サンプリング (全 num_timesteps ステップ, 確率的)。"""
        z, pad_mask, guidance_scale, do_cfg = self._init_sampling(
            seq_len, input_ids, skeleton_length, guidance_scale, device)
        B = z.size(0)
        for i in reversed(range(self.diffusion.num_timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            eps = self._predict_eps(z, t, input_ids, attention_mask, pad_mask, guidance_scale, do_cfg)
            alpha = self.diffusion.alphas[i]
            alpha_bar = self.diffusion.alpha_bars[i]
            beta = self.diffusion.betas[i]
            coef = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            mean = (z - coef * eps) / torch.sqrt(alpha)
            z = mean + (torch.sqrt(beta) * torch.randn_like(z) if i > 0 else 0.0)
        return self.vae.decode(self._denormalize_latent(z), pad_mask)

    @torch.no_grad()
    def sample_ddim(self, seq_len, input_ids=None, attention_mask=None,
                    skeleton_length=None, guidance_scale=None,
                    num_inference_steps=50, eta=0.0, device="cpu"):
        """DDIM サンプリング。eta=0 で決定的, eta=1 で DDPM 相当。"""
        z, pad_mask, guidance_scale, do_cfg = self._init_sampling(
            seq_len, input_ids, skeleton_length, guidance_scale, device)
        B = z.size(0)
        ab = self.diffusion.alpha_bars
        steps = torch.linspace(self.diffusion.num_timesteps - 1, 0,
                               num_inference_steps, device=device).round().long()
        steps = torch.unique_consecutive(steps)

        for idx in range(len(steps)):
            t_cur = int(steps[idx])
            t = torch.full((B,), t_cur, device=device, dtype=torch.long)
            eps = self._predict_eps(z, t, input_ids, attention_mask, pad_mask, guidance_scale, do_cfg)
            ab_t = ab[t_cur]
            ab_prev = ab[int(steps[idx + 1])] if idx + 1 < len(steps) \
                else torch.ones((), device=device)
            x0 = (z - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
            sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_prev)
            dir_xt = torch.sqrt(torch.clamp(1 - ab_prev - sigma ** 2, min=0.0)) * eps
            z = torch.sqrt(ab_prev) * x0 + dir_xt
            if eta > 0 and idx + 1 < len(steps):
                z = z + sigma * torch.randn_like(z)
        return self.vae.decode(self._denormalize_latent(z), pad_mask)


# --------------------------------------------------------------------------- #
#  簡易動作確認
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, L = 4, 32, 20

    # ===== Transformer backbone (J=50, C=3) =====
    J, C = 50, 3
    cfg = ModelConfig(skeleton_dim=J * C, max_seq_len=64, latent_dim=128,
                      d_model=128, nhead=4, num_layers=2, dim_feedforward=256,
                      num_timesteps=100, cfg_dropout=0.1, mode="vae")
    skeleton = torch.randn(B, T, J, C)
    lengths = torch.tensor([32, 28, 20, 16])
    m = SkeletonTextDiffusion(cfg)
    out = m(skeleton=skeleton, skeleton_length=lengths); out["loss"].backward()
    print(f"[transformer-vae] loss={out['loss'].item():.4f} recon={out['recon'].item():.4f} kl={out['kl'].item():.4f}")

    # ===== GCN backbone (手話グラフ, 連番化後 K 関節, C=3) =====
    Jg, Cg = sign_graph_num_joints(), 3   # K=48
    cfg_g = ModelConfig(skeleton_dim=Jg * Cg, max_seq_len=64, latent_dim=128,
                        d_model=128, nhead=4, num_layers=2, dim_feedforward=256,
                        num_timesteps=100, mode="vae",
                        vae_backbone="gcn", joint_coord_dim=Cg, num_joints=Jg,
                        gcn_temporal_kernel=5)
    skeleton_g = torch.randn(B, T, Jg, Cg)
    mg = SkeletonTextDiffusion(cfg_g)
    out = mg(skeleton=skeleton_g, skeleton_length=lengths); out["loss"].backward()
    print(f"[gcn-vae] loss={out['loss'].item():.4f} recon={out['recon'].item():.4f} kl={out['kl'].item():.4f}")

    # 共通の text 入力
    ii = torch.randint(0, 1000, (B, L)); am = torch.ones(B, L, dtype=torch.long); am[0, 15:] = 0

    # ===== GCN backbone + lddm (scratch text, in_context 条件付け) =====
    cfg_gl = ModelConfig(skeleton_dim=Jg * Cg, max_seq_len=64, latent_dim=128,
                         d_model=128, nhead=4, num_layers=2, dim_feedforward=256,
                         num_timesteps=100, cfg_dropout=0.1, mode="lddm",
                         vae_backbone="gcn", joint_coord_dim=Cg, num_joints=Jg,
                         conditioning_type="in_context",
                         text_encoder_type="scratch", text_vocab_size=1000,
                         text_num_layers=2, text_nhead=4, text_dim_feedforward=256, text_max_len=64)
    mgl = SkeletonTextDiffusion(cfg_gl)
    out = mgl(skeleton=skeleton_g, input_ids=ii, attention_mask=am, skeleton_length=lengths)
    out["loss"].backward()
    gen = mgl.sample_ddim(seq_len=24, input_ids=ii[:2], attention_mask=am[:2],
                          guidance_scale=2.5, num_inference_steps=20, eta=0.0)
    print(f"[gcn-lddm-scratch / in_context] loss={out['loss'].item():.4f}  gen shape={tuple(gen.shape)}")

    # ===== Transformer backbone + lddm (scratch text, cross_attention 条件付け) =====
    cfg_ca = ModelConfig(skeleton_dim=J * C, max_seq_len=64, latent_dim=128,
                         d_model=128, nhead=4, num_layers=2, dim_feedforward=256,
                         num_timesteps=100, cfg_dropout=0.1, mode="lddm",
                         conditioning_type="cross_attention",
                         text_encoder_type="scratch", text_vocab_size=1000,
                         text_num_layers=2, text_nhead=4, text_dim_feedforward=256, text_max_len=64)
    mca = SkeletonTextDiffusion(cfg_ca)
    out = mca(skeleton=skeleton, input_ids=ii, attention_mask=am, skeleton_length=lengths)
    out["loss"].backward()
    gen = mca.sample_ddim(seq_len=24, input_ids=ii[:2], attention_mask=am[:2],
                          guidance_scale=2.5, num_inference_steps=20, eta=0.0)
    print(f"[transformer-lddm-scratch / cross_attention] loss={out['loss'].item():.4f}  gen shape={tuple(gen.shape)}")

    # ===== pretrained text encoder (Transformer backbone, in_context) =====
    from transformers import BertConfig, BertModel
    tiny = BertModel(BertConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2,
                                num_attention_heads=2, intermediate_size=128,
                                max_position_embeddings=64))
    cfg_p = ModelConfig(skeleton_dim=J * C, max_seq_len=64, latent_dim=128,
                        d_model=128, nhead=4, num_layers=2, dim_feedforward=256,
                        num_timesteps=100, cfg_dropout=0.1, mode="lddm",
                        conditioning_type="cross_attention",
                        text_encoder_type="pretrained", freeze_text_encoder=True)
    mp = SkeletonTextDiffusion(cfg_p, text_encoder=tiny)
    out = mp(skeleton=skeleton, input_ids=ii, attention_mask=am, skeleton_length=lengths)
    out["loss"].backward()
    print(f"[transformer-lddm-pretrained / cross_attention] loss={out['loss'].item():.4f}")