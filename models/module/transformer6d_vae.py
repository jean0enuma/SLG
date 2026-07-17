"""
hand_transformer_vae_6d.py
===========================
HAND_BONES の 6D 回転表現を入出力とする Transformer ベース VAE.
hand_gcn_vae_6d.py (ST-GCN 版) と同一の入出力仕様:

    入力/出力: (B, T, 6, 20) or (T, 6, 20)
    潜在:      (B, T', V, latent_dim),  T' = T / 4

ST-GCN からの置き換え対応:
    空間 GCN  (SpatialGraphConv)  -> ボーン間の Self-Attention
                                     (+ 学習可能なボーン対バイアス = グラフ帰納バイアスの代替)
    時間 Conv (TCN, stride)       -> フレーム間の Self-Attention
                                     (時間ダウン/アップサンプルは Conv1d / interpolate で維持)
    隣接行列 A                     -> ボーン埋め込み + 相対バイアス (構造は学習で獲得)

可変長対応: lengths (B,) を渡すとパディングフレームを時間注意から除外.
損失は hand_gcn_vae_6d.vae_loss をそのまま再利用可能.
"""
from __future__ import annotations
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.Hand_gcn_vae_6d import HAND_BONES,BODY_BONES,ALL_BONES  # ボーン定義を共有

NEG_INF = -1e4  # softmax 用の実質 -inf (fp16 でも安全, NaN を発生させない)


# ----------------------------------------------------------------------
# 1. 基本部品
# ----------------------------------------------------------------------
def sinusoidal_pe(T: int, d_model: int, device=None) -> torch.Tensor:
    """(T, d_model) の正弦波位置符号. 可変長 T に対応するため学習パラメータにしない."""
    pos = torch.arange(T, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2, device=device).float()
                    * (-math.log(10000.0) / d_model))
    pe = torch.zeros(T, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class MHSA(nn.Module):
    """additive バイアス / key パディングマスク対応の Multi-Head Self-Attention.
    x: (N, L, D)"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, bias: torch.Tensor | None = None,
                key_pad: torch.Tensor | None = None):
        """
        bias:    (h, L, L)  additive attention バイアス (空間注意のボーン対バイアス)
        key_pad: (N, L) bool  True=パディング (時間注意で使用)
        """
        N, L, D = x.shape
        qkv = self.qkv(x).view(N, L, 3, self.h, self.dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # 各 (N, h, L, dh)

        attn_mask = None
        if bias is not None:
            attn_mask = bias.unsqueeze(0)              # (1, h, L, L)
        if key_pad is not None:
            pad = key_pad.view(N, 1, 1, L).float() * NEG_INF
            attn_mask = pad if attn_mask is None else attn_mask + pad

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(N, L, D)
        return self.proj(out)


class STTransformerBlock(nn.Module):
    """空間注意 (ボーン間) -> 時間注意 (フレーム間) -> FFN の分解型ブロック.
    ST-GCN の [GCN -> TCN] に対応. Pre-LN 構成. x: (B, T, V, D)"""

    def __init__(self, d_model: int, n_heads: int, num_bones: int,
                 ffn_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln_s = nn.LayerNorm(d_model)
        self.attn_s = MHSA(d_model, n_heads, dropout)
        # 空間注意の学習可能ボーン対バイアス (グラフ隣接行列の緩い代替, ゼロ初期化)
        self.bone_bias = nn.Parameter(torch.zeros(n_heads, num_bones, num_bones))

        self.ln_t = nn.LayerNorm(d_model)
        self.attn_t = MHSA(d_model, n_heads, dropout)

        self.ln_f = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_ratio * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x, time_pad: torch.Tensor | None = None):
        """time_pad: (B, T) bool  True=パディングフレーム"""
        B, T, V, D = x.shape

        # --- 空間注意: 各フレーム内でボーン 20 本が相互参照 ---
        h = self.ln_s(x).reshape(B * T, V, D)
        x = x + self.attn_s(h, bias=self.bone_bias).view(B, T, V, D)

        # --- 時間注意: 各ボーンの系列 T が相互参照 ---
        h = self.ln_t(x).permute(0, 2, 1, 3).reshape(B * V, T, D)
        kp = (time_pad.unsqueeze(1).expand(B, V, T).reshape(B * V, T)
              if time_pad is not None else None)
        h = self.attn_t(h, key_pad=kp).view(B, V, T, D).permute(0, 2, 1, 3)
        x = x + h

        # --- FFN ---
        x = x + self.ffn(self.ln_f(x))

        if time_pad is not None:  # パディング位置を零化 (ダウンサンプルへの漏れ防止)
            x = x.masked_fill(time_pad[:, :, None, None], 0.0)
        return x


class TemporalDown(nn.Module):
    """時間方向 1/2 ダウンサンプル (ST-GCN の stride=2 TCN に対応)."""

    def __init__(self, d_model: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel, stride=2,
                              padding=(kernel - 1) // 2)

    def forward(self, x):  # (B, T, V, D) -> (B, ceil(T/2), V, D)
        B, T, V, D = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * V, D, T)
        h = self.conv(h)
        T2 = h.shape[-1]
        return h.reshape(B, V, D, T2).permute(0, 3, 1, 2).contiguous()


class TemporalUp(nn.Module):
    """時間方向 x2 アップサンプル (ST-GCN の ConvTranspose TCN に対応)."""

    def __init__(self, d_model: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel,
                              padding=(kernel - 1) // 2)

    def forward(self, x):  # (B, T, V, D) -> (B, 2T, V, D)
        B, T, V, D = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * V, D, T)
        h = F.interpolate(h, scale_factor=2, mode="linear",
                          align_corners=False)
        h = self.conv(h)
        return h.reshape(B, V, D, 2 * T).permute(0, 3, 1, 2).contiguous()


# ----------------------------------------------------------------------
# 2. VAE 本体
# ----------------------------------------------------------------------
class HandTransformerVAE(nn.Module):
    """6D 回転特徴 (B, T, 6, 20) を入出力とする分解型時空間 Transformer VAE.

    Encoder: [Block] -> down/2 -> [Block] -> down/2 -> [Block] -> mu/logvar
    Decoder: proj    -> [Block] -> up x2  -> [Block] -> up x2  -> [Block] -> head
    潜在: (B, T', V, latent_dim), T' = T/4 (ST-GCN 版の圧縮率と一致)
    """

    def __init__(self, in_channels: int = 6, bones=HAND_BONES,
                 d_model: int = 128, n_heads: int = 8,
                 blocks_per_stage: int = 1, n_stages: int = 3,
                 latent_dim: int = 32, ffn_ratio: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.num_bones = len(bones)
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.n_down = n_stages - 1
        self.t_stride = 2 ** self.n_down
        if in_channels != 6:
            if bones==HAND_BONES:
                V=21
            elif bones==BODY_BONES:
                V=8
            elif bones==ALL_BONES:
                V=48
            else:
                raise ValueError(f"bones={bones} is not supported for in_channels={in_channels}.")
        else:
            V = self.num_bones
        self.embed = nn.Linear(in_channels, d_model)
        self.bone_emb = nn.Parameter(torch.zeros(1, 1, V, d_model))
        nn.init.trunc_normal_(self.bone_emb, std=0.02)

        def make_stage():
            return nn.ModuleList([
                STTransformerBlock(d_model, n_heads, V, ffn_ratio, dropout)
                for _ in range(blocks_per_stage)])

        # ---- Encoder ----
        self.enc_stages = nn.ModuleList([make_stage() for _ in range(n_stages)])
        self.enc_downs = nn.ModuleList(
            [TemporalDown(d_model) for _ in range(self.n_down)])
        self.enc_norm = nn.LayerNorm(d_model)
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.to_logvar.weight)   # sigma=1 から開始 (KL 爆発防止)
        nn.init.zeros_(self.to_logvar.bias)

        # ---- Decoder ----
        self.from_latent = nn.Linear(latent_dim, d_model)
        self.dec_stages = nn.ModuleList([make_stage() for _ in range(n_stages)])
        self.dec_ups = nn.ModuleList(
            [TemporalUp(d_model) for _ in range(self.n_down)])
        self.dec_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, in_channels)  # 活性化なし = 6D を線形回帰

    # ---- ユーティリティ ------------------------------------------
    @staticmethod
    def _pad_mask(lengths: torch.Tensor | None, T: int, B: int, device):
        """lengths (B,) -> time_pad (B, T) bool (True=パディング)."""
        if lengths is None:
            return None
        t = torch.arange(T, device=device).unsqueeze(0)
        return t >= lengths.unsqueeze(1)

    def _to_tokens(self, x):
        """(B, T, 6, 20) -> (B, T, V, D)  埋め込み + 位置/ボーン符号."""
        h = self.embed(x.movedim(-2, -1))                        # (B,T,V,D)
        pe = sinusoidal_pe(h.shape[1], self.d_model, h.device)
        return h + pe[None, :, None, :] + self.bone_emb

    def _match_time(self, x, T_target):
        """(B, T, V, D or C) の時間長を線形補間で T_target に合わせる."""
        if x.shape[1] == T_target:
            return x
        B, T, V, C = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * V, C, T)
        h = F.interpolate(h, size=T_target, mode="linear",
                          align_corners=False)
        return h.reshape(B, V, C, T_target).permute(0, 3, 1, 2).contiguous()

    # ---- VAE コア -------------------------------------------------
    def encode(self, x, lengths=None):
        """(B,T,6,20) -> mu, logvar 各 (B, T', V, latent_dim)."""
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)
        B, T = x.shape[:2]
        h = self._to_tokens(x)
        pad = self._pad_mask(lengths, T, B, x.device)

        for i, stage in enumerate(self.enc_stages):
            for blk in stage:
                h = blk(h, time_pad=pad)
            if i < self.n_down:
                h = self.enc_downs[i](h)
                if lengths is not None:
                    lengths = (lengths + 1) // 2
                    pad = self._pad_mask(lengths, h.shape[1], B, x.device)
                else:
                    pad = None
        h = self.enc_norm(h)
        return self.to_mu(h), self.to_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, T_target: int | None = None):
        """z: (B, T', V, latent_dim) -> (B, T, 6, 20)."""
        h = self.from_latent(z)
        pe = sinusoidal_pe(h.shape[1], self.d_model, h.device)
        h = h + pe[None, :, None, :] + self.bone_emb

        for i, stage in enumerate(self.dec_stages):
            for blk in stage:
                h = blk(h)
            if i < self.n_down:
                h = self.dec_ups[i](h)
        out = self.head(self.dec_norm(h))                        # (B,T,V,6)
        if T_target is not None:
            out = self._match_time(out, T_target)
        return out.movedim(-1, -2).contiguous()                  # (B,T,6,V)

    def forward(self, x, lengths=None):
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)
        T_in = x.shape[1]
        mu, logvar = self.encode(x, lengths)
        z = self.reparameterize(mu, logvar) if self.training else mu
        recon = self.decode(z, T_target=T_in)
        if squeeze:
            recon, mu, logvar = recon.squeeze(0), mu.squeeze(0), logvar.squeeze(0)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, n: int, T: int, device=None):
        """事前分布 N(0, I) から生成. T は出力フレーム数."""
        device = device or next(self.parameters()).device
        T_lat = max(1, T // self.t_stride)
        z = torch.randn(n, T_lat, self.num_bones, self.latent_dim,
                        device=device)
        return self.decode(z, T_target=T)


# ----------------------------------------------------------------------
# 3. 動作確認
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from models.module.STGCNHand  import (hand_joints_to_6d, make_bone_mask,
                                 vae_loss, rotation_6d_to_matrix)

    torch.manual_seed(0)
    model = HandTransformerVAE()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f} M")

    # --- 前処理 (ST-GCN 版と完全共通) ---
    joints = torch.randn(4, 64, 3, 21)
    d6, valid = hand_joints_to_6d(joints, return_valid=True)
    joint_mask = torch.rand(4, 64, 21) > 0.05
    bone_mask = make_bone_mask(joint_mask, valid_rot=valid)

    # --- 可変長バッチ ---
    lengths = torch.tensor([64, 50, 37, 20])
    time_valid = torch.arange(64).unsqueeze(0) < lengths.unsqueeze(1)
    bone_mask = bone_mask & time_valid.unsqueeze(-1)

    # --- 学習ステップ ---
    model.train()
    recon, mu, logvar = model(d6, lengths=lengths)
    print(f"recon {tuple(recon.shape)}  latent {tuple(mu.shape)}")
    assert recon.shape == d6.shape
    assert not torch.isnan(recon).any()

    total, logs = vae_loss(recon, d6, mu, logvar, mask=bone_mask,
                           beta=1e-3, free_bits=0.02)
    total.backward()
    print(f"loss: {logs}")
    print("backward OK")

    # --- 単一サンプル・奇数長 ---
    d6_single = hand_joints_to_6d(torch.randn(37, 3, 21))
    r, mu1, lv1 = model(d6_single)
    print(f"single: {tuple(d6_single.shape)} -> {tuple(r.shape)}")
    assert r.shape == d6_single.shape

    # --- 生成 ---
    model.eval()
    gen = model.sample(n=2, T=64)
    R = rotation_6d_to_matrix(gen.movedim(-2, -1))
    print(f"sampled: {tuple(gen.shape)}, det(R) mean={torch.det(R).mean():.4f}")