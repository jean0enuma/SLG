"""
latent_flow_moe.py  (修正版)
=============================
修正内容:
    [FIX-0] velocity の返り値をテンソル1本に戻した (タプル返しが sample の
            Euler 更新で TypeError を起こしていた). hi が必要な箇所は
            self.router(t) を再計算する.
    [FIX-1] cross-attention 用の学習可能 null_context (1トークン) を導入.
            旧実装の self.null_cond.reshape(2,1,-1) は形状不正で,
            低ノイズ分岐の [hi] インデックスも誤りだった.
    [FIX-2] sample の CFG 無条件分岐を context=None (-> null_context) に修正.
            旧実装の cond.mean + 全True マスクは「条件付き平均を参照しつつ
            全KVをマスクする」誤った無条件化だった.
    [FIX-3] context API をトークン列 (B, S, cond_dim) + context_mask に統一.
            adaLN 用のプーリングはマスク付き平均で内部生成する.
            (後方互換: 2D (B, cond_dim) を渡した場合はプーリング済みとして扱う.
             cross_attn=True では 3D 必須)
    [FIX-4] __init__ で self.part_decoders が引数 part_decoders(None) に
            上書きされ body_vae/hand_vae の登録が消えていた問題を修正.
"""
from __future__ import annotations
import math
from typing import Callable, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.module.transformer6d_vae import MHSA, sinusoidal_pe, NEG_INF


# ----------------------------------------------------------------------
# 1. RoPE 本体
# ----------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    """回転位置符号の cos/sin テーブルを生成する.
    head_dim は偶数であること. テーブルは系列長ごとにキャッシュする."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float()
                                   / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: dict = {}

    def forward(self, seq_len: int, device, dtype):
        key = (seq_len, device, dtype)
        if key not in self._cache:
            pos = torch.arange(seq_len, device=device).float()
            angles = torch.outer(pos, self.inv_freq.to(device))  # (L, dh/2)
            emb = torch.cat([angles, angles], dim=-1)             # (L, dh)
            self._cache = {key: (emb.cos().to(dtype), emb.sin().to(dtype))}
        return self._cache[key]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    """x: (N, h, L, dh), cos/sin: (L, dh) -> 回転適用後の同形状."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


# ----------------------------------------------------------------------
# 2. RoPE 付き Self-Attention
# ----------------------------------------------------------------------
class RoPEMHSA(nn.Module):
    """q, k に RoPE を適用する Multi-Head Self-Attention.
    x: (N, L, D). bias: (h, L, L) additive, key_pad: (N, L) True=パディング."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.dh, base=rope_base)

    def forward(self, x, bias: torch.Tensor | None = None,
                key_pad: torch.Tensor | None = None):
        N, L, D = x.shape
        qkv = self.qkv(x).view(N, L, 3, self.h, self.dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # 各 (N, h, L, dh)

        cos, sin = self.rope(L, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_mask = None
        if bias is not None:
            attn_mask = bias.unsqueeze(0)
        if key_pad is not None:
            pad = key_pad.view(N, 1, 1, L).float() * NEG_INF
            attn_mask = pad if attn_mask is None else attn_mask + pad

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(N, L, D))


# ----------------------------------------------------------------------
# 3. 部位分割の定義
# ----------------------------------------------------------------------
def default_part_index(n_body: int = 7, n_hand: int = 20) -> Dict[str, List[int]]:
    b = list(range(n_body))
    l = list(range(n_body, n_body + n_hand))
    r = list(range(n_body + n_hand, n_body + 2 * n_hand))
    return {"body": b, "left": l, "right": r}


# ----------------------------------------------------------------------
# 4. Timestep 埋め込みと DiT ブロック (adaLN-Zero)
# ----------------------------------------------------------------------
class TimestepEmbedding(nn.Module):
    """連続時刻 t in [0,1] -> (B, d_model)."""

    def __init__(self, d_model: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0)
                          * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(-1) * 1000.0 * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None, None, :]) + shift[:, None, None, :]


class DiTBlock(nn.Module):
    """分解型時空間 DiT ブロック (空間注意 -> 時間注意 -> FFN, adaLN-Zero)."""

    def __init__(self, d_model: int, n_heads: int, num_tokens: int,
                 ffn_ratio: int = 4):
        super().__init__()
        self.ln_s = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn_s = MHSA(d_model, n_heads)
        self.token_bias = nn.Parameter(
            torch.zeros(n_heads, num_tokens, num_tokens))
        self.ln_t = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn_t = MHSA(d_model, n_heads)
        self.ln_f = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model), nn.GELU(),
            nn.Linear(ffn_ratio * d_model, d_model))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 9 * d_model))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor,
                time_pad: torch.Tensor | None = None) -> torch.Tensor:
        B, T, V, D = x.shape
        (sh_s, sc_s, g_s, sh_t, sc_t, g_t,
         sh_f, sc_f, g_f) = self.mod(temb).chunk(9, dim=-1)

        h = modulate(self.ln_s(x), sh_s, sc_s).reshape(B * T, V, D)
        h = self.attn_s(h, bias=self.token_bias).view(B, T, V, D)
        x = x + g_s[:, None, None, :] * h

        h = modulate(self.ln_t(x), sh_t, sc_t)
        h = h.permute(0, 2, 1, 3).reshape(B * V, T, D)
        kp = (time_pad.unsqueeze(1).expand(B, V, T).reshape(B * V, T)
              if time_pad is not None else None)
        h = self.attn_t(h, key_pad=kp).view(B, V, T, D).permute(0, 2, 1, 3)
        x = x + g_t[:, None, None, :] * h

        h = self.ffn(modulate(self.ln_f(x), sh_f, sc_f))
        x = x + g_f[:, None, None, :] * h

        if time_pad is not None:
            x = x.masked_fill(time_pad[:, :, None, None], 0.0)
        return x


class RoPEDiTBlock(nn.Module):
    """adaLN-Zero + RoPE 時間注意の DiT ブロック (DiTBlock 互換)."""

    def __init__(self, d_model: int, n_heads: int, num_tokens: int,
                 ffn_ratio: int = 4):
        super().__init__()
        self.ln_s = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn_s = MHSA(d_model, n_heads)
        self.token_bias = nn.Parameter(
            torch.zeros(n_heads, num_tokens, num_tokens))
        self.ln_t = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn_t = RoPEMHSA(d_model, n_heads)
        self.ln_f = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model), nn.GELU(),
            nn.Linear(ffn_ratio * d_model, d_model))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 9 * d_model))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, x, temb, time_pad: torch.Tensor | None = None):
        B, T, V, D = x.shape
        (sh_s, sc_s, g_s, sh_t, sc_t, g_t,
         sh_f, sc_f, g_f) = self.mod(temb).chunk(9, dim=-1)

        h = modulate(self.ln_s(x), sh_s, sc_s).reshape(B * T, V, D)
        h = self.attn_s(h, bias=self.token_bias).view(B, T, V, D)
        x = x + g_s[:, None, None, :] * h

        h = modulate(self.ln_t(x), sh_t, sc_t)
        h = h.permute(0, 2, 1, 3).reshape(B * V, T, D)
        kp = (time_pad.unsqueeze(1).expand(B, V, T).reshape(B * V, T)
              if time_pad is not None else None)
        h = self.attn_t(h, key_pad=kp).view(B, V, T, D).permute(0, 2, 1, 3)
        x = x + g_t[:, None, None, :] * h

        h = self.ffn(modulate(self.ln_f(x), sh_f, sc_f))
        x = x + g_f[:, None, None, :] * h

        if time_pad is not None:
            x = x.masked_fill(time_pad[:, :, None, None], 0.0)
        return x


class MHCA(nn.Module):
    """Multi-Head Cross-Attention. query = 骨格潜在トークン, KV = テキストトークン."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, ctx, ctx_pad: torch.Tensor | None = None):
        N, L, D = x.shape
        S = ctx.shape[1]
        q = self.q(x).view(N, L, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(ctx).view(N, S, 2, self.h, self.dh) \
                           .permute(2, 0, 3, 1, 4)
        attn_mask = None
        if ctx_pad is not None:
            attn_mask = ctx_pad.view(N, 1, 1, S).float() * NEG_INF
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(N, L, D))


class CrossCondDiTBlock(nn.Module):
    """テキスト条件付き DiT ブロック (adaLN + cross-attention)."""

    def __init__(self, d_model: int, n_heads: int, num_tokens: int,
                 ffn_ratio: int = 4, is_rope: bool = False):
        super().__init__()
        self.ln_s = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn_s = MHSA(d_model, n_heads)
        self.token_bias = nn.Parameter(
            torch.zeros(n_heads, num_tokens, num_tokens))
        self.ln_t = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn_t = MHSA(d_model, n_heads) if not is_rope \
            else RoPEMHSA(d_model, n_heads)
        self.ln_x = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn_x = MHCA(d_model, n_heads)
        self.ln_f = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model), nn.GELU(),
            nn.Linear(ffn_ratio * d_model, d_model))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 12 * d_model))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor,
                context: torch.Tensor,
                ctx_pad: torch.Tensor | None = None,
                time_pad: torch.Tensor | None = None) -> torch.Tensor:
        B, T, V, D = x.shape
        (sh_s, sc_s, g_s, sh_t, sc_t, g_t,
         sh_x, sc_x, g_x, sh_f, sc_f, g_f) = self.mod(temb).chunk(12, dim=-1)

        h = modulate(self.ln_s(x), sh_s, sc_s).reshape(B * T, V, D)
        h = self.attn_s(h, bias=self.token_bias).view(B, T, V, D)
        x = x + g_s[:, None, None, :] * h

        h = modulate(self.ln_t(x), sh_t, sc_t)
        h = h.permute(0, 2, 1, 3).reshape(B * V, T, D)
        kp = (time_pad.unsqueeze(1).expand(B, V, T).reshape(B * V, T)
              if time_pad is not None else None)
        h = self.attn_t(h, key_pad=kp).view(B, V, T, D).permute(0, 2, 1, 3)
        x = x + g_t[:, None, None, :] * h

        h = modulate(self.ln_x(x), sh_x, sc_x).reshape(B, T * V, D)
        h = self.attn_x(h, context, ctx_pad=ctx_pad).view(B, T, V, D)
        x = x + g_x[:, None, None, :] * h

        h = self.ffn(modulate(self.ln_f(x), sh_f, sc_f))
        x = x + g_f[:, None, None, :] * h

        if time_pad is not None:
            x = x.masked_fill(time_pad[:, :, None, None], 0.0)
        return x


class MoEFFN(nn.Module):
    """MoE FFN (expert は 1 つの FFN で共有, adaLN-Zero 変調は共通)."""

    def __init__(self, d_model: int, ffn_ratio: int = 4, num_experts: int = 3):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model), nn.GELU(),
            nn.Linear(ffn_ratio * d_model, d_model))
        self.num_experts = num_experts
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 3 * d_model))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor,
                expert_idx: torch.Tensor) -> torch.Tensor:
        B, T, V, D = x.shape
        sh, sc, g = self.mod(temb).chunk(3, dim=-1)
        h = modulate(x, sh[:, None, None, :], sc[:, None, None, :])
        h = self.ffn(h)
        g_exp = g[:, None, None, :].expand(B, T, V, D)
        g_mask = F.one_hot(expert_idx.clamp(0, self.num_experts - 1),
                           num_classes=self.num_experts).float()
        g_mask = g_mask.view(B, T, 1, self.num_experts).expand(-1, -1, V, -1)
        g_masked = (g_exp.unsqueeze(-1) * g_mask).sum(dim=-1)
        return x + g_masked * h


class PartDenoiser(nn.Module):
    """潜在トークン列 (B, T', V_part, latent_dim) に対する速度場予測器."""

    def __init__(self, latent_dim: int, num_tokens: int, d_model: int = 256,
                 n_heads: int = 8, depth: int = 6, ffn_ratio: int = 4):
        super().__init__()
        self.proj_in = nn.Linear(latent_dim, d_model)
        self.token_emb = nn.Parameter(torch.zeros(1, 1, num_tokens, d_model))
        nn.init.trunc_normal_(self.token_emb, std=0.02)
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, num_tokens, ffn_ratio)
            for _ in range(depth)])
        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mod_out = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.proj_out = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.mod_out[1].weight)
        nn.init.zeros_(self.mod_out[1].bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self.d_model = d_model

    def forward(self, z: torch.Tensor, temb: torch.Tensor,
                time_pad: torch.Tensor | None = None) -> torch.Tensor:
        h = self.proj_in(z) + self.token_emb
        pe = sinusoidal_pe(h.shape[1], self.d_model, h.device)
        h = h + pe[None, :, None, :]
        for blk in self.blocks:
            h = blk(h, temb, time_pad=time_pad)
        sh, sc = self.mod_out(temb).chunk(2, dim=-1)
        h = modulate(self.norm_out(h), sh, sc)
        return self.proj_out(h)


class PartDenoiserXAttn(PartDenoiser):
    def __init__(self, latent_dim, num_tokens, d_model=256, n_heads=8,
                 depth=6, ffn_ratio=4):
        super().__init__(latent_dim, num_tokens, d_model, n_heads, 0, ffn_ratio)
        self.blocks = nn.ModuleList([
            CrossCondDiTBlock(d_model, n_heads, num_tokens, ffn_ratio)
            for _ in range(depth)])

    def forward(self, z, temb, context, ctx_pad=None, time_pad=None):
        h = self.proj_in(z) + self.token_emb
        h = h + sinusoidal_pe(h.shape[1], self.d_model,
                              h.device)[None, :, None, :]
        for blk in self.blocks:
            h = blk(h, temb, context, ctx_pad=ctx_pad, time_pad=time_pad)
        sh, sc = self.mod_out(temb).chunk(2, dim=-1)
        return self.proj_out(modulate(self.norm_out(h), sh, sc))


# ----------------------------------------------------------------------
# 5. Wan2.2 風 MoE ルーティング
# ----------------------------------------------------------------------
class TimestepRouter:
    """t (B,) -> bool (B,)  True = high-noise expert (全骨格)."""

    def __init__(self, boundary: float = 0.875):
        self.boundary = boundary
        self._custom: Callable[[torch.Tensor], torch.Tensor] | None = None

    def set_rule(self, fn: Callable[[torch.Tensor], torch.Tensor] | None):
        self._custom = fn

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if self._custom is not None:
            return self._custom(t).bool()
        return t >= self.boundary


# ----------------------------------------------------------------------
# 6. Flow Matching 本体
# ----------------------------------------------------------------------
class SignLatentFlowMoE(nn.Module):
    """凍結 VAE の潜在空間で速度場を学習する MoE Flow Matching."""

    def __init__(self, vae: nn.Module,
                 hand_vae: nn.Module | None = None,
                 body_vae: nn.Module | None = None,
                 part_index: Dict[str, List[int]] | None = None,
                 boundary: float = 0.875,
                 part_decoders: Dict[str, nn.Module] | None = None,
                 d_model: int = 256, n_heads: int = 8,
                 depth_high: int = 6, depth_low: int = 4,
                 cond_dim: int | None = None,
                 cross_attn: bool = False,
                 shared_lr: bool = False,
                 train_length_predictor: bool = False):
        super().__init__()
        self.vae = vae
        self.train_length_predictor = train_length_predictor
        # [FIX-4] part_decoders を後段の None 代入で上書きしない
        self.part_decoders = dict(part_decoders) if part_decoders else {}
        if body_vae is not None:
            self.body_vae = body_vae
            self.part_decoders["body"] = body_vae
            for p in self.body_vae.parameters():
                p.requires_grad_(False)
        if hand_vae is not None:
            self.hand_vae = hand_vae
            self.part_decoders["left"] = hand_vae
            self.part_decoders["right"] = hand_vae
            for p in self.hand_vae.parameters():
                p.requires_grad_(False)
        if not self.part_decoders:
            self.part_decoders = None   # sample の "is None" 判定と整合

        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.latent_dim = vae.latent_dim
        self.num_tokens = vae.num_bones
        self.part_index = part_index or default_part_index()
        assert sorted(sum(self.part_index.values(), [])) \
            == list(range(self.num_tokens)), "part_index がトークン全体を分割していません"

        self.router = TimestepRouter(boundary)
        self.t_embed = TimestepEmbedding(d_model)

        # ---- テキスト条件 ----
        # [FIX-3] context は常にトークン列 (B, S, cond_dim) を第一級とする.
        #   - adaLN 経路: マスク付き平均でプーリングして cond_proj -> temb 加算
        #   - cross-attn 経路: context_proj でトークン列を d_model へ射影
        # [FIX-1] null_cond (adaLN用ベクトル) と null_context (cross-attn用
        #   1トークン) を別パラメータとして持つ.
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.cond_proj = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, d_model), nn.SiLU(),
                nn.Linear(d_model, d_model))
            self.null_cond = nn.Parameter(torch.zeros(1, d_model))
            nn.init.trunc_normal_(self.null_cond, std=0.02)

        self.cross_attn = cross_attn
        if cross_attn:
            assert cond_dim is not None, "cross_attn=True には cond_dim が必要"
            self.context_proj = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, d_model), nn.SiLU(),
                nn.Linear(d_model, d_model))
            self.null_context = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.null_context, std=0.02)
            self.expert_high = PartDenoiserXAttn(
                self.latent_dim, self.num_tokens, d_model, n_heads, depth_high)
            self.experts_low = nn.ModuleDict({
                name: PartDenoiserXAttn(self.latent_dim, len(idx),
                                        d_model, n_heads, depth_low)
                for name, idx in self.part_index.items()})
            if shared_lr:
                self.experts_low['right']=self.experts_low['left']
        else:
            self.expert_high = PartDenoiser(
                self.latent_dim, self.num_tokens, d_model, n_heads, depth_high)
            self.experts_low = nn.ModuleDict({
                name: PartDenoiser(self.latent_dim, len(idx),
                                   d_model, n_heads, depth_low)
                for name, idx in self.part_index.items()})

        self.register_buffer("z_mean", torch.zeros(1))
        self.register_buffer("z_std", torch.ones(1))
        self.register_buffer("z_mean_hand", torch.zeros(1))
        self.register_buffer("z_std_hand", torch.ones(1))
        self.register_buffer("z_mean_body", torch.zeros(1))
        self.register_buffer("z_std_body", torch.ones(1))

        ### Length Predictor
        l_encoder=nn.TransformerEncoderLayer(d_model=cond_dim, nhead=8, dim_feedforward=4*cond_dim, batch_first=True)
        self.length_predictor=nn.TransformerEncoder(l_encoder, num_layers=2)
        self.length_fc=nn.Linear(cond_dim,1)
        if not self.train_length_predictor:
            for p in self.parameters():
                p.requires_grad_(True)
            for p in self.length_predictor.parameters():
                p.requires_grad_(False)
            for p in self.length_fc.parameters():
                p.requires_grad_(False)
        else:
            for p in self.parameters():
                p.requires_grad_(False)
            for p in self.length_predictor.parameters():
                p.requires_grad_(True)
            for p in self.length_fc.parameters():
                p.requires_grad_(True)


    def length_predict(self,context,context_mask,input_length,return_loss=False):
        # context: (B, S, cond_dim), context_mask: (B, S)(False=pad),input_length: (B,)
        # マスク付き平均プーリング
        lengths=self.length_predictor(context, src_key_padding_mask=~context_mask)
        lengths=self.length_fc(lengths[:,0]).squeeze(-1)#(B,)
        if return_loss:
            #relative loss
            loss=F.l1_loss(lengths,input_length.float())
            return loss
        return lengths

    # ---- 潜在の入出力 ---------------------------------------------
    @torch.no_grad()
    def set_latent_stats(self, d6_batch, lengths=None):
        mu, _ = self.vae.encode(d6_batch, lengths)
        valid = self._length_mask(lengths, mu.shape[1], mu.device)
        m = mu[valid] if valid is not None else mu
        self.z_mean = m.mean().reshape(1)
        self.z_std = m.std().clamp_min(1e-4).reshape(1)

    @torch.no_grad()
    def set_latent_stats_hands(self, d6_batch, lengths=None):
        left_d6, right_d6 = d6_batch[:, :, -40:-20], d6_batch[:, :, -20:]
        mu_l, _ = self.hand_vae.encode(left_d6, lengths)
        mu_r, _ = self.hand_vae.encode(right_d6, lengths)
        mu = torch.cat([mu_l, mu_r], dim=2)
        valid = self._length_mask(lengths, mu.shape[1], mu.device)
        m = mu[valid] if valid is not None else mu
        self.z_mean_hand = m.mean().reshape(1)
        self.z_std_hand = m.std().clamp_min(1e-4).reshape(1)

    @torch.no_grad()
    def set_latent_stats_body(self, d6_batch, lengths=None):
        body_d6 = d6_batch[:, :, :7]
        mu, _ = self.body_vae.encode(body_d6, lengths)
        valid = self._length_mask(lengths, mu.shape[1], mu.device)
        m = mu[valid] if valid is not None else mu
        self.z_mean_body = m.mean().reshape(1)
        self.z_std_body = m.std().clamp_min(1e-4).reshape(1)

    @torch.no_grad()
    def full_set_latent_stats(self, train_loader, device=None):
        """訓練データ全体の VAE 潜在平均・標準偏差を計算する."""
        if device is None:
            device = next(self.vae.parameters()).device
        vae_was_training = self.vae.training
        self.vae.eval()

        total_sum = torch.zeros((), device=device, dtype=torch.float64)
        total_sq_sum = torch.zeros((), device=device, dtype=torch.float64)
        total_count = 0
        for batch in train_loader:
            d6 = batch[0].to(device, non_blocking=True)
            lengths = batch[1].to(device, non_blocking=True)
            mu, _ = self.vae.encode(d6, lengths)
            valid = self._length_mask(lengths, mu.shape[1], mu.device)
            m = mu[valid] if valid is not None else mu
            m = m.to(torch.float64)
            total_sum += m.sum()
            total_sq_sum += (m * m).sum()
            total_count += m.numel()

        if total_count < 2:
            raise RuntimeError("潜在統計を計算するための有効データが不足しています。")
        mean = total_sum / total_count
        var = (total_sq_sum - total_count * mean.square()) / (total_count - 1)
        std = var.clamp_min(1e-8).sqrt().clamp_min(1e-4)
        self.z_mean.copy_(mean.to(self.z_mean.dtype).reshape(1))
        self.z_std.copy_(std.to(self.z_std.dtype).reshape(1))
        self.vae.train(vae_was_training)
        print(f"[latent stats] mean={self.z_mean.item():.6f}, "
              f"std={self.z_std.item():.6f}, count={total_count:,}")

    def _encode(self, d6, lengths=None):
        with torch.no_grad():
            mu, _ = self.vae.encode(d6, lengths)
        return (mu - self.z_mean) / self.z_std

    def hand_encode(self, d6, lengths=None):
        left_hand_d6 = d6[:, :, :, -40:-20]
        right_hand_d6 = d6[:, :, :, -20:]
        with torch.no_grad():
            mu_l, _ = self.hand_vae.encode(left_hand_d6, lengths)
            mu_r, _ = self.hand_vae.encode(right_hand_d6, lengths)
        return ((mu_l - self.z_mean_hand) / self.z_std_hand,
                (mu_r - self.z_mean_hand) / self.z_std_hand)

    def body_encode(self, d6, lengths=None):
        body_d6 = d6[:, :, :, :7]
        with torch.no_grad():
            mu, _ = self.body_vae.encode(body_d6, lengths)
        return (mu - self.z_mean_body) / self.z_std_body

    def _decode(self, z, T_target):
        return self.vae.decode(z * self.z_std + self.z_mean, T_target=T_target)

    # ---- 条件埋め込み ------------------------------------------------
    def _pool_context(self, context, context_mask):
        """[FIX-3] adaLN 用プーリング.
        context: (B, S, cond_dim) トークン列 or (B, cond_dim) プーリング済み.
        context_mask: (B, S) True=有効トークン (HF attention_mask 互換)."""
        if context is None:
            return None
        if context.dim() == 2:
            return context
        if context_mask is not None:
            m = context_mask.unsqueeze(-1).float()
            return (context * m).sum(1) / m.sum(1).clamp_min(1e-6)
        return context.mean(dim=1)

    def _combine_emb(self, t: torch.Tensor,
                     cond: torch.Tensor | None,
                     drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        temb = self.t_embed(t)
        if self.cond_dim is None:
            return temb
        B = t.shape[0]
        if cond is None:
            c = self.null_cond.expand(B, -1)
        else:
            c = self.cond_proj(cond)
            if drop_mask is not None and drop_mask.any():
                c = torch.where(drop_mask.unsqueeze(-1),
                                self.null_cond.expand(B, -1), c)
        return temb + c

    def _prepare_context(self, context, context_mask, drop_mask,
                         B: int, device):
        """[FIX-1/2] cross-attn 用 KV 列と ctx_pad を構成.
        無条件 (context=None) と CFG ドロップは学習可能 null_context に置換.
        戻り値: ctx (B, S, d_model), ctx_pad (B, S) True=パディング or None."""
        if context is None:
            return self.null_context.expand(B, 1, -1), None
        assert context.dim() == 3, \
            "cross_attn=True では context は (B, S, cond_dim) のトークン列"
        ctx = self.context_proj(context)
        ctx_pad = ~context_mask.bool() if context_mask is not None else None
        if drop_mask is not None and drop_mask.any():
            S = ctx.shape[1]
            null = self.null_context.expand(B, S, -1)
            ctx = torch.where(drop_mask[:, None, None], null, ctx)
            # ドロップしたサンプルは null 1トークンのみ有効化
            null_pad = torch.ones(B, S, dtype=torch.bool, device=device)
            null_pad[:, 0] = False
            base = ctx_pad if ctx_pad is not None \
                else torch.zeros(B, S, dtype=torch.bool, device=device)
            ctx_pad = torch.where(drop_mask[:, None], null_pad, base)
        return ctx, ctx_pad

    # ---- 長さマスク ---------------------------------------------------
    def _length_mask(self, lengths, T_lat: int, device):
        if lengths is None:
            return None
        lengths = torch.as_tensor(lengths, device=device)
        lat_len = torch.div(lengths + self.vae.t_stride - 1,
                            self.vae.t_stride, rounding_mode="floor")
        t = torch.arange(T_lat, device=device).unsqueeze(0)
        return t < lat_len.clamp(min=1).unsqueeze(1)

    # ---- MoE 速度場 -------------------------------------------------
    def velocity(self, z_t: torch.Tensor, t: torch.Tensor,
                 context: torch.Tensor | None = None,
                 context_mask: torch.Tensor | None = None,
                 drop_mask: torch.Tensor | None = None,
                 time_pad: torch.Tensor | None = None) -> torch.Tensor:
        """[FIX-0] 返り値はテンソル1本. hi が必要なら self.router(t) を使う.
        z_t: (B, T', V, ld), t: (B,)
        context: (B, S, cond_dim) テキストトークン列 (adaLN のみの場合は
                 (B, cond_dim) のプーリング済みも許容). None で無条件.
        context_mask: (B, S) True=有効トークン (HF attention_mask 互換)
        time_pad: (B, T') True=パディングフレーム."""
        B = z_t.shape[0]
        cond = self._pool_context(context, context_mask)
        temb = self._combine_emb(t, cond, drop_mask)
        if self.cross_attn:
            ctx, ctx_pad = self._prepare_context(
                context, context_mask, drop_mask, B, z_t.device)

        out_dtype = torch.get_autocast_dtype("cuda") \
            if torch.is_autocast_enabled() else z_t.dtype
        out = torch.empty_like(z_t, dtype=out_dtype)
        hi = self.router(t)

        if hi.any():
            tp = time_pad[hi] if time_pad is not None else None
            if self.cross_attn:
                cp = ctx_pad[hi] if ctx_pad is not None else None
                out[hi] = self.expert_high(z_t[hi], temb[hi], context=ctx[hi],
                                           ctx_pad=cp, time_pad=tp)
            else:
                out[hi] = self.expert_high(z_t[hi], temb[hi], time_pad=tp)
        if (~hi).any():
            lo = ~hi
            z_lo, temb_lo = z_t[lo], temb[lo]
            tp = time_pad[lo] if time_pad is not None else None
            out_lo = torch.empty(z_lo.shape, dtype=out_dtype,
                                 device=z_lo.device)
            if self.cross_attn:
                ctx_lo = ctx[lo]                                   # [FIX-1]
                cp = ctx_pad[lo] if ctx_pad is not None else None
                for name, idx in self.part_index.items():
                    out_lo[:, :, idx] = self.experts_low[name](
                        z_lo[:, :, idx], temb_lo, context=ctx_lo,
                        ctx_pad=cp, time_pad=tp)
            else:
                for name, idx in self.part_index.items():
                    out_lo[:, :, idx] = self.experts_low[name](
                        z_lo[:, :, idx], temb_lo, time_pad=tp)
            out[lo] = out_lo
        return out

    # ---- 学習 -------------------------------------------------------
    def training_loss(self, d6: torch.Tensor, lengths=None,
                      cond: torch.Tensor | None = None,
                      cond_mask: torch.Tensor | None = None,
                      cond_drop_prob: float = 0.1,
                      part_weight: Dict[str, float] | None = None):
        """
        d6:   (B, T, 6, Nb)  全骨格の6D特徴
        cond: (B, S, cond_dim) テキストトークン列. None で無条件学習
        cond_mask: (B, S) True=有効トークン (HF attention_mask 互換)
        """
        if self.train_length_predictor:
            # 長さ予測器の損失を計算
            length_loss = self.length_predict(cond, cond_mask, lengths, return_loss=True)
            return length_loss, {"length_loss": length_loss.item()}
        if hasattr(self, "hand_vae") and hasattr(self, "body_vae"):
            z_hand_l, z_hand_r = self.hand_encode(d6, lengths)
            z_body = self.body_encode(d6, lengths)
            z0 = torch.cat([z_body, z_hand_l, z_hand_r], dim=2)
        else:
            z0 = self._encode(d6, lengths)  # (B, T', V, ld)

        B, T_lat = z0.shape[0], z0.shape[1]
        valid = self._length_mask(lengths, T_lat, z0.device)
        time_pad = ~valid if valid is not None else None

        t = torch.rand(B, device=z0.device)
        eps = torch.randn_like(z0)
        if valid is not None:
            z0 = z0 * valid[:, :, None, None]
        z_t = (1 - t.view(B, 1, 1, 1)) * z0 + t.view(B, 1, 1, 1) * eps
        v_target = eps - z0

        drop_mask = None
        if cond is not None and cond_drop_prob > 0:
            drop_mask = torch.rand(B, device=z0.device) < cond_drop_prob
        v_pred = self.velocity(z_t, t, context=cond, context_mask=cond_mask,
                               drop_mask=drop_mask, time_pad=time_pad)
        hi = self.router(t)
        se = (v_pred - v_target).pow(2)

        if part_weight is not None:
            w = torch.ones(self.num_tokens, device=z0.device)
            for name, idx in self.part_index.items():
                w[idx] = part_weight.get(name, 1.0)
            se = se * w.view(1, 1, -1, 1)
        if valid is not None:
            m = valid[:, :, None, None].float()
            loss = (se * m).sum() / (m.sum() * se.shape[2] * se.shape[3] + 1e-8)
        else:
            loss = se.mean()

        logs = {"fm": loss.item(),
                "frac_high_noise": hi.float().mean().item()}
        return loss, logs

    # ---- サンプリング ------------------------------------------------
    @torch.no_grad()
    def sample(self, n: int, T: int, steps: int = 50,
               cond: torch.Tensor | None = None,
               cond_mask: torch.Tensor | None = None,
               guidance_scale: float = 1.0,
               lengths: torch.Tensor | Sequence[int] | None = None,
               solver: str = "heun",
               decode: str = "full", trim: bool = True, device=None):
        """
        cond: (n, S, cond_dim) テキストトークン列. None で無条件生成
        cond_mask: (n, S) True=有効トークン
        guidance_scale: CFG 係数. 無条件分岐は null_cond / null_context を使用
        solver: "heun" (2次, 1stepあたり velocity 評価2回) or "euler" (1次).
                同一計算量では Heun steps=N と Euler steps=2N が対応する.
        """
        device = device or next(self.parameters()).device
        use_cfg = (cond is not None and guidance_scale != 1.0
                   and self.cond_dim is not None)
        if lengths is not None:
            lengths = torch.as_tensor(lengths, device=device)
            T = int(lengths.max().item())
        T_lat = max(1, T // self.vae.t_stride)
        valid = self._length_mask(lengths, T_lat, device)
        time_pad = ~valid if valid is not None else None

        z = torch.randn(n, T_lat, self.num_tokens, self.latent_dim,
                        device=device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

        def get_velocity(z_in, t_in):
            """CFG 込みの速度場. Heun ではこれを1つの場として積分する."""
            if use_cfg:
                v_c = self.velocity(z_in, t_in, context=cond,
                                    context_mask=cond_mask, time_pad=time_pad)
                # [FIX-2] 無条件分岐: context=None -> null_cond / null_context
                v_u = self.velocity(z_in, t_in, context=None,
                                    time_pad=time_pad)
                return v_u + guidance_scale * (v_c - v_u)
            return self.velocity(z_in, t_in, context=cond,
                                 context_mask=cond_mask, time_pad=time_pad)

        # ---- (旧) Euler 法: 1次精度, 全体誤差 O(dt) --------------------
        # for i in range(steps):
        #     t_now = ts[i].expand(n)
        #     if use_cfg:
        #         v_c = self.velocity(z, t_now, context=cond,
        #                             context_mask=cond_mask, time_pad=time_pad)
        #         # [FIX-2] 無条件分岐: context=None -> null_cond / null_context
        #         v_u = self.velocity(z, t_now, context=None, time_pad=time_pad)
        #         v = v_u + guidance_scale * (v_c - v_u)
        #     else:
        #         v = self.velocity(z, t_now, context=cond,
        #                           context_mask=cond_mask, time_pad=time_pad)
        #     #Euler ステップで z を更新
        #     z = z - (ts[i] - ts[i + 1]) * v

        # ---- Heun 法 (predictor-corrector): 2次精度, 全体誤差 O(dt^2) --
        # predictor: 出発点の速度 v1 で Euler 仮ステップ
        # corrector: 到達点で速度 v2 を再評価し, 両端平均 (台形則) で進み直す
        # 最終ステップのみ Euler で終える (t=0 での速度評価は訓練分布の端で
        # 不安定なため. EDM (Karras et al. 2022) 以来の慣例)
        for i in range(steps):
            t_now = ts[i].expand(n)
            dt = ts[i] - ts[i + 1]                       # > 0

            v1 = get_velocity(z, t_now)
            z_pred = z - dt * v1                         # predictor (Euler)

            if solver == "euler" or i == steps - 1:
                z = z_pred
            else:
                t_next = ts[i + 1].expand(n)
                v2 = get_velocity(z_pred, t_next)
                z = z - dt * 0.5 * (v1 + v2)             # corrector (台形則)

        def _trim(x):
            if lengths is None or not trim:
                return x
            return [x[i, :int(L)] for i, L in enumerate(lengths)]

        if decode == "full" or self.part_decoders is None:
            d6 = self._decode(z, T_target=T)
            if decode == "full":
                return _trim(d6)
            return {name: _trim(d6[..., idx])
                    for name, idx in self.part_index.items()}
        stats = {"body": (self.z_mean_body, self.z_std_body),
                 "left": (self.z_mean_hand, self.z_std_hand),
                 "right": (self.z_mean_hand, self.z_std_hand)}
        return {name: _trim(self.part_decoders[name].decode(
                    z[:, :, idx] * stats[name][1] + stats[name][0],
                    T_target=T))
                for name, idx in self.part_index.items()}

    # ---- ルーティング変更 API ---------------------------------------
    def set_boundary(self, boundary: float):
        self.router.boundary = boundary

    def set_router(self, fn: Callable[[torch.Tensor], torch.Tensor] | None):
        self.router.set_rule(fn)


# ----------------------------------------------------------------------
# 7. 動作確認
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        from models.module.transformer6d_vae import HandTransformerVAE
    except ImportError:
        from hand_transformer_vae_6d import HandTransformerVAE

    torch.manual_seed(0)
    ALL_BONES_DUMMY = [(0, i + 1, 0) for i in range(47)]

    d6 = torch.randn(4, 32, 6, 47)
    lengths = torch.tensor([32, 24, 16, 8])
    text_tokens = torch.randn(4, 12, 512)                # (B, S, cond_dim)
    text_mask = torch.ones(4, 12, dtype=torch.bool)
    text_mask[1, 8:] = False                             # 可変長の文

    for use_xattn in [False, True]:
        print(f"\n=== cross_attn={use_xattn} ===")
        vae = HandTransformerVAE(bones=ALL_BONES_DUMMY, d_model=64, n_heads=4,
                                 n_stages=1, latent_dim=16)
        model = SignLatentFlowMoE(vae, boundary=0.875,
                                  d_model=128, n_heads=8,
                                  depth_high=3, depth_low=2,
                                  cond_dim=512, cross_attn=use_xattn)
        model.set_latent_stats(d6, lengths=lengths)

        # 学習ステップ (トークン列 + マスク + CFGドロップ + 可変長)
        # テスト用: 両エキスパートに必ずデータが流れるようルーティングを固定
        model.set_router(lambda t: t >= t.median())
        loss, logs = model.training_loss(
            d6, lengths=lengths, cond=text_tokens, cond_mask=text_mask,
            cond_drop_prob=0.5,
            part_weight={"body": 1.0, "left": 2.0, "right": 2.0})
        loss.backward()
        print("loss:", logs)
        g_hi = model.expert_high.proj_out.weight.grad
        g_lo = model.experts_low["left"].proj_out.weight.grad
        print("grad high:", g_hi.abs().sum().item() > 0,
              " grad low:", g_lo.abs().sum().item() > 0)

        # サンプリング (CFG / 無条件 / 可変長)
        model.set_router(None)   # デフォルト (boundary) に戻す
        model.eval()
        gen = model.sample(n=2, T=32, steps=6, cond=text_tokens[:2],
                           cond_mask=text_mask[:2], guidance_scale=3.0)
        print("sampled CFG    :", tuple(gen.shape))
        gen_u = model.sample(n=2, T=32, steps=6)
        print("sampled uncond :", tuple(gen_u.shape))
        gen_var = model.sample(n=3, T=32, steps=6, lengths=[32, 20, 12],
                               cond=text_tokens[:3], cond_mask=text_mask[:3],
                               guidance_scale=2.0)
        print("sampled varlen :", [tuple(g.shape) for g in gen_var])
        assert not any(torch.isnan(g).any() for g in gen_var)
    print("\nall tests passed")