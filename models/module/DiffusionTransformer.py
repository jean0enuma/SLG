import math
import torch
from torch import nn
import torch.nn.functional as F

# -------------------------
# utils: sinusoidal timestep embedding
# -------------------------
def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (B,) int64/float
    return: (B, dim)
    """
    half = dim // 2
    device = t.device
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device).float() / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

def exists(x):
    return x is not None

# -------------------------
# Transformer blocks (self-attn + optional cross-attn) with AdaLN-like conditioning
# -------------------------
class AdaLN(nn.Module):
    """
    Adaptive LayerNorm: LN(x) * (1+scale) + shift
    scale, shift computed from conditioning vector c (B, C)
    """
    def __init__(self, hidden_dim: int, cond_dim: int,scale_only: bool = False):
        super().__init__()
        self.scale_only = scale_only
        if scale_only:
            self.to_scale_shift = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, hidden_dim),
                #nn.Tanh()  # restrict scale to [-1,1]
            )
        else:
            self.to_scale_shift = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, 2 * hidden_dim)
            )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        # x: (B,T,H), c: (B,C)
        ss = self.to_scale_shift(c)  # (B, 2H)
        if self.scale_only:
            scale=ss
            return x * (1.0 + scale.unsqueeze(1))
        else:
            scale, shift = ss.chunk(2, dim=-1)
            return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
class AdaLNZero(nn.Module):
    """
    AdaLN-Zero module.

    Given:
      x: (B, T, C) or (B, ..., C)  ※最後の次元がチャネル
      cond: (B, D)                ※条件（timestep埋め込み+テキスト埋め込み等をまとめたもの）

    Computes:
      y = LN(x) * (1 + scale(cond)) + shift(cond)
      gate = gate(cond)

    'Zero' means the last linear layer is zero-initialized so that
    at init: scale=0, shift=0, gate=0  (=> block behaves like identity when used properly)
    """
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        eps: float = 1e-6,
        use_gate: bool = True,
        gate_tanh: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.cond_dim = cond_dim
        self.use_gate = use_gate
        self.gate_tanh = gate_tanh


        # Produce (shift, scale, gate) from condition
        out_dim = 2 * channels + (channels if use_gate else 0)

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_dim, bias=True),
        )

        # Zero-init last Linear so outputs are all zero at start
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        Returns:
          x_mod: normalized+modulated x
          gate: (optional) gating tensor same shape as x (or broadcastable)
        """
        if cond.dim() != 2:
            raise ValueError(f"cond must be (B, D). Got {tuple(cond.shape)}")
        if x.size(0) != cond.size(0):
            raise ValueError(f"Batch mismatch: x B={x.size(0)} vs cond B={cond.size(0)}")
        if x.size(-1) != self.channels:
            raise ValueError(f"x last dim must be channels={self.channels}, got {x.size(-1)}")

        h = x # (B, ..., C)

        params = self.mlp(cond)  # (B, out_dim)
        if self.use_gate:
            shift, scale, gate = torch.split(
                params, [self.channels, self.channels, self.channels], dim=-1
            )
        else:
            shift, scale = torch.split(params, [self.channels, self.channels], dim=-1)
            gate = None

        # Broadcast to match x shape: (B, 1, 1, ..., C)
        # x may be (B, T, C) or (B, H, W, C) etc.
        while shift.dim() < x.dim():
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
            if gate is not None:
                gate = gate.unsqueeze(1)

        x_mod = h * (1.0 + scale) + shift

        if gate is not None and self.gate_tanh:
            gate = torch.tanh(gate)

        return x_mod, gate

class TransformerDenoiserBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, n_heads: int, dropout=0.0, use_cross_attn: bool = True):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.ln1=nn.LayerNorm(hidden_dim)
        self.adaln1 = AdaLN(hidden_dim, cond_dim)
        self.adaln1_scale_only = AdaLN(hidden_dim, cond_dim, scale_only=True)  # for self-attn only scaling
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        if use_cross_attn:
            self.ln2 = nn.LayerNorm(hidden_dim)
            self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            #self.adaln2_scale_only = AdaLN(hidden_dim, cond_dim, scale_only=True)  # for cross-attn only scaling
            self.drop2 = nn.Dropout(dropout)
        self.ln3=nn.LayerNorm(hidden_dim)
        self.adaln3 = AdaLN(hidden_dim, cond_dim)
        self.adaln3_scale_only = AdaLN(hidden_dim, cond_dim, scale_only=True)  # for mlp only scaling
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,          # (B,T,H)
        c: torch.Tensor,          # (B,C)
        cond: torch.Tensor | None = None,   # (B,S,H) for cross-attn memory
        x_key_padding_mask: torch.Tensor | None = None,   # (B,T) bool (True=pad)
        cond_key_padding_mask: torch.Tensor | None = None # (B,S) bool
    ):
        # self-attn
        h=self.ln1(x)
        h = self.adaln1(h, c)
        attn_out, _ = self.self_attn(h, h, h, key_padding_mask=x_key_padding_mask, need_weights=False)
        attn_out=self.drop1(attn_out)
        x = x +self.adaln1_scale_only(attn_out,c) # only scale for self-attn

        # cross-attn (optional)
        if self.use_cross_attn and exists(cond):
            #h = self.adaln2(x, c)
            h=self.ln2(x)
            attn_out, _ = self.cross_attn(h, cond, cond, key_padding_mask=cond_key_padding_mask, need_weights=False)
            #attn_out=self.adaln2_scale_only(attn_out, c)  # only scale for cross-attn
            x = x + self.drop2(attn_out)

        # mlp
        h=self.ln3(x)
        h = self.adaln3(h, c)
        mlp_out=self.drop3(self.mlp(h))
        x = x + self.adaln3_scale_only(mlp_out,c)  # only scale for mlp
        return x

# -------------------------
# Diffusion Transformer (DiT-like) for 1D sequences
# -------------------------
class DiffusionTransformer(nn.Module):
    """
    Input:  x_t (B,T,D)
    Output: eps or x0 (B,T,D) depending on objective
    Optional cond: (B,S,cond_in_dim) -> projected then cross-attn
    """
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 512,
        depth: int = 8,
        n_heads: int = 8,
        time_embed_dim: int = 256,
        cond_in_dim: int | None = None,
        dropout: float = 0.0,
        objective: str = "eps",  # "eps" or "x0"
        use_cross_attn: bool = True,
        max_len: int = 2048,
    ):
        super().__init__()
        assert objective in ["eps", "x0","v0"]
        self.objective = objective
        self.hidden_dim = hidden_dim

        self.in_proj = nn.Linear(data_dim, hidden_dim)

        # learned positional embedding (simpler & strong for 1D seq)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, hidden_dim))

        # time embedding -> cond vector c
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # optional conditioning projection (text embeddings etc.)
        self.use_cross_attn = use_cross_attn and (cond_in_dim is not None)
        self.cond_proj = nn.Linear(cond_in_dim, hidden_dim) if (cond_in_dim is not None) else None

        # also allow an extra global condition vector (e.g., language id embedding)
        self.extra_cond_dim = hidden_dim
        self.extra_cond_proj = nn.Identity()

        # combine time cond + extra cond into c
        # c = time_c + extra_c
        self.blocks = nn.ModuleList([
            TransformerDenoiserBlock(
                hidden_dim=hidden_dim,
                cond_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
                use_cross_attn=self.use_cross_attn,
            )
            for _ in range(depth)
        ])

        self.out_ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, data_dim)

        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(
        self,
        x_t: torch.Tensor,                  # (B,T,D)
        t: torch.Tensor,                    # (B,)
        cond: torch.Tensor | None = None,   # (B,S,cond_in_dim)
        extra_cond: torch.Tensor | None = None,  # (B,H) e.g., language embedding already projected
        x_key_padding_mask: torch.Tensor | None = None,
        cond_key_padding_mask: torch.Tensor | None = None,
    ):
        B, T, _ = x_t.shape
        assert T <= self.pos_emb.shape[1], f"T={T} exceeds max_len={self.pos_emb.shape[1]}"

        x = self.in_proj(x_t) + self.pos_emb[:, :T, :]

        # time cond
        te = sinusoidal_timestep_embedding(t, self.time_mlp[0].in_features)  # (B, time_embed_dim)
        c = self.time_mlp(te)  # (B, H)

        if exists(extra_cond):
            # extra_cond should be (B,H) already; if not, project outside or replace Identity with Linear
            c = c + self.extra_cond_proj(extra_cond)

        # conditioning memory
        mem = None
        if self.use_cross_attn and exists(cond):
            mem = self.cond_proj(cond)  # (B,S,H)

        for blk in self.blocks:
            x = blk(
                x=x,
                c=c,
                cond=mem,
                x_key_padding_mask=x_key_padding_mask,
                cond_key_padding_mask=cond_key_padding_mask
            )

        x = self.out_ln(x)
        return self.out_proj(x)  # (B,T,D)