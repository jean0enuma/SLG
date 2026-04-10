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
    def __init__(self, hidden_dim: int, cond_dim: int,scale_shft_gate=(True, True, False)):
        super().__init__()
        self.scale_shift_gate = scale_shft_gate
        num_apply= sum(scale_shft_gate)
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_apply * hidden_dim)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        # x: (B,T,H), c: (B,C)
        ss = self.to_scale_shift(c)  # (B, 2H)
        ss_chunk=ss.chunk(sum(self.scale_shift_gate), dim=-1)
        scale,shift,gate=None,None,None
        idx=0
        if self.scale_shift_gate[0]:
            scale=ss_chunk[idx]
            idx+=1
        if self.scale_shift_gate[1]:
            shift=ss_chunk[idx]
            idx+=1
        if self.scale_shift_gate[2]:
            gate=ss_chunk[idx]
        # Apply LN
        x_ln = F.layer_norm(x, x.shape[2:])  # (B,T,H)
        if scale is not None:
            x_ln = x_ln * (1.0 + scale.unsqueeze(1))  # (B,T,H)
        if shift is not None:
            x_ln = x_ln + shift.unsqueeze(1)  # (B,T,H)
        if gate is not None:
            gate = gate.unsqueeze(1)  # (B,1,H) for broadcasting
        return x_ln, gate

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

