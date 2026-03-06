import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """
    RoPE for a given head_dim (must be even).
    Applies rotary position embedding to q,k: shape (B, H, T, Dh)
    """
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_sin_cos(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        # positions: (T,)
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,d->td", t, self.inv_freq)  # (T, Dh/2)
        sin = freqs.sin().to(dtype=dtype)  # (T, Dh/2)
        cos = freqs.cos().to(dtype=dtype)  # (T, Dh/2)
        return sin, cos

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # x: (..., Dh) where Dh is even; split into pairs
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # (x1, x2) -> (-x2, x1)
        out = torch.stack((-x2, x1), dim=-1)  # (..., Dh/2, 2)
        return out.flatten(-2)                # (..., Dh)

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q,k: (B, H, T, Dh)
        """
        B, H, T, Dh = q.shape
        sin, cos = self._build_sin_cos(T, q.device, q.dtype)  # (T, Dh/2)

        # Expand to (1,1,T,Dh/2) and interleave to match Dh pairs
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1,1,T,Dh/2)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,Dh/2)

        # For interleaved dims, broadcast sin/cos to Dh by pairing
        # We compute:
        # x_rot = x * cos + rotate_half(x) * sin
        # Need sin/cos shaped to (..., Dh) interleaved
        sin_full = torch.repeat_interleave(sin, 2, dim=-1)  # (1,1,T,Dh)
        cos_full = torch.repeat_interleave(cos, 2, dim=-1)  # (1,1,T,Dh)

        q_out = q * cos_full + self._rotate_half(q) * sin_full
        k_out = k * cos_full + self._rotate_half(k) * sin_full
        return q_out, k_out
class MultiheadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """
        x: (B, T, d_model)
        attn_mask: (T, T) bool where True means "mask out" (future), optional
        """
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, H, T, Dh)
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # Apply RoPE to q,k
        q, k = self.rope.apply(q, k)

        # Scaled dot-product attention
        # scores: (B,H,T,T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            # attn_mask True => -inf
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B,H,T,Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B,T,D)
        out = self.out(out)
        return out
class DecoderLayerRoPE(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiheadSelfAttentionRoPE(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,                 # (B,T,D)
        memory: torch.Tensor,              # (B,S,D)
        tgt_mask: torch.Tensor | None = None,      # (T,T) bool (causal)
        memory_key_padding_mask: torch.Tensor | None = None,  # (B,S) bool True=PAD
    ):
        # Self-attn (RoPE)
        x = tgt
        x2 = self.self_attn(self.norm1(x), attn_mask=tgt_mask)
        x = x + self.dropout(x2)

        # Cross-attn (standard MHA): query=tgt, key/value=memory
        q = self.norm2(x)
        x2, _ = self.cross_attn(
            query=q,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,  # True=PAD
            need_weights=False,
        )
        x = x + self.dropout(x2)

        # FFN
        x2 = self.ff(self.norm3(x))
        x = x + x2
        return x
