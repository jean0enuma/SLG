import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.AdaLN import AdaLN


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FactorizedSelfAttention(nn.Module):
    """
    Factorized self-attention:
      1) spatial attention over S = H*W for each frame
      2) temporal attention over T for each spatial position

    Input:
        x: (B, T, S, D)

    Output:
        x: (B, T, S, D)
    """
    def __init__(
        self,
        dim: int,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, S, D)
        """
        B, T, S, D = x.shape

        # -----------------------------------
        # spatial attention
        # each frame independently: (B*T, S, D)
        # -----------------------------------
        x_sp = x.reshape(B * T, S, D)
        h_sp, _ = self.spatial_attn(x_sp, x_sp, x_sp, need_weights=False)
        x = x + h_sp.view(B, T, S, D)

        # -----------------------------------
        # temporal attention
        # each spatial position independently: (B*S, T, D)
        # -----------------------------------
        x_tm = x.permute(0, 2, 1, 3).reshape(B * S, T, D)
        h_tm, _ = self.temporal_attn(x_tm, x_tm, x_tm, need_weights=False)
        h_tm = h_tm.view(B, S, T, D).permute(0, 2, 1, 3)

        x = x + h_tm
        return x


class AdaLNCrossTextSTTransformerBlock(nn.Module):
    """
    Spatial-Temporal factorized version of AdaLNCrossTextTransformerBlock

    Inputs
    ------
    x:                     (B, T*S, D)
    cond_emb:              (B, cond_dim)
    text_tokens:           (B, L, D)
    text_key_padding_mask: (B, L), True=padding

    Extra args
    ----------
    T: int
    S: int   # H*W
    """
    def __init__(
        self,
        d_model: int,
        cond_dim: int,
        nhead: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)

        self.adaln1 = AdaLN(d_model=d_model, time_dim=cond_dim)
        self.adaln2 = AdaLN(d_model=d_model, time_dim=cond_dim)
        self.adaln3 = AdaLN(d_model=d_model, time_dim=cond_dim)

        self.factorized_self_attn = FactorizedSelfAttention(
            dim=d_model,
            nhead=nhead,
            dropout=dropout,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.ff = FeedForward(d_model, mult=ff_mult, dropout=dropout)

    @staticmethod
    def modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale) + shift

    def forward(
        self,
        x: torch.Tensor,                                # (B, T*S, D)
        cond_emb: torch.Tensor,                         # (B, cond_dim)
        text_tokens: torch.Tensor,                      # (B, L, D)
        text_key_padding_mask: torch.Tensor = None,     # (B, L)
        T: int = None,
        S: int = None,
    ) -> torch.Tensor:
        if T is None or S is None:
            raise ValueError("T and S must be provided.")
        B, N, D = x.shape
        if N != T * S:
            raise ValueError(f"N={N} does not match T*S={T*S}")

        # ---------------------------------------
        # 1) factorized self-attention
        # ---------------------------------------
        h = self.norm1(x)
        scale1, shift1, gate1 = self.adaln1(h, cond_emb)
        h = self.modulate(h, scale1, shift1)

        h_4d = h.view(B, T, S, D)
        h_sa = self.factorized_self_attn(h_4d).view(B, N, D)
        x = x + gate1 * h_sa

        # ---------------------------------------
        # 2) cross-attention
        # ---------------------------------------
        h = self.norm2(x)
        scale2, shift2, gate2 = self.adaln2(h, cond_emb)
        h = self.modulate(h, scale2, shift2)

        h_ca, _ = self.cross_attn(
            query=h,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )
        x = x + gate2 * h_ca

        # ---------------------------------------
        # 3) FFN
        # ---------------------------------------
        h = self.norm3(x)
        scale3, shift3, gate3 = self.adaln3(h, cond_emb)
        h = self.modulate(h, scale3, shift3)

        h_ff = self.ff(h)
        x = x + gate3 * h_ff

        return x