import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.AdaLN import AdaLN


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    timesteps: (B,)
    return: (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# =========================================================
# Basic blocks
# =========================================================

class ConvGNAct3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class CrossAttentionText(nn.Module):
    """
    Query: 3D feature tokens
    Key/Value: text tokens

    x:          (B, C, T, H, W)
    text_tokens:(B, L, D)
    """
    def __init__(self, feat_dim: int, text_dim: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.kv_proj = nn.Linear(text_dim, feat_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(feat_dim, feat_dim)

    def forward(
        self,
        x: torch.Tensor,                              # (B, C, T, H, W)
        text_h: torch.Tensor,                         # (B, L, text_dim)
        attention_mask: Optional[torch.Tensor] = None # (B, L), 1=valid, 0=pad
    ) -> torch.Tensor:
        B, C, T, H, W = x.shape
        N = T * H * W

        x_tokens = x.permute(0, 2, 3, 4, 1).reshape(B, N, C)    # (B,N,C)
        q = self.q_proj(x_tokens)
        kv = self.kv_proj(text_h)                                # (B,L,C)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        out, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = self.out_proj(out)
        out = out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        return out


class AdaLNResBlock3D(nn.Module):
    """
    3D ResBlock with AdaLN modulation.
    Condition = fused(time_emb + text_emb)

    x:        (B, C, T, H, W)
    cond_emb: (B, cond_dim)
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(num_groups=min(num_groups, in_ch), num_channels=in_ch, affine=False)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch, affine=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU()

    @staticmethod
    def _reshape_adaln(scale, shift, gate):
        # AdaLN returns (B,1,D) for token sequence; convert to conv format
        # target conv feature: (B, C, T, H, W)
        scale = scale.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1,1)
        shift = shift.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        gate = gate.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        return scale, shift, gate

    def modulate(self, x, scale, shift):
        return x * (1.0 + scale) + shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        h2 = self.norm2(h)
        h2 = self.act(h2)
        h2 = self.dropout(h2)
        h2 = self.conv2(h2)

        return residual + h2


class UNetBlock3D(nn.Module):
    """
    ResBlock + optional cross attention
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        text_hidden_dim: int,
        use_cross_attn: bool = True,
        nhead: int = 8,
        num_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.resblock = AdaLNResBlock3D(
            in_ch=in_ch,
            out_ch=out_ch,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = CrossAttentionText(
                feat_dim=out_ch,
                text_dim=text_hidden_dim,
                nhead=nhead,
                dropout=dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        text_h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.resblock(x)
        if self.use_cross_attn:
            x = x + self.cross_attn(x, text_h, attention_mask)
        return x


class Downsample3D(nn.Module):
    """
    Downsample only spatial dimensions: H,W / 2
    Keep T unchanged
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    """
    Upsample only spatial dimensions: H,W x 2
    Keep T unchanged
    """
    def __init__(self, channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(
            channels, channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

    def forward(self, x):
        return self.deconv(x)


# =========================================================
# UNet Denoiser
# =========================================================

class AdaLNCrossTextUNetDenoiser(nn.Module):
    """
    UNet version denoiser for latent diffusion.

    Inputs
    ------
    xt:         (B, T, Cz, Hz, Wz)
    t:          (B,)
    text_emb:   (B, text_emb_dim)
    text_h:     (B, L, text_hidden_dim)
    attention_mask: (B, L), 1=valid

    Output
    ------
    pred_noise: (B, T, Cz, Hz, Wz)
    """
    def __init__(
        self,
        latent_ch: int,
        text_emb_dim: int,
        text_hidden_dim: int,
        base_ch: int = 128,
        channel_mults: List[int] = [1, 2, 4],
        num_res_blocks: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
        time_embed_dim: int = 512,
        cond_dim: int = 512,
        num_groups: int = 8,
        use_cross_attn_in_all_levels: bool = True,
    ):
        super().__init__()
        self.latent_ch = latent_ch
        self.base_ch = base_ch
        self.time_embed_dim = time_embed_dim
        self.cond_dim = cond_dim

        # input projection
        self.in_proj = nn.Conv3d(latent_ch, base_ch, kernel_size=3, padding=1)

        # global condition from time + text_emb
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.text_emb_mlp = nn.Sequential(
            nn.Linear(text_emb_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_fuse = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # -------------------------
        # encoder
        # -------------------------
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels = []

        cur_ch = base_ch
        for i, mult in enumerate(channel_mults):
            out_ch = base_ch * mult
            stage = nn.ModuleList()

            for _ in range(num_res_blocks):
                stage.append(
                    UNetBlock3D(
                        in_ch=cur_ch,
                        out_ch=out_ch,
                        text_hidden_dim=text_hidden_dim,
                        use_cross_attn=use_cross_attn_in_all_levels,
                        nhead=nhead,
                        num_groups=num_groups,
                        dropout=dropout,
                    )
                )
                cur_ch = out_ch
                self.skip_channels.append(cur_ch)

            self.down_blocks.append(stage)

            if i != len(channel_mults) - 1:
                self.downsamples.append(Downsample3D(cur_ch))

        # -------------------------
        # middle
        # -------------------------
        self.mid_block1 = UNetBlock3D(
            in_ch=cur_ch,
            out_ch=cur_ch,
            text_hidden_dim=text_hidden_dim,
            use_cross_attn=True,
            nhead=nhead,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.mid_block2 = UNetBlock3D(
            in_ch=cur_ch,
            out_ch=cur_ch,
            text_hidden_dim=text_hidden_dim,
            use_cross_attn=True,
            nhead=nhead,
            num_groups=num_groups,
            dropout=dropout,
        )

        # -------------------------
        # decoder
        # -------------------------
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        reversed_mults = list(reversed(channel_mults))
        skip_channels = list(reversed(self.skip_channels))

        for i, mult in enumerate(reversed_mults):
            out_ch = base_ch * mult
            stage = nn.ModuleList()

            for _ in range(num_res_blocks):
                skip_ch = skip_channels.pop(0)
                stage.append(
                    UNetBlock3D(
                        in_ch=cur_ch + skip_ch,
                        out_ch=out_ch,
                        text_hidden_dim=text_hidden_dim,
                        use_cross_attn=use_cross_attn_in_all_levels,
                        nhead=nhead,
                        num_groups=num_groups,
                        dropout=dropout,
                    )
                )
                cur_ch = out_ch

            self.up_blocks.append(stage)

            if i != len(reversed_mults) - 1:
                self.upsamples.append(Upsample3D(cur_ch))

        # output
        self.out_norm = nn.GroupNorm(num_groups=min(num_groups, cur_ch), num_channels=cur_ch)
        self.out_act = nn.SiLU()
        self.out_proj = nn.Conv3d(cur_ch, latent_ch, kernel_size=3, padding=1)

    def make_global_condition(self, t: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)
        txt_emb = self.text_emb_mlp(text_emb)
        cond = self.cond_fuse(torch.cat([t_emb, txt_emb], dim=-1))
        return cond

    def forward(
        self,
        xt: torch.Tensor,                          # (B, T, Cz, Hz, Wz)
        t: torch.Tensor,                           # (B,)
        text_emb: torch.Tensor,                    # (B, text_emb_dim)
        text_h: torch.Tensor,                      # (B, L, text_hidden_dim)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # rearrange to 3D conv format
        x = xt.permute(0, 2, 1, 3, 4)             # (B, Cz, T, H, W)
        cond = self.make_global_condition(t, text_emb)

        x = self.in_proj(x)

        skips = []
        downsample_idx = 0

        # encoder
        for i, stage in enumerate(self.down_blocks):
            for block in stage:
                x = block(x,  text_h, attention_mask)
                skips.append(x)
            if i != len(self.down_blocks) - 1:
                x = self.downsamples[downsample_idx](x)
                downsample_idx += 1

        # middle
        x = self.mid_block1(x, text_h, attention_mask)
        x = self.mid_block2(x, text_h, attention_mask)

        # decoder
        upsample_idx = 0
        for i, stage in enumerate(self.up_blocks):
            for block in stage:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, text_h, attention_mask)
            if i != len(self.up_blocks) - 1:
                x = self.upsamples[upsample_idx](x)
                upsample_idx += 1

        x = self.out_proj(self.out_act(self.out_norm(x)))

        # back to latent format
        x = x.permute(0, 2, 1, 3, 4)              # (B, T, Cz, H, W)
        return x