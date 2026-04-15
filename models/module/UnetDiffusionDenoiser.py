import math
import torch
from torch import nn
import torch.nn.functional as F

# =========================================================
# utility
# =========================================================
def masked_mean(x, mask, dim):
    # x: (...), mask: True=pad
    valid = (~mask).float()
    while valid.dim() < x.dim():
        valid = valid.unsqueeze(-1)
    return (x * valid).sum(dim=dim) / valid.sum(dim=dim).clamp(min=1.0)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        t = t.float()
        emb_scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# =========================================================
# 1D UNet blocks
# =========================================================
class TimeFiLM(nn.Module):
    """
    time embedding -> scale, shift
    """
    def __init__(self, time_dim, channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels * 2)
        )

    def forward(self, x, t_emb):
        # x: (B,C,T), t_emb: (B,time_dim)
        scale_shift = self.mlp(t_emb)  # (B,2C)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        return x * (1 + scale) + shift


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_film = TimeFiLM(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb, padding_mask=None):
        # x: (B,C,T), padding_mask: (B,T), True=pad
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.time_film(h, t_emb)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        out = h + self.skip(x)

        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(1).float()
            out = out * valid

        return out


class CrossAttention1D(nn.Module):
    """
    sequence feature (B,C,T) <-> text feature (B,L,D)
    """
    def __init__(self, channels, text_dim, nhead=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.text_proj = nn.Linear(text_dim, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, x, text_tokens, text_padding_mask=None, x_padding_mask=None):
        # x: (B,C,T) -> (B,T,C)
        x_seq = x.transpose(1, 2)
        q = self.norm(x_seq)
        kv = self.text_proj(text_tokens)

        attn_out, _ = self.attn(
            q, kv, kv,
            key_padding_mask=text_padding_mask,
            need_weights=False
        )
        x_seq = x_seq + attn_out
        x_seq = x_seq + self.out(x_seq)

        if x_padding_mask is not None:
            valid = (~x_padding_mask).unsqueeze(-1).float()
            x_seq = x_seq * valid

        return x_seq.transpose(1, 2)


class Downsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, padding_mask=None):
        x = self.conv(x)
        T_out = x.size(-1)
        new_mask = None
        if padding_mask is not None:
            B, T = padding_mask.shape
            # Conv出力の実サイズに合わせてマスクを調整
            T_needed = T_out * 2
            if T < T_needed:
                padding_mask = F.pad(padding_mask.float(), (0, T_needed - T), value=1.0).bool()
            else:
                padding_mask = padding_mask[:, :T_needed]
            new_mask = padding_mask.view(B, T_out, 2).all(dim=-1)
            valid = (~new_mask).unsqueeze(1).float()
            x = x * valid
        return x, new_mask


class Upsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, target_len=None, padding_mask=None):
        x = self.conv(x)
        if target_len is not None and x.size(-1) != target_len:
            x = F.interpolate(x, size=target_len, mode="nearest")

        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(1).float()
            x = x * valid
        return x


# =========================================================
# UNet denoiser
# =========================================================
class UNetDiffusionDenoiser(nn.Module):
    """
    Input:  (B,T,z_dim)
    Output: (B,T,z_dim)
    """
    def __init__(
        self,
        latent_dim,
        model_dim,
        time_dim,
        nhead,
        text_cond_dim,
        dropout=0.1,
        channel_mults=(1, 2, 4),
        use_attn_levels=(True, True, True),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.time_dim = time_dim

        self.in_proj = nn.Conv1d(latent_dim, model_dim, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        chs = [model_dim * m for m in channel_mults]

        # down path
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_ch = model_dim
        for i, ch in enumerate(chs):
            block1 = ResBlock1D(in_ch, ch, time_dim, dropout)
            block2 = ResBlock1D(ch, ch, time_dim, dropout)
            self.down_blocks.append(nn.ModuleList([block1, block2]))

            if use_attn_levels[i]:
                self.down_attns.append(CrossAttention1D(ch, text_cond_dim, nhead=nhead, dropout=dropout))
            else:
                self.down_attns.append(nn.Identity())

            if i < len(chs) - 1:
                self.downsamples.append(Downsample1D(ch))
            in_ch = ch

        # bottleneck
        mid_ch = chs[-1]
        self.mid_block1 = ResBlock1D(mid_ch, mid_ch, time_dim, dropout)
        self.mid_attn = CrossAttention1D(mid_ch, text_cond_dim, nhead=nhead, dropout=dropout)
        self.mid_block2 = ResBlock1D(mid_ch, mid_ch, time_dim, dropout)

        # up path
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()

        rev_chs = list(reversed(chs))
        cur_ch = rev_chs[0]
        for i, ch in enumerate(rev_chs[1:]):
            self.upsamples.append(Upsample1D(cur_ch))

            # skip concatするので channels は cur_ch + ch
            block1 = ResBlock1D(cur_ch + ch, ch, time_dim, dropout)
            block2 = ResBlock1D(ch, ch, time_dim, dropout)
            self.up_blocks.append(nn.ModuleList([block1, block2]))
            self.up_attns.append(CrossAttention1D(ch, text_cond_dim, nhead=nhead, dropout=dropout))
            cur_ch = ch

        self.out_norm = nn.GroupNorm(8, cur_ch)
        self.out_conv = nn.Conv1d(cur_ch, latent_dim, kernel_size=3, padding=1)

    def forward(
        self,
        x_t,                      # (B,T,z_dim)
        t,                        # (B,)
        text_tokens,              # (B,L,H)
        x_padding_mask=None,      # (B,T) True=pad
        text_padding_mask=None,   # (B,L) True=pad
        src_mask=None,            # unused in UNet
    ):
        del src_mask

        # (B,T,C) -> (B,C,T)
        x = x_t.transpose(1, 2)
        x = self.in_proj(x)

        if x_padding_mask is not None:
            valid = (~x_padding_mask).unsqueeze(1).float()
            x = x * valid

        t_emb = self.time_mlp(t)

        skips = []
        masks = []

        # -------------------------
        # down
        # -------------------------
        cur_mask = x_padding_mask
        for i, blocks in enumerate(self.down_blocks):
            x = blocks[0](x, t_emb, cur_mask)
            x = blocks[1](x, t_emb, cur_mask)

            attn = self.down_attns[i]
            if not isinstance(attn, nn.Identity):
                x = attn(x, text_tokens, text_padding_mask=text_padding_mask, x_padding_mask=cur_mask)

            skips.append(x)
            masks.append(cur_mask)

            if i < len(self.downsamples):
                x, cur_mask = self.downsamples[i](x, cur_mask)

        # -------------------------
        # mid
        # -------------------------
        x = self.mid_block1(x, t_emb, cur_mask)
        x = self.mid_attn(x, text_tokens, text_padding_mask=text_padding_mask, x_padding_mask=cur_mask)
        x = self.mid_block2(x, t_emb, cur_mask)

        # -------------------------
        # up
        # -------------------------
        for i, blocks in enumerate(self.up_blocks):
            skip = skips[-(i + 2)]   # 最深部以外をskipに使う
            skip_mask = masks[-(i + 2)]

            x = self.upsamples[i](x, target_len=skip.size(-1), padding_mask=skip_mask)
            x = torch.cat([x, skip], dim=1)

            x = blocks[0](x, t_emb, skip_mask)
            x = blocks[1](x, t_emb, skip_mask)
            x = self.up_attns[i](x, text_tokens, text_padding_mask=text_padding_mask, x_padding_mask=skip_mask)

            cur_mask = skip_mask

        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)

        if cur_mask is not None:
            valid = (~cur_mask).unsqueeze(1).float()
            x = x * valid

        return x.transpose(1, 2)  # (B,T,z_dim)