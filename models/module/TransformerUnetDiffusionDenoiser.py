import math
import torch
from torch import nn
import torch.nn.functional as F


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


class AdaLN(nn.Module):
    """
    LayerNorm + time conditioning
    x: (B, T, D)
    t_emb: (B, time_dim)
    """
    def __init__(self, d_model, time_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, d_model * 2)
        )

    def forward(self, x, t_emb):
        x = self.norm(x)
        scale, shift = self.to_scale_shift(t_emb).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)   # (B,1,D)
        shift = shift.unsqueeze(1)   # (B,1,D)
        return x * (1 + scale) + shift


class TransformerResBlock(nn.Module):
    """
    Conv blockの代わりに使うTransformer block
    self-attn -> FFN with residual
    """
    def __init__(self, d_model, time_dim, nhead=8, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.adaln1 = AdaLN(d_model, time_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.adaln2 = AdaLN(d_model, time_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, t_emb, padding_mask=None, attn_mask=None):
        # x: (B,T,D)
        res = x
        h = self.adaln1(x, t_emb)
        h, _ = self.self_attn(
            h, h, h,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
            need_weights=False
        )
        x = res + self.dropout1(h)

        res = x
        h = self.adaln2(x, t_emb)
        h = self.ffn(h)
        x = res + self.dropout2(h)

        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).float()
            x = x * valid
        return x


class CrossAttentionBlock(nn.Module):
    """
    x: (B,T,D)
    text_tokens: (B,L,H)
    """
    def __init__(self, d_model, text_dim, nhead=8, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, text_tokens, x_padding_mask=None, text_padding_mask=None):
        res = x
        q = self.norm_q(x)
        kv = self.text_proj(text_tokens)

        h, _ = self.cross_attn(
            q, kv, kv,
            key_padding_mask=text_padding_mask,
            need_weights=False
        )
        x = res + self.dropout1(h)

        res = x
        h = self.norm2(x)
        h = self.ffn(h)
        x = res + self.dropout2(h)

        if x_padding_mask is not None:
            valid = (~x_padding_mask).unsqueeze(-1).float()
            x = x * valid
        return x


class DownsampleTransformer(nn.Module):
    """
    時系列長を半分に圧縮
    feature次元は維持
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, padding_mask=None):
        # x: (B,T,D)
        B, T, D = x.shape

        if T % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1), mode="constant", value=0.0)
            if padding_mask is not None:
                padding_mask = F.pad(padding_mask, (0, 1), value=True)
            T = T + 1

        x = x.view(B, T // 2, 2, D).mean(dim=2)  # (B,T/2,D)

        new_mask = None
        if padding_mask is not None:
            new_mask = padding_mask.view(B, T // 2, 2).all(dim=-1)
            valid = (~new_mask).unsqueeze(-1).float()
            x = x * valid

        return x, new_mask


class UpsampleTransformer(nn.Module):
    """
    時系列長を2倍に復元
    """
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model * 2)

    def forward(self, x, target_len=None, padding_mask=None):
        # x: (B,T,D)
        B, T, D = x.shape
        x = self.proj(x).view(B, T, 2, D).reshape(B, T * 2, D)

        if target_len is not None and x.size(1) != target_len:
            x = x[:, :target_len]

        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).float()
            x = x * valid

        return x


class TransformerUNetDenoiser(nn.Module):
    """
    Transformer版U-Net denoiser
    Input : (B,T,z_dim)
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
        num_levels=3,
        depth_per_level=2,
        use_text_at_levels=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.num_levels = num_levels
        self.depth_per_level = depth_per_level

        if use_text_at_levels is None:
            use_text_at_levels = [True] * num_levels

        self.in_proj = nn.Linear(latent_dim, model_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # positional embeddingは毎forwardで動的生成
        self.down_blocks = nn.ModuleList()
        self.down_cross = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(num_levels):
            blocks = nn.ModuleList([
                TransformerResBlock(
                    d_model=model_dim,
                    time_dim=time_dim,
                    nhead=nhead,
                    dropout=dropout
                )
                for _ in range(depth_per_level)
            ])
            self.down_blocks.append(blocks)

            if use_text_at_levels[i]:
                self.down_cross.append(
                    CrossAttentionBlock(
                        d_model=model_dim,
                        text_dim=text_cond_dim,
                        nhead=nhead,
                        dropout=dropout
                    )
                )
            else:
                self.down_cross.append(nn.Identity())

            if i < num_levels - 1:
                self.downsamples.append(DownsampleTransformer())

        self.mid_blocks = nn.ModuleList([
            TransformerResBlock(
                d_model=model_dim,
                time_dim=time_dim,
                nhead=nhead,
                dropout=dropout
            ),
            TransformerResBlock(
                d_model=model_dim,
                time_dim=time_dim,
                nhead=nhead,
                dropout=dropout
            )
        ])
        self.mid_cross = CrossAttentionBlock(
            d_model=model_dim,
            text_dim=text_cond_dim,
            nhead=nhead,
            dropout=dropout
        )

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.up_cross = nn.ModuleList()
        self.skip_merge = nn.ModuleList()

        for i in range(num_levels - 1):
            self.upsamples.append(UpsampleTransformer(model_dim))
            self.skip_merge.append(nn.Linear(model_dim * 2, model_dim))

            blocks = nn.ModuleList([
                TransformerResBlock(
                    d_model=model_dim,
                    time_dim=time_dim,
                    nhead=nhead,
                    dropout=dropout
                )
                for _ in range(depth_per_level)
            ])
            self.up_blocks.append(blocks)

            self.up_cross.append(
                CrossAttentionBlock(
                    d_model=model_dim,
                    text_dim=text_cond_dim,
                    nhead=nhead,
                    dropout=dropout
                )
            )

        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, latent_dim)

    def position_embedding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1,T,D)

    def forward(
        self,
        x_t,                      # (B,T,z_dim)
        t,                        # (B,)
        text_tokens,              # (B,L,H)
        x_padding_mask=None,      # (B,T) True=pad
        text_padding_mask=None,   # (B,L) True=pad
        src_mask=None,            # optional self-attn mask
    ):
        x = self.in_proj(x_t)  # (B,T,D)
        x = x + self.position_embedding(x.size(1), x.size(2), x.device)
        t_emb = self.time_mlp(t)

        if x_padding_mask is not None:
            valid = (~x_padding_mask).unsqueeze(-1).float()
            x = x * valid

        skips = []
        masks = []

        cur_x = x
        cur_mask = x_padding_mask

        # -------------------------
        # Down path
        # -------------------------
        for i in range(self.num_levels):
            for blk in self.down_blocks[i]:
                cur_x = blk(cur_x, t_emb, padding_mask=cur_mask, attn_mask=src_mask)

            if not isinstance(self.down_cross[i], nn.Identity):
                cur_x = self.down_cross[i](
                    cur_x, text_tokens,
                    x_padding_mask=cur_mask,
                    text_padding_mask=text_padding_mask
                )

            skips.append(cur_x)
            masks.append(cur_mask)

            if i < self.num_levels - 1:
                cur_x, cur_mask = self.downsamples[i](cur_x, cur_mask)
                cur_x = cur_x + self.position_embedding(cur_x.size(1), cur_x.size(2), cur_x.device)

        # -------------------------
        # Bottleneck
        # -------------------------
        cur_x = self.mid_blocks[0](cur_x, t_emb, padding_mask=cur_mask, attn_mask=src_mask)
        cur_x = self.mid_cross(
            cur_x, text_tokens,
            x_padding_mask=cur_mask,
            text_padding_mask=text_padding_mask
        )
        cur_x = self.mid_blocks[1](cur_x, t_emb, padding_mask=cur_mask, attn_mask=src_mask)

        # -------------------------
        # Up path
        # -------------------------
        for i in range(self.num_levels - 1):
            skip = skips[-(i + 2)]
            skip_mask = masks[-(i + 2)]

            cur_x = self.upsamples[i](cur_x, target_len=skip.size(1), padding_mask=skip_mask)
            cur_x = cur_x + self.position_embedding(cur_x.size(1), cur_x.size(2), cur_x.device)

            cur_x = torch.cat([cur_x, skip], dim=-1)
            cur_x = self.skip_merge[i](cur_x)

            for blk in self.up_blocks[i]:
                cur_x = blk(cur_x, t_emb, padding_mask=skip_mask, attn_mask=src_mask)

            cur_x = self.up_cross[i](
                cur_x, text_tokens,
                x_padding_mask=skip_mask,
                text_padding_mask=text_padding_mask
            )
            cur_mask = skip_mask

        cur_x = self.out_norm(cur_x)
        if cur_mask is not None:
            valid = (~cur_mask).unsqueeze(-1).float()
            cur_x = cur_x * valid

        return self.out_proj(cur_x)