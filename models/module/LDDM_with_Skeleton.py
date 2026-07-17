import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.CLIP_Skeleton import TextEncoder
from models.module.AdaLN import AdaLN
from models.module.STAttentionDenoiser import AdaLNCrossTextSTTransformerBlock
from models.module.UnetDenoiser import AdaLNCrossTextUNetDenoiser


# =========================================================
# Utils
# =========================================================

def exists(x):
    return x is not None


def default(x, d):
    return x if exists(x) else d


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


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
# 2D VAE
# video: (B, C, T, H, W)
# latent: (B, T, Cz, Hz, Wz)
# =========================================================

class ConvEncoder2D(nn.Module):
    """
    Input:  (B*T, C, 224, 224)
    Output: mu/logvar -> (B*T, z_ch, 28, 28)

    Downsampling:
        224 -> 112 -> 56 -> 28
    """
    def __init__(self, in_ch=3, base_ch=64, z_ch=4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1),  # 224 -> 112
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1),  # 112 -> 56
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=4, stride=2, padding=1),  # 56 -> 28
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
        )

        self.mid = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
        )

        self.to_mu = nn.Conv2d(base_ch * 4, z_ch, kernel_size=1)
        self.to_logvar = nn.Conv2d(base_ch * 4, z_ch, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.mid(x)

        mu = self.to_mu(x)
        logvar = self.to_logvar(x)
        return mu, logvar


class ConvDecoder2D(nn.Module):
    """
    Input:  (B*T, z_ch, 28, 28)
    Output: (B*T, C, 224, 224)

    Upsampling:
        28 -> 56 -> 112 -> 224
    """
    def __init__(self, out_ch=3, base_ch=64, z_ch=4):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(z_ch, base_ch * 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
        )

        self.mid = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 4, kernel_size=4, stride=2, padding=1),  # 28 -> 56
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1),  # 56 -> 112
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(),
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1),  # 112 -> 224
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
        )

        self.out = nn.Conv2d(base_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = self.in_proj(z)
        x = self.mid(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out(x)
        return x


class VideoVAE2D(nn.Module):
    """
    video: (B, C, T, H, W)
    frame-wise 2D VAE

    For H=W=224:
        latent spatial size = 28 x 28
    """
    def __init__(self, in_ch=3, base_ch=64, z_ch=4):
        super().__init__()
        self.encoder = ConvEncoder2D(in_ch=in_ch, base_ch=base_ch, z_ch=z_ch)
        self.decoder = ConvDecoder2D(out_ch=in_ch, base_ch=base_ch, z_ch=z_ch)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, video: torch.Tensor):
        """
        video: (B, C, T, H, W)
        returns:
            z:      (B, T, z_ch, H/8, W/8)
            mu:     (B, T, z_ch, H/8, W/8)
            logvar: (B, T, z_ch, H/8, W/8)
        """
        B, C, T, H, W = video.shape
        x = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        _, Cz, Hz, Wz = z.shape
        z = z.view(B, T, Cz, Hz, Wz)
        mu = mu.view(B, T, Cz, Hz, Wz)
        logvar = logvar.view(B, T, Cz, Hz, Wz)
        return z, mu, logvar

    def decode(self, z: torch.Tensor):
        """
        z: (B, T, z_ch, Hz, Wz)
        returns:
            recon_video: (B, C, T, H, W)
        """
        B, T, Cz, Hz, Wz = z.shape
        z_ = z.reshape(B * T, Cz, Hz, Wz)
        x_rec = self.decoder(z_)

        _, C, H, W = x_rec.shape
        x_rec = x_rec.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return x_rec

    def forward(self, video: torch.Tensor):
        z, mu, logvar = self.encode(video)
        recon = self.decode(z)
        return recon, z, mu, logvar




# =========================================================
# Diffusion Schedule
# =========================================================

class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()

        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        alpha_bars_prev = torch.cat([
            torch.ones(1, dtype=torch.float32),
            alpha_bars[:-1]
        ], dim=0)

        # DDPM posterior variance
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))

        posterior_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_mean_coef2 = (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars)

        self.num_steps = num_steps

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)

        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ):
        """
        x0: (B, T, Cz, Hz, Wz)
        t:  (B,)
        """
        noise = default(noise, torch.randn_like(x0))
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1, 1)
        xt = sqrt_ab * x0 + sqrt_omb * noise
        return xt, noise


# =========================================================
# Denoiser
# AdaLN(time + text embeddings) + CrossAttn(text last_hidden_state)
# =========================================================

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


class AdaLNCrossTextTransformerBlock(nn.Module):
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
        self.adaln3 = AdaLN(d_model=d_model, time_dim=cond_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
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
        x: torch.Tensor,                                # (B, N, D)
        cond_emb: torch.Tensor,                         # (B, cond_dim)
        text_tokens: torch.Tensor,                      # (B, L, D)
        text_key_padding_mask: Optional[torch.Tensor],  # (B, L), True=padding
        T=None,
        S=None,
    ):
        # self-attention
        h = self.norm1(x)
        scale1, shift1, gate1 = self.adaln1(h, cond_emb)
        h = self.modulate(h, scale1, shift1)
        h_sa, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + gate1 * h_sa

        # cross-attention
        h = self.norm2(x)
        h_ca, _ = self.cross_attn(
            query=h,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )
        x = x + h_ca

        # ffn
        h = self.norm3(x)
        scale3, shift3, gate3 = self.adaln3(h, cond_emb)
        h = self.modulate(h, scale3, shift3)
        h_ff = self.ff(h)
        x = x + gate3 * h_ff

        return x


class AdaLNCrossTextDenoiser(nn.Module):
    """
    xt:         (B, T, Cz, Hz, Wz)
    t:          (B,)
    text_emb:   (B, text_emb_dim)          <- TextEncoder()["embeddings"]
    text_h:     (B, L, text_hidden_dim)    <- TextEncoder()["last_hidden_state"]
    """
    def __init__(
        self,
        latent_ch: int,
        text_emb_dim: int,
        text_hidden_dim: int,
        model_dim: int = 512,
        depth: int = 8,
        nhead: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        time_embed_dim: int = 512,
        cond_dim: int = 512,
        max_time: int = 256,
        max_hw_tokens: int = 1024,
    ):
        super().__init__()
        self.latent_ch = latent_ch
        self.model_dim = model_dim
        self.max_time = max_time
        self.max_hw_tokens = max_hw_tokens
        self.time_embed_dim = time_embed_dim

        self.latent_in = nn.Linear(latent_ch, model_dim)
        self.text_token_proj = nn.Linear(text_hidden_dim, model_dim)

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

        self.temporal_pos = nn.Parameter(torch.randn(1, max_time, 1, model_dim) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, max_hw_tokens, model_dim) * 0.02)

        self.blocks = nn.ModuleList([
            AdaLNCrossTextSTTransformerBlock(
                d_model=model_dim,
                cond_dim=cond_dim,
                nhead=nhead,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, latent_ch)

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
        attention_mask: Optional[torch.Tensor] = None,  # (B, L), 1=valid
    ) -> torch.Tensor:
        B, T, Cz, Hz, Wz = xt.shape
        S = Hz * Wz

        if T > self.max_time:
            raise ValueError(f"T={T} exceeds max_time={self.max_time}")
        if S > self.max_hw_tokens:
            raise ValueError(f"S={S} exceeds max_hw_tokens={self.max_hw_tokens}")

        # latent -> token
        x = xt.permute(0, 1, 3, 4, 2).reshape(B, T, S, Cz)
        x = self.latent_in(x)
        x = x + self.temporal_pos[:, :T] + self.spatial_pos[:, :, :S]
        x = x.reshape(B, T * S, self.model_dim)

        # condition
        cond = self.make_global_condition(t, text_emb)
        text_tokens = self.text_token_proj(text_h)

        text_key_padding_mask = None
        if attention_mask is not None:
            text_key_padding_mask = (attention_mask == 0)

        for blk in self.blocks:
            x = blk(
                x=x,
                cond_emb=cond,
                text_tokens=text_tokens,
                text_key_padding_mask=text_key_padding_mask,
                T=T,
                S=S,
            )

        x = self.out_proj(self.out_norm(x))
        x = x.view(B, T, S, Cz).reshape(B, T, Hz, Wz, Cz).permute(0, 1, 4, 2, 3)
        return x


# =========================================================
# Outputs
# =========================================================

@dataclass
class VAEPretrainOutput:
    loss: torch.Tensor
    recon_loss: torch.Tensor
    kl_loss: torch.Tensor
    recon_video: torch.Tensor
    latent: torch.Tensor


@dataclass
class LDDMTrainOutput:
    loss: torch.Tensor
    diff_loss: torch.Tensor
    pred: torch.Tensor
    target: torch.Tensor
    pred_eps: torch.Tensor
    pred_x0: torch.Tensor
    noisy_latent: torch.Tensor
    clean_latent: torch.Tensor


# =========================================================
# Main model
# =========================================================

class VideoTextLDDM(nn.Module):
    """
    mode:
      - "vae"  : VAE pretraining
      - "lddm" : latent diffusion training
    """

    def __init__(
        self,
        vae: VideoVAE2D,
        text_encoder: TextEncoder,
        denoiser: AdaLNCrossTextDenoiser,
        schedule: DiffusionSchedule,
        vae_scale: float = 0.18215,
        default_mode: str = "lddm",
        freeze_text_encoder_in_lddm: bool = True,
        kl_weight: float = 1e-4,
        prediction_type: str = "eps",   # "eps", "x", "x0", "v"
    ):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.denoiser = denoiser
        self.schedule = schedule
        self.vae_scale = vae_scale
        self.freeze_text_encoder_in_lddm = freeze_text_encoder_in_lddm
        self.kl_weight = kl_weight

        self.prediction_type = prediction_type.lower()
        if self.prediction_type not in ["eps", "x", "x0", "v"]:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")

        self.mode = None
        self.set_mode(default_mode)

    # -----------------------------------------------------
    # mode control
    # -----------------------------------------------------
    def set_mode(self, mode: str):
        if mode not in ["vae", "lddm"]:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode

        if mode == "vae":
            set_requires_grad(self.vae, True)
            set_requires_grad(self.denoiser, False)
            set_requires_grad(self.text_encoder, False)

            self.vae.train()
            self.denoiser.eval()
            self.text_encoder.eval()

        elif mode == "lddm":
            set_requires_grad(self.vae, False)
            set_requires_grad(self.denoiser, True)
            set_requires_grad(self.text_encoder, not self.freeze_text_encoder_in_lddm)

            self.vae.eval()
            self.denoiser.train()
            if self.freeze_text_encoder_in_lddm:
                self.text_encoder.eval()
            else:
                self.text_encoder.train()

    # -----------------------------------------------------
    # helpers
    # -----------------------------------------------------
    @staticmethod
    def vae_kl_loss(mu, logvar):
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if any(p.requires_grad for p in self.text_encoder.parameters()):
            text_out = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            with torch.no_grad():
                text_out = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

        return {
            "text_emb": text_out["embeddings"],          # (B, proj_dim)
            "text_h": text_out["last_hidden_state"],     # (B, L, hidden_size)
        }

    def encode_video_to_latent(self, video: torch.Tensor):
        if any(p.requires_grad for p in self.vae.parameters()):
            z, mu, logvar = self.vae.encode(video)
        else:
            with torch.no_grad():
                z, mu, logvar = self.vae.encode(video)

        z = z * self.vae_scale
        return z, mu, logvar

    def decode_latent_to_video(self, z: torch.Tensor):
        z = z / self.vae_scale
        return self.vae.decode(z)
    def set_prediction_type(self, prediction_type: str):
        prediction_type = prediction_type.lower()
        if prediction_type not in ["eps", "x", "x0", "v"]:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")
        self.prediction_type = prediction_type

    def _normalize_prediction_type(self) -> str:
        if self.prediction_type == "x":
            return "x0"
        return self.prediction_type

    def predict_v_from_x0_eps(
        self,
        x0: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_ab = self.schedule.sqrt_alpha_bars[t].view(-1, 1, 1, 1, 1)
        sqrt_omb = self.schedule.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1, 1)
        v = sqrt_ab * eps - sqrt_omb * x0
        return v

    def predict_x0_from_eps(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_ab = self.schedule.sqrt_alpha_bars[t].view(-1, 1, 1, 1, 1)
        sqrt_omb = self.schedule.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1, 1)
        x0 = (zt - sqrt_omb * eps) / (sqrt_ab + 1e-8)
        return x0

    def predict_eps_from_x0(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_ab = self.schedule.sqrt_alpha_bars[t].view(-1, 1, 1, 1, 1)
        sqrt_omb = self.schedule.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1, 1)
        eps = (zt - sqrt_ab * x0) / (sqrt_omb + 1e-8)
        return eps

    def predict_x0_from_v(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_ab = self.schedule.sqrt_alpha_bars[t].view(-1, 1, 1, 1, 1)
        sqrt_omb = self.schedule.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1, 1)
        x0 = sqrt_ab * zt - sqrt_omb * v
        return x0

    def predict_eps_from_v(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_ab = self.schedule.sqrt_alpha_bars[t].view(-1, 1, 1, 1, 1)
        sqrt_omb = self.schedule.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1, 1)
        eps = sqrt_omb * zt + sqrt_ab * v
        return eps

    def get_training_target(
        self,
        z0: torch.Tensor,
        zt: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        pred_type = self._normalize_prediction_type()

        if pred_type == "eps":
            return noise
        elif pred_type == "x0":
            return z0
        elif pred_type == "v":
            return self.predict_v_from_x0_eps(z0, noise, t)
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

    def model_output_to_eps_x0(
        self,
        model_pred: torch.Tensor,
        zt: torch.Tensor,
        t: torch.Tensor,
        clip_x0: bool = False,
    ):
        pred_type = self._normalize_prediction_type()

        if pred_type == "eps":
            pred_eps = model_pred
            pred_x0 = self.predict_x0_from_eps(zt, t, pred_eps)

        elif pred_type == "x0":
            pred_x0 = model_pred
            if clip_x0:
                pred_x0 = pred_x0.clamp(-1.0, 1.0)
            pred_eps = self.predict_eps_from_x0(zt, t, pred_x0)

        elif pred_type == "v":
            pred_x0 = self.predict_x0_from_v(zt, t, model_pred)
            if clip_x0:
                pred_x0 = pred_x0.clamp(-1.0, 1.0)
            pred_eps = self.predict_eps_from_v(zt, t, model_pred)

        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

        if clip_x0 and pred_type == "eps":
            pred_x0 = pred_x0.clamp(-1.0, 1.0)
            pred_eps = self.predict_eps_from_x0(zt, t, pred_x0)

        return pred_eps, pred_x0
    # -----------------------------------------------------
    # VAE pretraining
    # -----------------------------------------------------
    def forward_vae(self, video: torch.Tensor) -> VAEPretrainOutput:
        z, mu, logvar = self.vae.encode(video)
        recon_video = self.vae.decode(z)

        recon_loss = F.mse_loss(recon_video, video)
        kl_loss = self.vae_kl_loss(mu, logvar)
        loss = recon_loss + self.kl_weight * kl_loss

        return VAEPretrainOutput(
            loss=loss,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            recon_video=recon_video,
            latent=z,
        )

    # -----------------------------------------------------
    # LDDM training
    # -----------------------------------------------------
    def forward_lddm(
            self,
            video: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
    ) -> LDDMTrainOutput:
        B = video.shape[0]
        device = video.device

        text_out = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        text_emb = text_out["text_emb"]
        text_h = text_out["text_h"]

        z0, _, _ = self.encode_video_to_latent(video)

        t = torch.randint(
            low=0,
            high=self.schedule.num_steps,
            size=(B,),
            device=device,
            dtype=torch.long,
        )

        zt, noise = self.schedule.q_sample(z0, t)

        model_pred = self.denoiser(
            xt=zt,
            t=t,
            text_emb=text_emb,
            text_h=text_h,
            attention_mask=attention_mask,
        )

        target = self.get_training_target(
            z0=z0,
            zt=zt,
            t=t,
            noise=noise,
        )

        diff_loss = F.mse_loss(model_pred, target)
        loss = diff_loss

        pred_eps, pred_x0 = self.model_output_to_eps_x0(
            model_pred=model_pred,
            zt=zt,
            t=t,
            clip_x0=False,
        )

        return LDDMTrainOutput(
            loss=loss,
            diff_loss=diff_loss,
            pred=model_pred,
            target=target,
            pred_eps=pred_eps,
            pred_x0=pred_x0,
            noisy_latent=zt,
            clean_latent=z0,
        )
    # -----------------------------------------------------
    # unified forward
    # -----------------------------------------------------
    def forward(
        self,
        video: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ):
        run_mode = self.mode if mode is None else mode

        if run_mode == "vae":
            return self.forward_vae(video)

        elif run_mode == "lddm":
            if input_ids is None or attention_mask is None:
                raise ValueError("input_ids and attention_mask are required in lddm mode.")
            return self.forward_lddm(
                video=video,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        else:
            raise ValueError(f"Unsupported mode: {run_mode}")

    # =====================================================
    # inference helpers
    # =====================================================

    @torch.no_grad()
    def predict_model_output(
            self,
            zt: torch.Tensor,
            t: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
    ):
        text_out = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        model_pred = self.denoiser(
            xt=zt,
            t=t,
            text_emb=text_out["text_emb"],
            text_h=text_out["text_h"],
            attention_mask=attention_mask,
        )
        return model_pred

    @torch.no_grad()
    def predict_x0_from_eps(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        pred_noise: torch.Tensor,
    ):
        sqrt_ab = self.schedule.sqrt_alpha_bars[t].view(-1, 1, 1, 1, 1)
        sqrt_omb = self.schedule.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1, 1)
        x0 = (zt - sqrt_omb * pred_noise) / (sqrt_ab + 1e-8)
        return x0

    # =====================================================
    # DDPM sampling
    # =====================================================

    @torch.no_grad()
    def p_sample_ddpm(
            self,
            zt: torch.Tensor,
            t_scalar: int,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            clip_x0: bool = False,
    ):
        B = zt.shape[0]
        device = zt.device
        t = torch.full((B,), t_scalar, device=device, dtype=torch.long)

        model_pred = self.predict_model_output(
            zt=zt,
            t=t,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pred_eps, pred_x0 = self.model_output_to_eps_x0(
            model_pred=model_pred,
            zt=zt,
            t=t,
            clip_x0=clip_x0,
        )

        coef1 = self.schedule.posterior_mean_coef1[t].view(-1, 1, 1, 1, 1)
        coef2 = self.schedule.posterior_mean_coef2[t].view(-1, 1, 1, 1, 1)
        posterior_mean = coef1 * pred_x0 + coef2 * zt

        posterior_variance = self.schedule.posterior_variance[t].view(-1, 1, 1, 1, 1)
        posterior_log_variance = self.schedule.posterior_log_variance_clipped[t].view(-1, 1, 1, 1, 1)

        if t_scalar > 0:
            noise = torch.randn_like(zt)
            z_prev = posterior_mean + torch.exp(0.5 * posterior_log_variance) * noise
        else:
            z_prev = posterior_mean

        return z_prev, pred_x0, pred_eps

    @torch.no_grad()
    def sample_ddpm(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            shape: Optional[Tuple[int, int, int, int, int]] = None,  # (B,T,Cz,Hz,Wz)
            z_T: Optional[torch.Tensor] = None,
            return_latent: bool = False,
            clip_x0: bool = False,
    ):
        if z_T is None:
            if shape is None:
                raise ValueError("Either shape or z_T must be provided.")
            z = torch.randn(shape, device=input_ids.device)
        else:
            z = z_T

        for t in reversed(range(self.schedule.num_steps)):
            z, _, _ = self.p_sample_ddpm(
                zt=z,
                t_scalar=t,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                clip_x0=clip_x0,
            )

        if return_latent:
            return z

        return self.decode_latent_to_video(z)

    # =====================================================
    # DDIM sampling
    # =====================================================

    @torch.no_grad()
    def p_sample_ddim(
            self,
            zt: torch.Tensor,
            t_scalar: int,
            t_prev_scalar: int,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            eta: float = 0.0,
            clip_x0: bool = False,
    ):
        B = zt.shape[0]
        device = zt.device
        t = torch.full((B,), t_scalar, device=device, dtype=torch.long)

        model_pred = self.predict_model_output(
            zt=zt,
            t=t,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pred_eps, pred_x0 = self.model_output_to_eps_x0(
            model_pred=model_pred,
            zt=zt,
            t=t,
            clip_x0=clip_x0,
        )

        alpha_bar_t = self.schedule.alpha_bars[t].view(-1, 1, 1, 1, 1)

        if t_prev_scalar >= 0:
            t_prev = torch.full((B,), t_prev_scalar, device=device, dtype=torch.long)
            alpha_bar_prev = self.schedule.alpha_bars[t_prev].view(-1, 1, 1, 1, 1)
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)

        sigma = eta * torch.sqrt(
            ((1 - alpha_bar_prev) / (1 - alpha_bar_t + 1e-8)) *
            (1 - alpha_bar_t / (alpha_bar_prev + 1e-8))
        )

        dir_coeff = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma ** 2, min=0.0))
        noise = torch.randn_like(zt) if eta > 0 else torch.zeros_like(zt)

        z_prev = (
                torch.sqrt(alpha_bar_prev) * pred_x0 +
                dir_coeff * pred_eps +
                sigma * noise
        )

        return z_prev, pred_x0, pred_eps

    @torch.no_grad()
    def sample_ddim(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            shape: Optional[Tuple[int, int, int, int, int]] = None,
            z_T: Optional[torch.Tensor] = None,
            ddim_steps: int = 50,
            eta: float = 0.0,
            return_latent: bool = False,
            clip_x0: bool = False,
    ):
        if z_T is None:
            if shape is None:
                raise ValueError("Either shape or z_T must be provided.")
            z = torch.randn(shape, device=input_ids.device)
        else:
            z = z_T

        total_steps = self.schedule.num_steps
        step_indices = torch.linspace(
            total_steps - 1, 0, ddim_steps, device=input_ids.device
        ).long().tolist()

        for i, t in enumerate(step_indices):
            t_prev = step_indices[i + 1] if i + 1 < len(step_indices) else -1
            z, _, _ = self.p_sample_ddim(
                zt=z,
                t_scalar=t,
                t_prev_scalar=t_prev,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                eta=eta,
                clip_x0=clip_x0,
            )

        if return_latent:
            return z

        return self.decode_latent_to_video(z)
if __name__ == "__main__":
    import torch
    from transformers import BertTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================================
    # dummy config
    # =====================================
    B = 2
    C = 3
    T = 90
    H = 224
    W = 224

    # =====================================
    # dummy video
    # =====================================
    video = torch.randn(B, C, T, H, W).to(device)

    # =====================================
    # dummy text
    # =====================================
    texts = [
        "a person waves hand",
        "someone raises both hands",
    ]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    # =====================================
    # models
    # =====================================
    vae = VideoVAE2D(
        in_ch=3,
        base_ch=32,
        z_ch=4,
    ).to(device)

    text_encoder = TextEncoder(
        model_name="bert-base-uncased",
        proj_dim=256,
        dropout=0.1,
        pooling="cls",
    ).to(device)

    denoiser = AdaLNCrossTextUNetDenoiser(
        latent_ch=4,
        text_emb_dim=256,
        text_hidden_dim=text_encoder.embed_model.config.hidden_size,
        base_ch= 64,
        channel_mults = [1, 2, 4],
        num_res_blocks = 2,
        nhead = 8,
        dropout = 0.1,
        time_embed_dim = 512,
        cond_dim = 512,
        num_groups = 8,
        use_cross_attn_in_all_levels = True,
    ).to(device)

    schedule = DiffusionSchedule(num_steps=100)

    model = VideoTextLDDM(
        vae=vae,
        text_encoder=text_encoder,
        denoiser=denoiser,
        schedule=schedule,
        prediction_type="eps",  # ← "eps", "x0", "v" に変更可能
    ).to(device)

    # =====================================
    # 1. VAE pretraining mode
    # =====================================
    scaler= torch.cuda.amp.GradScaler(enabled=True)
    with torch.amp.autocast('cuda',dtype=torch.bfloat16):

        print("=== VAE MODE ===")
        model.set_mode("vae")
        """
        out_vae = model(video=video)
        print("VAE loss:", out_vae.loss.item())
        print("latent shape:", out_vae.latent.shape)
        ### backward check
        scaler.scale(out_vae.loss).backward()
        print("Backward pass successful.")
        """

        # =====================================
        # 2. LDDM training
        # =====================================
        print("\n=== LDDM MODE ===")
        model.set_mode("lddm")
        model.set_prediction_type("v")  # ← 切替確認

        out_lddm = model(
            video=video,
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )

        print("diff loss:", out_lddm.loss.item())
        print("pred shape:", out_lddm.pred.shape)
        print("target shape:", out_lddm.target.shape)
        ### backward check with scaler
        scaler.scale(out_lddm.loss).backward()
        print("Backward pass successful with scaler.")
        exit()
        # =====================================
        # backward check
        # =====================================
        out_lddm.loss.backward()
        print("Backward pass successful.")
        # =====================================
        # 3. DDPM sampling
        # =====================================
        print("\n=== DDPM SAMPLE ===")

        sample_ddpm = model.sample_ddpm(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            shape=(B, T, 4, 28, 28),  # latent shape
        )

        print("DDPM output shape:", sample_ddpm.shape)

        # =====================================
        # 4. DDIM sampling
        # =====================================
        print("\n=== DDIM SAMPLE ===")

        sample_ddim = model.sample_ddim(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            shape=(B, T, 4, 28, 28),
            ddim_steps=20,
            eta=0.0,
        )

        print("DDIM output shape:", sample_ddim.shape)


        print("\n=== DONE ===")