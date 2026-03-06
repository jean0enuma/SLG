# Transformer-based VQ-VAE for coarse sign pose: (B, T, 36) -> codes -> (B, T, 36)
# - Encoder/Decoder use TransformerEncoder blocks (non-causal; autoencoder)
# - Temporal downsample/upsample are learned Conv1d / ConvTranspose1d over time
# - Quantizer: VQ or RVQ (Residual VQ)
#
# Input expected:
#   x: (B, T, 36)  where 36 = body(18) + L(wrist(3),x(3),z(3)) + R(wrist(3),x(3),z(3))
# Optional:
#   hand_valid_mask: (B, T, 2) bool for (left_valid, right_valid) to mask hand losses
#
# Notes:
# - Transformer tends to be more expressive; to keep VQ stable, we:
#     * apply LayerNorm
#     * downsample AFTER a few Transformer layers (or before; both work—this is stable)
#     * optionally use RVQ (recommended)
#
# Example:
#   model = VQVAETransformer1D(in_dim=36, d_model=256, code_dim=128, n_codes=1024, stride=4, rvq_stages=2)
#   out = model(x, hand_valid_mask=mask)
#   out["loss_total"].backward()

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

def downsample_mask_to_latent(mask_bt: torch.Tensor, K: int) -> torch.Tensor:
    """
    mask_bt: (B, T) bool or 0/1
    K: latent length (z_e.shape[-1])
    return: (B, K) float (0/1)
    """
    # (B,1,T) float
    m = mask_bt.float().unsqueeze(1)
    # nearest で 0/1 を保ったままリサイズ
    m_k = F.interpolate(m, size=K, mode="nearest")  # (B,1,K)
    return m_k.squeeze(1)  # (B,K)
def create_mask(target_length, max_len):
    # target_length: (batch_size,)
    batch_size = target_length.size(0)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32, device=target_length.device)
    for i in range(batch_size):
        mask[i, :target_length[i]] = 1.0
    return mask  # (batch_size, max_len)
# -------------------------
# Positional Encoding (sinusoidal)
# -------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, T: int, device=None, dtype=None):
        """
        returns: (1, T, dim)
        """
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        half = self.dim // 2
        t = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # (T,1)
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) *
            torch.arange(0, half, device=device, dtype=dtype) / half
        ).unsqueeze(0)  # (1,half)
        args = t * freqs  # (T,half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (T,2*half)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb.unsqueeze(0)  # (1,T,dim)


# -------------------------
# Vector Quantizers
# -------------------------
class VectorQuantizer(nn.Module):
    """
    Nearest-neighbor VQ with straight-through estimator.
    """
    def __init__(self, n_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta
        self.codebook = nn.Embedding(n_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / n_codes, 1.0 / n_codes)

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        z_e: (B, K, C)  (batch_first)
        returns:
          z_q: (B, K, C) straight-through
          codes: (B, K)
          loss_vq: scalar
          perplexity: scalar
        """
        B, K, C = z_e.shape
        assert C == self.code_dim

        z = z_e.reshape(-1, C)  # (B*K, C)
        e = self.codebook.weight  # (M, C)

        # dist = ||z||^2 + ||e||^2 - 2 z e^T
        z2 = (z ** 2).sum(dim=1, keepdim=True)          # (B*K,1)
        e2 = (e ** 2).sum(dim=1).unsqueeze(0)           # (1,M)
        ze = z @ e.t()                                  # (B*K,M)
        dist = z2 + e2 - 2 * ze                          # (B*K,M)

        idx = torch.argmin(dist, dim=1)                  # (B*K,)
        z_q = self.codebook(idx).view(B, K, C)           # (B,K,C)

        # straight-through
        z_q_st = z_e + (z_q - z_e).detach()

        # losses
        loss_codebook = F.mse_loss(z_q, z_e.detach())
        loss_commit = F.mse_loss(z_e, z_q.detach())
        loss_vq = loss_codebook + self.beta * loss_commit

        # perplexity
        onehot = F.one_hot(idx, num_classes=self.n_codes).float()
        avg = onehot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg * torch.log(avg + 1e-10)))

        return {
            "z_q": z_q_st,
            "codes": idx.view(B, K),
            "loss_vq": loss_vq,
            "perplexity": perplexity,
        }


class ResidualVectorQuantizer(nn.Module):
    """
    RVQ: z ≈ q1 + q2(residual) + ... + qS
    """
    def __init__(self, n_codes: int, code_dim: int, stages: int = 2, beta: float = 0.25):
        super().__init__()
        self.stages = stages
        self.vqs = nn.ModuleList([VectorQuantizer(n_codes, code_dim, beta=beta) for _ in range(stages)])

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        residual = z_e
        z_sum = torch.zeros_like(z_e)
        total_loss = 0.0
        codes_all = []
        perplex_all = []

        for vq in self.vqs:
            out = vq(residual)
            z_q = out["z_q"]
            z_sum = z_sum + z_q

            # residual update (keep gradient path on residual)
            residual = residual - (z_q - residual).detach()

            total_loss = total_loss + out["loss_vq"]
            codes_all.append(out["codes"])
            perplex_all.append(out["perplexity"])

        codes = torch.stack(codes_all, dim=1)  # (B, stages, K)
        perplexity = torch.stack(perplex_all).mean()

        return {
            "z_q": z_sum,
            "codes": codes,
            "loss_vq": total_loss,
            "perplexity": perplexity,
        }


# -------------------------
# Loss weights
# -------------------------
@dataclass
class VQLossWeights:
    recon_pos: float = 1.0
    recon_dir: float = 1.0
    vq: float = 1.0
    vel: float = 0.05  # small


# -------------------------
# Transformer VQ-VAE
# -------------------------
class VQVAETransformer1D(nn.Module):
    """
    x: (B, T, in_dim=36)
    encode:
      proj -> pos -> TransformerEnc -> downsample(conv stride) -> to_code_dim -> VQ
    decode:
      from_code_dim -> upsample(deconv stride) -> TransformerEnc -> out_proj
    """
    def __init__(
        self,
        in_dim: int = 36,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers_enc: int = 2,
        n_layers_dec: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
        code_dim: int = 128,
        n_codes: int = 1024,
        stride: int = 4,
        rvq_stages: int = 2,
        vq_beta: float = 0.25,
        loss_w: VQLossWeights = VQLossWeights(),
        # coarse split indices (same meaning as in previous code)
        pos_idx: Tuple[slice, ...] = (slice(0, 18), slice(18, 21), slice(29, 32)),  # body + Lwrist + Rwrist
        dir_idx: Tuple[slice, ...] = (slice(21, 27), slice(32, 38)),               # L(x,z) + R(x,z)
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.in_dim = in_dim
        self.d_model = d_model
        self.code_dim = code_dim
        self.n_codes = n_codes
        self.stride = stride
        self.loss_w = loss_w
        self.pos_idx = pos_idx
        self.dir_idx = dir_idx

        self.pos_emb = SinusoidalPosEmb(d_model)

        # ---- Encoder ----
        self.in_proj = nn.Linear(in_dim, d_model)
        self.enc_ln_in = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers_enc)

        # downsample in time (learned)
        # (B,T,d_model) -> (B,d_model,T) -> Conv1d stride -> (B,d_model,K)
        if stride > 1:
            self.down = nn.Conv1d(d_model, d_model, kernel_size=2 * stride, stride=stride, padding=stride // 2)
        else:
            self.down = nn.Identity()
        self.to_code = nn.Linear(d_model, code_dim)

        # ---- Quantizer ----
        if rvq_stages <= 1:
            self.quant = VectorQuantizer(n_codes=n_codes, code_dim=code_dim, beta=vq_beta)
            self.is_rvq = False
        else:
            self.quant = ResidualVectorQuantizer(n_codes=n_codes, code_dim=code_dim, stages=rvq_stages, beta=vq_beta)
            self.is_rvq = True

        # ---- Decoder ----
        self.from_code = nn.Linear(code_dim, d_model)
        if stride > 1:
            self.up = nn.ConvTranspose1d(
                d_model, d_model, kernel_size=2 * stride, stride=stride, padding=stride // 2, output_padding=stride % 2
            )
        else:
            self.up = nn.Identity()

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=n_layers_dec)

        self.dec_ln_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, in_dim)

    @torch.no_grad()
    def code_usage_histogram_update(
            self,
            x: torch.Tensor,
            prev_hist: Optional[Dict[str, torch.Tensor]] = None,
            *,
            input_length: Optional[torch.Tensor] = None,
            hand_valid_mask: Optional[torch.Tensor] = None,
            per_stage: bool = True,
            normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Update (accumulate) code-usage histogram using the current batch.

        This function is designed for *batch-wise accumulation*:
          - Input: prev_hist (previously accumulated histogram dict) or None
          - Output: updated histogram dict (counts are accumulated)

        Typical usage:
            hist = None
            for x, lengths in loader:
                hist = model.code_usage_histogram_update(x, hist, input_length=lengths)
            # after loop, hist['probs'] / hist['perplexity_from_hist'] describe dataset-level usage

        Args:
          x: (B,T,in_dim)
          prev_hist: dict previously returned by this function (or None to initialize)
          input_length: optional (B,) valid frame lengths to ignore padding
          hand_valid_mask: optional (B,T,2) (not required for histogram, passed through forward)
          per_stage: if True and RVQ is used, also accumulate per-stage histograms
          normalize: if True, compute probs/perplexity for the *accumulated* histogram

        Returns:
          dict with at least:
            - counts: (n_codes,) accumulated counts
            - probs: (n_codes,) probabilities from accumulated counts (if normalize=True)
            - perplexity_from_hist: scalar perplexity from accumulated probs (if normalize=True)
          For RVQ (if per_stage=True):
            - counts_per_stage: (stages,n_codes) accumulated per-stage counts
            - probs_per_stage: (stages,n_codes) per-stage probabilities (if normalize=True)
            - perplexity_per_stage: (stages,) per-stage perplexities (if normalize=True)
        """
        out = self.forward(
            x,
            hand_valid_mask=hand_valid_mask,
            input_length=input_length,
            return_recon=False,
        )
        codes = out["codes"]  # (B,K) or (B,S,K)
        predicted_poses = out["predicted_poses"]  # (B,T,in_dim)

        # latent mask (B,K) to ignore padded frames if input_length is provided
        if input_length is not None:
            if input_length.dim() != 1:
                raise ValueError("input_length must be a 1D tensor of shape (B,)")
            B, T = x.shape[0], x.shape[1]
            time_mask = create_mask(input_length.to(x.device), max_len=T)  # (B,T) float
            K = codes.shape[-1]
            latent_mask = downsample_mask_to_latent(time_mask, K=K).bool()  # (B,K)
        else:
            latent_mask = None

        device = x.device

        # Initialize accumulators if needed
        if prev_hist is None:
            prev_hist = {}
            prev_hist["counts"] = torch.zeros(self.n_codes, dtype=torch.long, device=device)
            if codes.dim() == 3 and per_stage:
                S = codes.shape[1]
                prev_hist["counts_per_stage"] = torch.zeros(S, self.n_codes, dtype=torch.long, device=device)
        prev_hist['predicted_poses'] = predicted_poses  # store last batch's predicted poses (for analysis)

        def _batch_counts(code_bk: torch.Tensor, mask_bk: Optional[torch.Tensor]) -> torch.Tensor:
            # code_bk: (B,K) int
            if mask_bk is not None:
                flat = code_bk[mask_bk].reshape(-1)
            else:
                flat = code_bk.reshape(-1)
            flat = flat.to(torch.int64)
            return torch.bincount(flat, minlength=self.n_codes).to(device)

        # ---- accumulate ----
        if codes.dim() == 2:
            c_batch = _batch_counts(codes, latent_mask)
            prev_hist["counts"] = prev_hist["counts"] + c_batch

        elif codes.dim() == 3:
            B, S, K = codes.shape
            # aggregate across stages
            if latent_mask is not None:
                mask_bsk = latent_mask.unsqueeze(1).expand(B, S, K)
                flat_all = codes[mask_bsk].reshape(-1).to(torch.int64)
            else:
                flat_all = codes.reshape(-1).to(torch.int64)
            c_batch = torch.bincount(flat_all, minlength=self.n_codes).to(device)
            prev_hist["counts"] = prev_hist["counts"] + c_batch

            if per_stage:
                if "counts_per_stage" not in prev_hist:
                    prev_hist["counts_per_stage"] = torch.zeros(S, self.n_codes, dtype=torch.long, device=device)
                for s in range(S):
                    cs = _batch_counts(codes[:, s, :], latent_mask)
                    prev_hist["counts_per_stage"][s] = prev_hist["counts_per_stage"][s] + cs
        else:
            raise ValueError(f"Unexpected codes shape: {tuple(codes.shape)}")

        # ---- compute probs/perplexity from accumulated counts (optional) ----
        if normalize:
            counts = prev_hist["counts"]
            total = counts.sum().clamp_min(1)
            probs = counts.float() / total.float()
            perplex = torch.exp(-(probs * torch.log(probs.clamp_min(1e-10))).sum())
            prev_hist["probs"] = probs
            prev_hist["perplexity_from_hist"] = perplex

            if "counts_per_stage" in prev_hist:
                cps = prev_hist["counts_per_stage"]
                totals = cps.sum(dim=1, keepdim=True).clamp_min(1)
                pps = cps.float() / totals.float()
                pxs = torch.exp(-(pps * torch.log(pps.clamp_min(1e-10))).sum(dim=1))
                prev_hist["probs_per_stage"] = pps
                prev_hist["perplexity_per_stage"] = pxs
        else:
            prev_hist["probs"] = torch.zeros(self.n_codes, dtype=torch.float32, device=device)
            prev_hist["perplexity_from_hist"] = torch.tensor(float("nan"), device=device)

        return prev_hist

    # ---------- helpers ----------
    @staticmethod
    def _safe_crop_or_pad_time(x: torch.Tensor, T: int) -> torch.Tensor:
        # x: (B, T_hat, C)
        T_hat = x.shape[1]
        if T_hat > T:
            return x[:, :T, :]
        if T_hat < T:
            return F.pad(x, (0, 0, 0, T - T_hat))
        return x

    def _gather_slices(self, x: torch.Tensor, slices: Tuple[slice, ...]) -> torch.Tensor:
        parts = [x[..., s] for s in slices]
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    # ---------- encode/decode ----------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,in_dim)
        returns z_e: (B,K,code_dim)
        """
        B, T, _ = x.shape
        h = self.in_proj(x)  # (B,T,d_model)
        h = self.enc_ln_in(h)

        pe = self.pos_emb(T, device=h.device, dtype=h.dtype)  # (1,T,d_model)
        h = h + pe

        h = self.encoder(h)  # (B,T,d_model)

        # downsample
        h_t = h.transpose(1, 2).contiguous()   # (B,d_model,T)
        h_k = self.down(h_t).transpose(1, 2).contiguous()  # (B,K,d_model)
        z_e = self.to_code(h_k)                # (B,K,code_dim)
        return z_e

    def decode(self, z_q: torch.Tensor, T_out: int) -> torch.Tensor:
        """
        z_q: (B,K,code_dim)
        returns x_hat: (B,T_out,in_dim)
        """
        h = self.from_code(z_q)  # (B,K,d_model)

        # upsample
        h_t = h.transpose(1, 2).contiguous()     # (B,d_model,K)
        h_up = self.up(h_t).transpose(1, 2).contiguous()  # (B,T_hat,d_model)

        h_up = self._safe_crop_or_pad_time(h_up, T_out)

        # add positional embedding at target length
        pe = self.pos_emb(T_out, device=h_up.device, dtype=h_up.dtype)
        h_up = h_up + pe

        h_up = self.decoder(h_up)
        h_up = self.dec_ln_out(h_up)
        x_hat = self.out_proj(h_up)
        return x_hat

    # ---------- forward & losses ----------
    def forward(
        self,
        x: torch.Tensor,  # (B,T,36)
        *,
        hand_valid_mask: Optional[torch.Tensor] = None,  # (B,T,2) bool
        input_length: Optional[int] = None,  # if provided, will crop/pad input to this length
        return_recon: bool = True,
        no_return_loss: bool = False,  # if True, skip loss computation and return empty dict (for inference)
    ) -> Dict[str, torch.Tensor]:
        B, T, C = x.shape
        assert C == self.in_dim

        z_e = self.encode(x)               # (B,K,code_dim)
        if input_length is not None:
            mask=create_mask(input_length,max_len=T).unsqueeze(-1) #(B,T,1)
        else:
            mask=torch.ones(B,T,1,device=x.device)

        z_e = z_e
        q = self.quant(z_e)                # dict
        z_q = q["z_q"]                     # (B,K,code_dim)
        x_hat = self.decode(z_q, T_out=T)  # (B,T,36)
        if no_return_loss:
            return {"x_hat": x_hat, "codes": q["codes"],"perplexity": q["perplexity"].detach(), "loss_vq": q["loss_vq"]}

        # ---- losses: pos (L1/Huber), dir (cos), vel (small), vq ----
        x_pos = self._gather_slices(x, self.pos_idx)
        h_pos = self._gather_slices(x_hat, self.pos_idx)
        loss_recon_pos = F.smooth_l1_loss(h_pos, x_pos, reduction='none')

        loss_recon_pos=(loss_recon_pos*mask).sum()/mask.sum()
        x_dir = self._gather_slices(x, self.dir_idx)
        h_dir = self._gather_slices(x_hat, self.dir_idx)

        # direction loss as cosine per 3D vector
        # x_dir/h_dir dims are 12: [Lx(3),Lz(3),Rx(3),Rz(3)] => reshape (B,T,4,3)
        x_vec = x_dir.view(B, T, 4, 3)
        h_vec = h_dir.view(B, T, 4, 3)
        eps = 1e-8
        x_n = x_vec / (x_vec.norm(dim=-1, keepdim=True) + eps)
        h_n = h_vec / (h_vec.norm(dim=-1, keepdim=True) + eps)
        cos = (x_n * h_n).sum(dim=-1)  # (B,T,4)
        dir_loss_per = 1.0 - cos

        if hand_valid_mask is not None:
            # mask: (B,T,2) -> (B,T,4) for [Lx,Lz,Rx,Rz]
            mL = hand_valid_mask[..., 0].float()
            mR = hand_valid_mask[..., 1].float()
            m = torch.stack([mL, mL, mR, mR], dim=-1)  # (B,T,4)
            loss_recon_dir = (dir_loss_per * m).sum() / m.sum().clamp_min(1.0)
        else:
            loss_recon_dir = dir_loss_per.mean()

        vel_hat = x_hat[:, 1:] - x_hat[:, :-1]
        vel = x[:, 1:] - x[:, :-1]
        vel_mask = mask[:, 1:] * mask[:, :-1]  # only consider velocity where both frames are valid
        loss_vel = F.l1_loss(vel_hat, vel, reduction='none')
        loss_vel = (loss_vel * vel_mask).sum() / vel_mask.sum()
        loss_vq = q["loss_vq"]

        loss_total = (
            self.loss_w.recon_pos * loss_recon_pos
            + self.loss_w.recon_dir * loss_recon_dir
            + self.loss_w.vel * loss_vel
            + self.loss_w.vq * loss_vq
        )

        out = {
            "loss_total": loss_total,
            "loss_recon_pos": loss_recon_pos.detach(),
            "loss_recon_dir": loss_recon_dir.detach(),
            "loss_vel": loss_vel.detach(),
            "loss_vq": loss_vq.detach(),
            "perplexity": q["perplexity"].detach(),
            "codes": q["codes"],   # (B,K) for VQ, (B,stages,K) for RVQ
        }
        if return_recon:
            out["x_hat"] = x_hat
        return out
class VQVAETransformer1DSeparated(nn.Module):
    def __init__(
            self,
            pose_dim: int = 16,
            hand_dim: int = 8,
            extra_dim: int = 4,
            pose_d_model: int = 256,
            hand_d_model: int = 128,
            extra_d_model: int = 64,
            n_pose_heads: int = 8,
            n_hand_heads: int = 4,
            n_extra_heads: int = 4,
            n_pose_layers_enc: int = 2,
            n_hand_layers_enc: int = 2,
            n_extra_layers_enc: int = 2,
            n_pose_layers_dec: int = 2,
            n_hand_layers_dec: int = 2,
            n_extra_layers_dec: int = 2,
            ff_mult: int = 4,
            dropout: float = 0.1,
            pose_code_dim: int = 128,
            hand_code_dim: int = 64,
            extra_code_dim: int = 32,
            n_pose_codes: int =64,
            n_hand_codes: int = 128,
            n_extra_codes: int = 32,
            stride: int = 4,
            rvq_stages: int = 2,
            vq_beta: float = 0.25,
            loss_w: VQLossWeights = VQLossWeights(),
            pose_idx: Tuple[slice, ...] = (slice(0, 12), slice(12,14), slice(23, 25)),  # body + Lwrist + Rwrist
            dir_l_idx: Tuple[slice, ...] = (slice(15, 23),),               # L(x,z) + R(x,z)
            dir_r_idx: Tuple[slice, ...] = (slice(26, 34),)
    ):
        super().__init__()
        self.pose_idx = pose_idx
        self.dir_l_idx = dir_l_idx
        self.dir_r_idx = dir_r_idx
        self.extra_dim = extra_dim
        self.loss_w = loss_w
        self.pose_vqvae = VQVAETransformer1D(
            in_dim=pose_dim,
            d_model=pose_d_model,
            n_heads=n_pose_heads,
            n_layers_enc=n_pose_layers_enc,
            n_layers_dec=n_pose_layers_dec,
            ff_mult=ff_mult,
            dropout=dropout,
            code_dim=pose_code_dim,
            n_codes=n_pose_codes,
            stride=stride,
            rvq_stages=rvq_stages,
            vq_beta=vq_beta,
            loss_w=loss_w,
        )
        self.left_vqvae = VQVAETransformer1D(
            in_dim=hand_dim,
            d_model=hand_d_model,
            n_heads=n_hand_heads,
            n_layers_enc=n_hand_layers_enc,
            n_layers_dec=n_hand_layers_dec,
            ff_mult=ff_mult,
            dropout=dropout,
            code_dim=hand_code_dim,
            n_codes=n_hand_codes,
            stride=stride,
            rvq_stages=rvq_stages,
            vq_beta=vq_beta,
            loss_w=loss_w,
        )
        self.right_vqvae = VQVAETransformer1D(
            in_dim=hand_dim,
            d_model=hand_d_model,
            n_heads=n_hand_heads,
            n_layers_enc=n_hand_layers_enc,
            n_layers_dec=n_hand_layers_dec,
            ff_mult=ff_mult,
            dropout=dropout,
            code_dim=hand_code_dim,
            n_codes=n_hand_codes,
            stride=stride,
            rvq_stages=rvq_stages,
            vq_beta=vq_beta,
            loss_w=loss_w,
        )
        self.extra_vqvae = VQVAETransformer1D(
            in_dim=extra_dim,
            d_model=extra_d_model,
            n_heads=n_extra_heads,
            n_layers_enc=n_extra_layers_enc,
            n_layers_dec=n_extra_layers_dec,
            ff_mult=ff_mult,
            dropout=dropout,
            code_dim=extra_code_dim,
            n_codes=n_extra_codes,
            stride=stride,
            rvq_stages=rvq_stages,
            vq_beta=vq_beta,
            loss_w=loss_w,
        )
    def _gather_slices(self, x: torch.Tensor, slices: Tuple[slice, ...]) -> torch.Tensor:
        parts = [x[..., s] for s in slices]
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    def forward(self, x: torch.Tensor, hand_valid_mask: Optional[torch.Tensor] = None, input_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        # x: (B,T,36) -> split into pose/hand/extra
        pose = self._gather_slices(x, self.pose_idx)  # (B,T,pose_dim)
        l_hand = self._gather_slices(x, self.dir_l_idx)   # (B,T,hand_dim)
        r_hand= self._gather_slices(x, self.dir_r_idx)   # (B,T,hand_dim)
        extra = x[..., -self.extra_dim:]              # (B,T,extra_dim)

        out_pose = self.pose_vqvae(pose, hand_valid_mask=hand_valid_mask, input_length=input_length,no_return_loss=True)
        out_l_hand = self.left_vqvae(l_hand, hand_valid_mask=hand_valid_mask, input_length=input_length,no_return_loss=True)
        out_r_hand= self.right_vqvae(r_hand, hand_valid_mask=hand_valid_mask, input_length=input_length,no_return_loss=True)
        out_extra = self.extra_vqvae(extra, hand_valid_mask=hand_valid_mask, input_length=input_length,no_return_loss=True)

        pose_recon_loss=F.smooth_l1_loss(out_pose['x_hat'],pose,reduction='none')
        hand_recon_loss=(F.smooth_l1_loss(out_l_hand['x_hat'],l_hand,reduction='none')+F.smooth_l1_loss(out_r_hand['x_hat'],r_hand,reduction='none'))/2.0
        extra_recon_loss=F.smooth_l1_loss(out_extra['x_hat'],extra,reduction='none')
        if input_length is not None:
            mask=create_mask(input_length,max_len=x.shape[1]).unsqueeze(-1) #(B,T,1)
        else:
            mask=torch.ones(x.shape[0],x.shape[1],1,device=x.device)
        pose_recon_loss=(pose_recon_loss*mask).sum()/mask.sum()
        hand_recon_loss=(hand_recon_loss*mask).sum()/mask.sum()
        extra_recon_loss=(extra_recon_loss*mask).sum()/mask.sum()

        pose_vq_loss=out_pose['loss_vq']
        hand_vq_loss=(out_l_hand['loss_vq']+out_r_hand['loss_vq'])/2.0
        extra_vq_loss=out_extra['loss_vq']

        total_loss = self.loss_w.recon_pos * (pose_recon_loss + hand_recon_loss + extra_recon_loss) + \
                     self.loss_w.vq * (pose_vq_loss + hand_vq_loss + extra_vq_loss)
        perplexity=(out_pose['perplexity'] + out_l_hand['perplexity'] + out_r_hand['perplexity'] + out_extra['perplexity'])/4.0
        return {
            "loss_total": total_loss,
            "pose_recon_loss": pose_recon_loss.detach(),
            "hand_recon_loss": hand_recon_loss.detach(),
            "extra_recon_loss": extra_recon_loss.detach(),
            "pose_vq_loss": pose_vq_loss.detach(),
            "hand_vq_loss": hand_vq_loss.detach(),
            "extra_vq_loss": extra_vq_loss.detach(),
            "pose_codes": out_pose["codes"],
            "left_codes": out_l_hand["codes"],
            "right_codes": out_r_hand["codes"],
            "extra_codes": out_extra["codes"],
            "pose_x_hat": out_pose["x_hat"],
            "left_x_hat": out_l_hand["x_hat"],
            "right_x_hat": out_r_hand["x_hat"],
            "extra_x_hat": out_extra["x_hat"],
            "perplexity": perplexity.detach(),
        }
# -------------------------
# Quick sanity check
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C = 2, 100, 36
    x = torch.randn(B, T, C)

    # all valid
    mask = torch.ones(B, T, 2, dtype=torch.bool)

    model = VQVAETransformer1D(
        in_dim=36,
        d_model=256,
        n_heads=8,
        n_layers_enc=4,
        n_layers_dec=4,
        ff_mult=4,
        dropout=0.1,
        code_dim=128,
        n_codes=1024,
        stride=4,
        rvq_stages=2,   # 1 => plain VQ
        vq_beta=0.25,
    )

    out = model(x, hand_valid_mask=mask)
    print("x_hat:", out["x_hat"].shape)   # (B,T,36)
    print("codes:", out["codes"].shape)   # (B,stages,K) if RVQ else (B,K)
    print("loss_total:", float(out["loss_total"]))
    print("perplexity:", float(out["perplexity"]))