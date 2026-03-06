# VQ-VAE (1D temporal) for coarse sign pose: (B, T, 36) -> codes -> (B, T, 36)
# - Encoder/Decoder: TCN-style 1D conv over time
# - Temporal downsample by stride (e.g., 4) to shorten code sequence length
# - Supports (optional) Residual Vector Quantization (RVQ) with multiple codebooks
#
# Usage:
#   model = VQVAE1D(in_dim=36, hidden=256, code_dim=128, n_codes=1024, stride=4, rvq_stages=2)
#   out = model(x, mask_hand=mask)  # x: (B,T,36)
#   loss = out["loss_total"]
#
# Notes:
# - This code is designed for your "coarse pose" (body6 + (wrist,x,z)*2 = 36 dims).
# - For direction dims (x,z) you may prefer cosine loss; we include both position and direction losses.

import math
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
# Small building blocks
# -------------------------
class ResBlock1D(nn.Module):
    def __init__(self, ch: int, dropout: float = 0.0, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=pad, dilation=dilation),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, kernel_size=3, padding=pad, dilation=dilation),
        )

    def forward(self, x):
        return x + self.net(x)


class Downsample1D(nn.Module):
    def __init__(self, ch: int, stride: int):
        super().__init__()
        # kernel_size = 2*stride for smoother downsample
        k = 2 * stride
        p = stride // 2
        self.conv = nn.Conv1d(ch, ch, kernel_size=k, stride=stride, padding=p)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, ch: int, stride: int):
        super().__init__()
        # ConvTranspose with kernel_size = 2*stride
        k = 2 * stride
        p = stride // 2
        self.deconv = nn.ConvTranspose1d(ch, ch, kernel_size=k, stride=stride, padding=p, output_padding=stride % 2)

    def forward(self, x):
        return self.deconv(x)


# -------------------------
# Vector Quantizers
# -------------------------
class VectorQuantizer(nn.Module):
    """
    Standard VQ (nearest neighbor) with straight-through estimator.
    """
    def __init__(self, n_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta
        self.codebook = nn.Embedding(n_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / n_codes, 1.0 / n_codes)

    def forward(self, z_e: torch.Tensor,input_length=None) -> Dict[str, torch.Tensor]:
        """
        z_e: (B, C, K) where C=code_dim, K=compressed length
        returns dict with:
          z_q: quantized (B,C,K)
          codes: (B,K) indices
          loss_vq: scalar
          perplexity: scalar
        """
        B, C, K = z_e.shape
        assert C == self.code_dim

        # (B*K, C)
        z = z_e.permute(0, 2, 1).contiguous().view(-1, C)

        # compute distances to codebook: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        e = self.codebook.weight  # (n_codes, C)
        z2 = (z ** 2).sum(dim=1, keepdim=True)          # (B*K, 1)
        e2 = (e ** 2).sum(dim=1).unsqueeze(0)           # (1, n_codes)
        ze = z @ e.t()                                   # (B*K, n_codes)
        dist = z2 + e2 - 2 * ze                           # (B*K, n_codes)

        codes = torch.argmin(dist, dim=1)                 # (B*K,)
        z_q = self.codebook(codes).view(B, K, C).permute(0, 2, 1).contiguous()  # (B,C,K)

        # straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        loss_codebook = F.mse_loss(z_q, z_e.detach())
        loss_commit = F.mse_loss(z_e, z_q.detach())
        loss_vq = loss_codebook + self.beta * loss_commit

        # perplexity
        onehot = F.one_hot(codes, num_classes=self.n_codes).float()
        avg_probs = onehot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            "z_q": z_q_st,
            "codes": codes.view(B, K),
            "loss_vq": loss_vq,
            "perplexity": perplexity,
        }


class ResidualVectorQuantizer(nn.Module):
    """
    RVQ: quantize residuals in multiple stages.
    z_e -> q1 + q2(residual) + ... + qS
    """
    def __init__(self, n_codes: int, code_dim: int, stages: int = 2, beta: float = 0.25):
        super().__init__()
        self.stages = stages
        self.vqs = nn.ModuleList([VectorQuantizer(n_codes, code_dim, beta=beta) for _ in range(stages)])

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        residual = z_e
        z_sum = torch.zeros_like(z_e)
        total_loss = 0.0
        all_codes = []
        perplexities = []

        for vq in self.vqs:
            out = vq(residual)
            z_q = out["z_q"]  # straight-through
            z_sum = z_sum + z_q
            # Update residual using non-straight quantized value for stability:
            # We can approximate residual update by using detached quantized delta.
            residual = (residual - (z_q - residual).detach())  # keeps grad on residual path
            total_loss = total_loss + out["loss_vq"]
            all_codes.append(out["codes"])
            perplexities.append(out["perplexity"])

        codes = torch.stack(all_codes, dim=1)  # (B, stages, K)
        perplexity = torch.stack(perplexities).mean()

        return {
            "z_q": z_sum,
            "codes": codes,
            "loss_vq": total_loss,
            "perplexity": perplexity,
        }


# -------------------------
# VQ-VAE main model
# -------------------------
@dataclass
class VQLossWeights:
    recon_pos: float = 1.0
    recon_dir: float = 1.0
    vq: float = 1.0
    vel: float = 0.05  # small


class VQVAE1D(nn.Module):
    """
    Input x: (B, T, in_dim) where in_dim=36
    Internally uses (B, in_dim, T).
    Compress time by stride, quantize in code_dim space, decode back.
    """
    def __init__(
        self,
        in_dim: int = 36,
        hidden: int = 256,
        code_dim: int = 128,
        n_codes: int = 1024,
        stride: int = 4,
        n_res_blocks: int = 4,
        dropout: float = 0.0,
        rvq_stages: int = 1,
        vq_beta: float = 0.25,
        loss_w: VQLossWeights = VQLossWeights(),
        # which dims are "position-like" and which are direction-like
        # default for your coarse vector: body(18) + L(wrist(3),x(3),z(3)) + R(...)
        pos_idx: Tuple[slice, ...] = (slice(0, 18), slice(18, 21), slice(27, 30)),  # body + Lwrist + Rwrist
        dir_idx: Tuple[slice, ...] = (slice(21, 27), slice(30, 36)),  # L(x,z) + R(x,z)
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.code_dim = code_dim
        self.n_codes = n_codes
        self.stride = stride
        self.loss_w = loss_w
        self.pos_idx = pos_idx
        self.dir_idx = dir_idx

        # Encoder
        self.enc_in = nn.Conv1d(in_dim, hidden, kernel_size=3, padding=1)
        enc_blocks = []
        for i in range(n_res_blocks):
            enc_blocks.append(ResBlock1D(hidden, dropout=dropout, dilation=1))
        self.enc_blocks = nn.Sequential(*enc_blocks)
        self.down = Downsample1D(hidden, stride=stride)
        self.enc_out = nn.Conv1d(hidden, code_dim, kernel_size=1)

        # Quantizer
        if rvq_stages <= 1:
            self.quant = VectorQuantizer(n_codes=n_codes, code_dim=code_dim, beta=vq_beta)
            self.is_rvq = False
        else:
            self.quant = ResidualVectorQuantizer(n_codes=n_codes, code_dim=code_dim, stages=rvq_stages, beta=vq_beta)
            self.is_rvq = True

        # Decoder
        self.dec_in = nn.Conv1d(code_dim, hidden, kernel_size=1)
        self.up = Upsample1D(hidden, stride=stride)
        dec_blocks = []
        for i in range(n_res_blocks):
            dec_blocks.append(ResBlock1D(hidden, dropout=dropout, dilation=1))
        self.dec_blocks = nn.Sequential(*dec_blocks)
        self.dec_out = nn.Conv1d(hidden, in_dim, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,Cin) -> (B,Cin,T)
        x = x.transpose(1, 2).contiguous()
        h = self.enc_in(x)
        h = self.enc_blocks(h)
        h = self.down(h)      # (B,hidden,K)
        z_e = self.enc_out(h) # (B,code_dim,K)
        return z_e

    def decode(self, z_q: torch.Tensor, T_out: int) -> torch.Tensor:
        h = self.dec_in(z_q)
        h = self.up(h)        # (B,hidden,~T)
        h = self.dec_blocks(h)
        x_hat = self.dec_out(h)  # (B,in_dim,~T)

        # Crop/pad to match original T
        T_hat = x_hat.shape[-1]
        if T_hat > T_out:
            x_hat = x_hat[..., :T_out]
        elif T_hat < T_out:
            x_hat = F.pad(x_hat, (0, T_out - T_hat))

        # back to (B,T,C)
        return x_hat.transpose(1, 2).contiguous()

    @staticmethod
    def _cosine_loss(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8,input_length_mask=None) -> torch.Tensor:
        # a,b: (...,D)
        a_n = a / (a.norm(dim=-1, keepdim=True) + eps)
        b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
        return ((1.0 - (a_n * b_n).sum(dim=-1))*input_length_mask)/input_length_mask.sum()

    def _gather_slices(self, x: torch.Tensor, slices: Tuple[slice, ...]) -> torch.Tensor:
        # x: (B,T,C)
        parts = [x[..., s] for s in slices]
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    def forward(
        self,
        x: torch.Tensor,                 # (B,T,36)
        *,
        hand_valid_mask: Optional[torch.Tensor] = None,  # (B,T,2) bool/0-1 for (L,R)
        input_length: Optional[int] = None,  # if provided, will crop/pad input to this length
        return_recon: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        hand_valid_mask:
          - If provided: shape (B,T,2), where [:,:,0]=left_valid, [:,:,1]=right_valid
          - We'll ignore direction losses (and wrist loss) for invalid hands.
        """
        B, T, C = x.shape
        assert C == self.in_dim

        z_e = self.encode(x)
        if input_length is not None:
            mask=create_mask(input_length,max_len=T).unsqueeze(-1) #(B,T,1)
            mask_k=downsample_mask_to_latent(mask.squeeze(-1), z_e.shape[-1]) #(B,K)
        else:
            mask=torch.ones(B,T,1,device=x.device)
            mask_k=torch.ones(B,z_e.shape[-1],device=x.device)
        z_e = z_e * mask_k.unsqueeze(1)  # zero out invalid latent
        q = self.quant(z_e)
        z_q = q["z_q"]
        x_hat = self.decode(z_q, T_out=T)

        # -------------------------
        # Losses
        # -------------------------
        # position-like reconstruction (body + wrists)
        x_pos = self._gather_slices(x, self.pos_idx)
        h_pos = self._gather_slices(x_hat, self.pos_idx)
        loss_recon_pos = F.smooth_l1_loss(h_pos, x_pos, reduction='none')

        loss_recon_pos=(loss_recon_pos*mask).sum()/mask.sum()

        # direction-like reconstruction (x,z) with cosine loss
        x_dir = self._gather_slices(x, self.dir_idx)
        h_dir = self._gather_slices(x_hat, self.dir_idx)

        # If mask is provided, mask out invalid hands directions.
        # Our dir slices correspond to [Lx,Lz,Rx,Rz] => 12 dims total (two 6-d blocks).
        # We'll build a per-frame mask for L/R and apply.
        if hand_valid_mask is not None:
            # hand_valid_mask: (B,T,2)
            mL = hand_valid_mask[..., 0].float()  # (B,T)
            mR = hand_valid_mask[..., 1].float()
            # expand to direction dims:
            # L block is first 6 dims, R block last 6 dims
            m_dir = torch.cat([mL[..., None].repeat(1, 1, 6), mR[..., None].repeat(1, 1, 6)], dim=-1)  # (B,T,12)
            # avoid all-zero
            denom = m_dir.sum().clamp_min(1.0)
            # cosine loss per vector chunk: compute per 3-d vector then mask
            # reshape to vectors: (B,T,4,3) => [Lx,Lz,Rx,Rz]
            x_vec = x_dir.view(B, T, 4, 3)
            h_vec = h_dir.view(B, T, 4, 3)
            m_vec = torch.stack([mL, mL, mR, mR], dim=-1)  # (B,T,4)
            # cosine similarity per vec
            eps = 1e-8
            x_n = x_vec / (x_vec.norm(dim=-1, keepdim=True) + eps)
            h_n = h_vec / (h_vec.norm(dim=-1, keepdim=True) + eps)
            cos = (x_n * h_n).sum(dim=-1)  # (B,T,4)
            loss_recon_dir = ((1.0 - cos) * m_vec).sum() / m_vec.sum().clamp_min(1.0)
        else:
            # unmasked cosine loss across all direction dims as 3D vectors
            x_vec = x_dir.view(B, T, -1, 3)
            h_vec = h_dir.view(B, T, -1, 3)
            loss_recon_dir = self._cosine_loss(h_vec, x_vec)

        # velocity smoothness (small)
        vel_hat = x_hat[:, 1:] - x_hat[:, :-1]
        vel = x[:, 1:] - x[:, :-1]
        vel_mask=mask[:, 1:] * mask[:, :-1]  # only consider velocity where both frames are valid
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
            "codes": q["codes"],  # (B,K) for VQ, (B,stages,K) for RVQ
        }
        if return_recon:
            out["x_hat"] = x_hat
        return out


# -------------------------
# Quick sanity check
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C = 2, 100, 36
    x = torch.randn(B, T, C)

    # Example hand validity mask: all valid
    mask = torch.ones(B, T, 2, dtype=torch.bool)

    model = VQVAE1D(
        in_dim=36,
        hidden=256,
        code_dim=128,
        n_codes=1024,
        stride=4,
        n_res_blocks=4,
        dropout=0.0,
        rvq_stages=2,   # set 1 for plain VQ
        vq_beta=0.25,
    )

    out = model(x, hand_valid_mask=mask)
    print("x_hat:", out["x_hat"].shape)
    print("codes:", out["codes"].shape)
    print("loss_total:", float(out["loss_total"]))
    print("perplexity:", float(out["perplexity"]))