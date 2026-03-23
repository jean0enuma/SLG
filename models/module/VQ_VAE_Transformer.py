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
    mask = torch.ones((batch_size, max_len), dtype=torch.float32, device=target_length.device)
    for i in range(batch_size):
        mask[i, :target_length[i]] = 0.0
    return mask  # (batch_size, max_len)
def create_slide_window_mask(length,window_size,device):
    mask=torch.ones((length, length), dtype=torch.bool, device=device)
    for i in range(length):
        start=max(0,i-window_size//2)
        end=min(length,i+window_size//2+1)
        mask[i,start:end]=False
    return mask  # (length, length)
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


class VectorQuantizerWithRestart(nn.Module):
    def __init__(self, n_codes: int, code_dim: int, beta: float = 0.25, restart_threshold: float = 1.0):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta
        self.restart_threshold = restart_threshold

        self.codebook = nn.Embedding(n_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / n_codes, 1.0 / n_codes)

        # 使用頻度を記録するバッファ (EMA)
        self.register_buffer('usage_count', torch.zeros(n_codes))

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.training:
            return self.inference(z_e)
        B, K, C = z_e.shape
        assert C == self.code_dim

        z = z_e.reshape(-1, C)  # (B*K, C)
        e = self.codebook.weight  # (M, C)

        # 1. 距離計算とインデックス抽出
        z2 = (z ** 2).sum(dim=1, keepdim=True)
        e2 = (e ** 2).sum(dim=1).unsqueeze(0)
        ze = z @ e.t()
        dist = z2 + e2 - 2 * ze

        idx = torch.argmin(dist, dim=1)  # (B*K,)

        # --- Random Restart 用の統計更新 ---
        if self.training:
            current_usage = torch.bincount(idx, minlength=self.n_codes).float()
            self.usage_count.mul_(0.99).add_(current_usage, alpha=0.01)

        # 2. 量子化
        z_q = self.codebook(idx).view(B, K, C)

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

    @torch.no_grad()
    def inference(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, K, C = z_e.shape
        assert C == self.code_dim

        z = z_e.reshape(-1, C)
        e = self.codebook.weight

        z2 = (z ** 2).sum(dim=1, keepdim=True)
        e2 = (e ** 2).sum(dim=1).unsqueeze(0)
        ze = z @ e.t()
        dist = z2 + e2 - 2 * ze

        idx = torch.argmin(dist, dim=1)
        q = self.codebook(idx).view(B, K, C)
        # perplexity
        encodings = torch.zeros(z.shape[0], self.n_codes, device=z.device)
        encodings.scatter_(1, idx.reshape([-1, 1]), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-12)))

        return {
            "z_q": q,
            "codes": idx.view(B, K),
            "loss_vq": torch.tensor(0.0, device=z.device),  # ノイズ置換型は通常のVQ損失を使用しない
            "perplexity": perplexity,
        }

    @torch.no_grad()
    def random_restart(self, z_flat,threshold=None):
        """利用率が低いコードを現在の入力 z_e で再初期化する"""
        if threshold==None:
            threshold=self.restart_threshold
        dead_indices = torch.where(self.usage_count < self.restart_threshold)[0]

        if len(dead_indices) > 0:
            print("Random restart: resetting {} codes".format(len(dead_indices)))
            n_resets = min(len(dead_indices), z_flat.size(0))
            random_indices = torch.randperm(z_flat.size(0), device=z_flat.device)[:n_resets]

            # 型(dtype)を代入先に合わせてキャスト
            replacement = z_flat[random_indices].to(dtype=self.codebook.weight.dtype)

            # コードブックの上書き
            self.codebook.weight.data[dead_indices[:n_resets]] = replacement+torch.randn_like(replacement)*0.01  # 少しノイズを加えて多様性を促す
            # 使用頻度バッファのリセット（すぐにまた消去されないようにthresholdで初期化）
            self.usage_count[dead_indices[:n_resets]] = threshold
class NoiseSubstitutionVQ(nn.Module):
    """
    ノイズ置換型VQ: z_q = (1 - m) * q + m * noise
    m: (B,K,1) 0/1 mask for substitution
    noise: (B,K,C) Gaussian noise
    """

    def __init__(self, n_codes: int, code_dim: int, beta: float = 0.25, restart_threshold: float = 1.0):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta
        self.restart_threshold = restart_threshold

        self.codebook = nn.Embedding(n_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / n_codes, 1.0 / n_codes)

        # 使用頻度を記録するバッファ (EMA)
        self.register_buffer("usage_count", torch.zeros(n_codes))

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.training:
                return self.inference(z_e)
        else:
            B, K, C = z_e.shape
            assert C == self.code_dim

            z = z_e.reshape(-1, C)
            e = self.codebook.weight

            z2 = (z ** 2).sum(dim=1, keepdim=True)
            e2 = (e ** 2).sum(dim=1).unsqueeze(0)
            ze = z @ e.t()
            dist = z2 + e2 - 2 * ze

            idx = torch.argmin(dist, dim=1)# (B*K,)
            current_usage = torch.bincount(idx, minlength=self.n_codes).float()
            self.usage_count.mul_(0.99).add_(current_usage, alpha=0.01)
            q = self.codebook(idx)# (B*K,C)
            #self._restart_dead_codes(z)

            #zとqの距離が大きい場合にノイズ置換
            z_hat=torch.linalg.norm(z-q,dim=1,keepdim=True)  # (B*K,1)
            #gauassian random vector
            noise = torch.randn_like(z)
            norm_noise=torch.linalg.norm(noise, dim=-1, keepdim=True)# (B*K,1)
            vq_error=(z_hat / (norm_noise + 1e-10))*noise  # (B*K,1) 0~1
            quantized=(z+vq_error).view(B,K,C)  # (B,K,C)
            # perplexity
            # claculating the perplexity (average usage of codebook entries)
            encodings = torch.zeros(z.shape[0], self.n_codes, device=z.device)
            encodings.scatter_(1, idx.reshape([-1, 1]), 1)
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-12)))

            return {
                "z_q": quantized,
                "z": z,
                "codes": idx.view(B, K),
                "loss_vq": torch.tensor(0.0, device=z.device),  # ノイズ置換型は通常のVQ損失を使用しない
                "perplexity": perplexity,
            }
    @torch.no_grad()
    def inference(self,z_e:torch.Tensor)->Dict[str,torch.Tensor]:
        B, K, C = z_e.shape
        assert C == self.code_dim

        z = z_e.reshape(-1, C)
        e = self.codebook.weight

        z2 = (z ** 2).sum(dim=1, keepdim=True)
        e2 = (e ** 2).sum(dim=1).unsqueeze(0)
        ze = z @ e.t()
        dist = z2 + e2 - 2 * ze

        idx = torch.argmin(dist, dim=1)
        q = self.codebook(idx).view(B,K,C)
        # perplexity
        encodings = torch.zeros(z.shape[0], self.n_codes, device=z.device)
        encodings.scatter_(1, idx.reshape([-1, 1]), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-12)))

        return {
            "z_q": q,
            "codes": idx.view(B, K),
            "loss_vq": torch.tensor(0.0, device=z.device),  # ノイズ置換型は通常のVQ損失を使用しない
            "perplexity": perplexity,
        }

    @torch.no_grad()
    def random_restart(self, z_flat,threshold=None):
        """利用率が低いコードを現在の入力 z_e で再初期化する"""
        dead_indices = torch.where(self.usage_count < self.restart_threshold)[0]

        if len(dead_indices) > 0:
            print("Random restart: resetting {} codes".format(len(dead_indices)))
            n_resets = min(len(dead_indices), z_flat.size(0))
            random_indices = torch.randperm(z_flat.size(0), device=z_flat.device)[:n_resets]

            # 型(dtype)を代入先に合わせてキャスト
            replacement = z_flat[random_indices].to(dtype=self.codebook.weight.dtype)

            # コードブックの上書き
            self.codebook.weight.data[dead_indices[:n_resets]] = replacement+torch.randn_like(replacement)*0.01  # 少しノイズを加えて多様性を促す
            # 使用頻度バッファのリセット（すぐにまた消去されないように平均値等で初期化）
            self.usage_count[dead_indices[:n_resets]] = threshold
class VectorQuantizerCategorical(VectorQuantizerWithRestart):
    def __init__(self, n_codes: int, code_dim: int, beta: float = 0.25, restart_threshold: float = 1.0,tau=1.0,hard=False):
        super().__init__(n_codes, code_dim, beta, restart_threshold)
        self.tau=tau
        self.hard=hard
    def forward(self,z_e):
        B, K, C = z_e.shape
        assert C == self.code_dim

        z = z_e.reshape(-1, C)  # (B*K, M)
        e = self.codebook.weight  # (M, C)
        z2 = (z ** 2).sum(dim=1, keepdim=True)
        e2 = (e ** 2).sum(dim=1).unsqueeze(0)
        ze = z @ e.t()
        dist = z2 + e2 - 2 * ze# (B*K,M)

        #gambel-softmaxで近似的にone-hot化
        if self.training:
            code_prob=F.gumbel_softmax(-dist,tau=self.tau,hard=self.hard,dim=-1)  # (B*K,M)
            q=code_prob @ e  # (B*K,C)
            kl_discrete = torch.sum(code_prob * torch.log(code_prob + 1e-12), dim=-1).mean()  # (B*K,)
            # perplexity
            avg_probs = code_prob.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-12)))
            indices=torch.argmax(code_prob, dim=-1)  # (B*K,)

        else:
            indices=torch.argmin(dist,dim=-1)  # (B*K,)
            q=self.codebook(indices)  # (B*K,C)
            kl_discrete=torch.tensor(0.0, device=z.device)  # 推論時はKLはゼロとする
            avg_probs=F.one_hot(indices,num_classes=self.n_codes).float().mean(dim=0)  # (M,)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-12)))

        quantized=q.view(B,K,self.code_dim)  # (B,K,C)
        return {
            "z_q": quantized,
            "codes": indices.view(B, K),
            "loss_vq": kl_discrete,  # カテゴリカルVQは通常のVQ損失を使用しない
            "perplexity": perplexity,
            "dist": dist.view(B,K,self.n_codes),  # (B,K,M) 距離も返す（必要に応じて）
        }
    def random_restart(self, z_flat,threshold=None):
        pass  # カテゴリカルVQはコードブックの更新がないためランダムリスタートは不要


class ResidualVectorQuantizer(nn.Module):
    """
    RVQ: z ≈ q1 + q2(residual) + ... + qS
    """
    def __init__(self, n_codes: int, code_dim: int, stages: int = 2, beta: float = 0.25):
        super().__init__()
        self.stages = stages
        self.vqs = nn.ModuleList([NoiseSubstitutionVQ(n_codes, code_dim, beta=beta) for _ in range(stages)])

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
    @torch.no_grad()
    def random_restart(self, z_flat,threshold=None):
        pass

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
        pos_idx: Tuple[slice, ...] = (slice(0, 12), slice(12, 14), slice(54, 56)),  # body + Lwrist + Rwrist
            dir_idx: Tuple[slice, ...] = (slice(12, 54),slice(54, 96)),  # L(x,z) + R(x,z)
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
            self.quant = NoiseSubstitutionVQ(n_codes=n_codes, code_dim=code_dim, beta=vq_beta)
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
            mask=create_mask(input_length,max_len=T) #(B,T,1)
        else:
            mask=torch.ones(B,T,device=x.device)

        z_e = z_e
        q = self.quant(z_e)                # dict
        z_q = q["z_q"]                     # (B,K,code_dim)
        x_hat = self.decode(z_q, T_out=T)  # (B,T,36)
        if no_return_loss:
            return {"x_hat": x_hat, "codes": q["codes"],"perplexity": q["perplexity"].detach(), "loss_vq": q["loss_vq"]}

        # ---- losses: pos (L1/Huber), dir (cos), vel (small), vq ----
        x_pos = self._gather_slices(x, self.pos_idx)
        h_pos = self._gather_slices(x_hat, self.pos_idx)
        loss_recon_pos = F.smooth_l1_loss(h_pos, x_pos, reduction='none').mean(dim=-1)  # (B,T)

        loss_recon_pos=(loss_recon_pos*mask).sum()/mask.sum()
        x_dir = self._gather_slices(x, self.dir_idx)
        h_dir = self._gather_slices(x_hat, self.dir_idx)

        # cosine distance per vector, then mean over vectors
        loss_recon_dir=F.smooth_l1_loss(h_dir, x_dir, reduction='none')# (B,T,dir_dim)3
        left_loss_recon_dir=loss_recon_dir[..., :x_dir.shape[-1]//2].mean(dim=-1) #(B,T)
        right_loss_recon_dir=loss_recon_dir[..., x_dir.shape[-1]//2:].mean(dim=-1) #(B,T)
        dir_loss_per=torch.cat([left_loss_recon_dir.unsqueeze(-1), right_loss_recon_dir.unsqueeze(-1)], dim=-1) #(B,T,2)
        if hand_valid_mask is not None:
            # mask: (B,T,2) -> (B,T,4) for [Lx,Lz,Rx,Rz]
            mL = hand_valid_mask[..., 0].float()
            mR = hand_valid_mask[..., 1].float()
            m = torch.stack([mL,mR], dim=-1)  # (B,T,2)
            loss_recon_dir = (dir_loss_per * m).sum() / m.sum().clamp_min(1.0)
        else:
            loss_recon_dir = dir_loss_per.mean()

        vel_hat = x_hat[:, 1:] - x_hat[:, :-1]
        vel = x[:, 1:] - x[:, :-1]
        vel_mask = mask[:, 1:] * mask[:, :-1]  # only consider velocity where both frames are valid
        loss_vel = F.l1_loss(vel_hat, vel, reduction='none').mean(dim=-1)  # (B,T-1)
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
            'z_e': z_e.detach(),
        }
        if return_recon:
            out["x_hat"] = x_hat
        return out
    @torch.no_grad()
    def random_restart(self, z_e: torch.Tensor,threshold=None):
        B, K, C = z_e.shape
        z = z_e.reshape(-1, C)
        self.quant.random_restart(z,threshold=threshold)
class VQVAETransformer1DSimple(VQVAETransformer1D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

        z_e = self.encode(x)  # (B,K,code_dim)
        if input_length is not None:
            mask = create_mask(input_length, max_len=T)  # (B,T,1)
        else:
            mask = torch.ones(B, T, device=x.device)

        z_e = z_e
        q = self.quant(z_e)  # dict
        z_q = q["z_q"]  # (B,K,code_dim)
        x_hat = self.decode(z_q, T_out=T)  # (B,T,36)
        if no_return_loss:
            return {"x_hat": x_hat, "codes": q["codes"], "perplexity": q["perplexity"].detach(),
                    "loss_vq": q["loss_vq"]}
        recon_loss=F.mse_loss(x_hat, x, reduction='none') #(B,T)
        recon_loss=(recon_loss.mean(dim=-1)*mask).sum()/(mask.sum())
        loss_vq = q["loss_vq"]
        input_vel=x[:, 1:] - x[:, :-1]
        recon_vel=x_hat[:, 1:] - x_hat[:, :-1]
        vel_mask=mask[:, 1:] * mask[:, :-1]
        loss_vel=F.mse_loss(recon_vel, input_vel, reduction='none').mean(dim=-1) #(B,T-1)
        loss_vel=(loss_vel*vel_mask).sum()/vel_mask.sum()
        loss_total = self.loss_w.recon_pos * recon_loss + self.loss_w.vq * loss_vq+self.loss_w.vel*loss_vel
        out = {
            "loss_total": loss_total,
            "loss_recon_pos": recon_loss.detach(),
            "loss_vel": loss_vel.detach(),
            "loss_vq": loss_vq.detach(),
            "perplexity": q["perplexity"].detach(),
            "codes": q["codes"],  # (B,K) for VQ, (B,stages,K) for RVQ
        }
        if return_recon:
            out["x_hat"] = x_hat
        return out


class VQVAETransformer1DSeparated(nn.Module):
    def __init__(
            self,
            pose_dim: int = 16,
            hand_dim: int = 42,
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
            pose_idx: Tuple[slice, ...] = (slice(0, 12), slice(12,14), slice(54, 56)),  # body + Lwrist + Rwrist
            dir_l_idx: Tuple[slice, ...] = (slice(12, 54),),               # L(x,z) + R(x,z)
            dir_r_idx: Tuple[slice, ...] = (slice(54, 96),)
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
        self.right_vqvae = self.left_vqvae

    @torch.no_grad()
    def code_usage_histogram_update(
            self,
            x: torch.Tensor,
            pose_prev_hist: Optional[Dict[str, torch.Tensor]] = None,
            left_prev_hist: Optional[Dict[str, torch.Tensor]] = None,
            right_prev_hist: Optional[Dict[str, torch.Tensor]] = None,
            *,
            input_length: Optional[torch.Tensor] = None,
            hand_valid_mask: Optional[torch.Tensor] = None,
            per_stage: bool = True,
            normalize: bool = True,
    ):
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
        )
        pose_codes=out["pose_codes"]  # (B,K) or (B,S,K)
        l_hand_codes=out["left_codes"]  # (B,K) or (B,S,K)
        r_hand_codes=out["right_codes"]  # (B,K
        # latent mask (B,K) to ignore padded frames if input_length is provided
        K_p = pose_codes.shape[-1]  # assuming all have same K
        K_l= l_hand_codes.shape[-1]
        K_r= r_hand_codes.shape[-1]
        pose_n_codes=self.pose_vqvae.n_codes
        left_n_codes=self.left_vqvae.n_codes
        right_n_codes=self.right_vqvae.n_codes

        if input_length is not None:
            if input_length.dim() != 1:
                raise ValueError("input_length must be a 1D tensor of shape (B,)")
            B, T = x.shape[0], x.shape[1]
            time_mask = create_mask(input_length.to(x.device), max_len=T)  # (B,T) float
            latent_mask = downsample_mask_to_latent(time_mask, K=K_p).bool()  # (B,K)
        else:
            latent_mask = None

        device = x.device

        # Initialize accumulators if needed
        if pose_prev_hist is None or left_prev_hist is None or right_prev_hist is None:
            pose_prev_hist = {}
            left_prev_hist = {}
            right_prev_hist = {}
            pose_prev_hist["counts"] = torch.zeros(pose_n_codes, dtype=torch.long, device=device)
            left_prev_hist["counts"] = torch.zeros(left_n_codes, dtype=torch.long, device=device)
            right_prev_hist["counts"] = torch.zeros(right_n_codes, dtype=torch.long, device=device)
            if pose_codes.dim() == 3 and per_stage:
                S = pose_codes.shape[1]
                pose_prev_hist["counts_per_stage"] = torch.zeros(S, pose_n_codes, dtype=torch.long, device=device)
                left_prev_hist["counts_per_stage"] = torch.zeros(S, left_n_codes, dtype=torch.long, device=device)
                right_prev_hist["counts_per_stage"] = torch.zeros(S, right_n_codes, dtype=torch.long, device=device)
        def _batch_counts(code_bk: torch.Tensor, mask_bk: Optional[torch.Tensor],n_codes: int) -> torch.Tensor:
            # code_bk: (B,K) int
            if mask_bk is not None:
                flat = code_bk[mask_bk].reshape(-1)
            else:
                flat = code_bk.reshape(-1)
            flat = flat.to(torch.int64)
            return torch.bincount(flat, minlength=n_codes).to(device)

        # ---- accumulate ----
        if pose_codes.dim() == 2:
            c_batch = _batch_counts(pose_codes, None,n_codes=pose_n_codes)
            pose_prev_hist["counts"] = pose_prev_hist["counts"] + c_batch
            l_batch=_batch_counts(l_hand_codes, None,n_codes=left_n_codes)
            r_batch=_batch_counts(r_hand_codes, None,n_codes=right_n_codes)
            left_prev_hist["counts"]=left_prev_hist["counts"]+l_batch
            right_prev_hist["counts"]=right_prev_hist["counts"]+r_batch

        elif pose_codes.dim() == 3:
            B, S, K = pose_codes.shape
            # aggregate across stages
            flat_all = pose_codes.reshape(-1).to(torch.int64)
            flat_left= l_hand_codes.reshape(-1).to(torch.int64)
            flat_right= r_hand_codes.reshape(-1).to(torch.int64)
            c_batch = torch.bincount(flat_all, minlength=pose_n_codes).to(device)
            l_batch= torch.bincount(flat_left, minlength=left_n_codes).to(device)
            r_batch= torch.bincount(flat_right, minlength=right_n_codes).to(device)
            pose_prev_hist["counts"] = pose_prev_hist["counts"] + c_batch
            left_prev_hist["counts"]=left_prev_hist["counts"]+l_batch
            right_prev_hist["counts"]=right_prev_hist["counts"]+r_batch

            if per_stage:
                if "counts_per_stage" not in pose_prev_hist:
                    pose_prev_hist["counts_per_stage"] = torch.zeros(S, pose_n_codes, dtype=torch.long, device=device)
                    left_prev_hist["counts_per_stage"] = torch.zeros(S, left_n_codes, dtype=torch.long, device=device)
                    right_prev_hist["counts_per_stage"] = torch.zeros(S, right_n_codes, dtype=torch.long, device=device)
                for s in range(S):
                    cs = _batch_counts(pose_codes[:, s, :], None,n_codes=pose_n_codes)
                    ls = _batch_counts(l_hand_codes[:, s, :], None,n_codes=left_n_codes)
                    rs = _batch_counts(r_hand_codes[:, s, :], None,n_codes=right_n_codes)
                    pose_prev_hist["counts_per_stage"][s] = pose_prev_hist["counts_per_stage"][s] + cs
                    left_prev_hist["counts_per_stage"][s] = left_prev_hist["counts_per_stage"][s] + ls
                    right_prev_hist["counts_per_stage"][s] = right_prev_hist["counts_per_stage"][s] + rs
        else:
            raise ValueError(f"Unexpected codes shape: {tuple(pose_codes.shape)}")

        # ---- compute probs/perplexity from accumulated counts (optional) ----
        if normalize:
            pose_counts = pose_prev_hist["counts"]
            left_counts = left_prev_hist["counts"]
            right_counts = right_prev_hist["counts"]
            pose_total = pose_counts.sum().clamp_min(1)
            pose_probs = pose_counts.float() / pose_total.float()
            left_probs = left_counts.float() / left_counts.sum().clamp_min(1).float()
            right_probs = right_counts.float() / right_counts.sum().clamp_min(1).float()
            pose_perplex = torch.exp(-(pose_probs * torch.log(pose_probs.clamp_min(1e-10))).sum())
            left_perplex = torch.exp(-(left_probs * torch.log(left_probs.clamp_min(1e-10))).sum())
            right_perplex = torch.exp(-(right_probs * torch.log(right_probs.clamp_min(1e-10))).sum())
            pose_prev_hist["probs"] = pose_probs
            left_prev_hist["probs"] = left_probs
            right_prev_hist["probs"] = right_probs
            pose_prev_hist["perplexity_from_hist"] = pose_perplex
            left_prev_hist["perplexity_from_hist"] = left_perplex
            right_prev_hist["perplexity_from_hist"] = right_perplex

            if "counts_per_stage" in pose_prev_hist:
                pose_cps = pose_prev_hist["counts_per_stage"]
                l_cps = left_prev_hist["counts_per_stage"]
                r_cps = right_prev_hist["counts_per_stage"]
                pose_totals = pose_cps.sum(dim=1, keepdim=True).clamp_min(1)
                l_totals = l_cps.sum(dim=1, keepdim=True).clamp_min(1)
                r_totals = r_cps.sum(dim=1, keepdim=True).clamp
                pose_pps = pose_cps.float() / pose_totals.float()
                l_pps = l_cps.float() / l_totals.float()
                r_pps = r_cps.float() / r_totals.float()
                pose_pxs = torch.exp(-(pose_pps * torch.log(pose_pps.clamp_min(1e-10))).sum(dim=1))
                l_pxs = torch.exp(-(l_pps * torch.log(l_pps.clamp_min(1e-10))).sum(dim=1))
                r_pxs = torch.exp(-(r_pps * torch.log(r_pps.clamp_min(1e-10))).sum(dim=1))
                pose_prev_hist["probs_per_stage"] = pose_pps
                left_prev_hist["probs_per_stage"] = l_pps
                right_prev_hist["probs_per_stage"] = r_pps
                pose_prev_hist["perplexity_per_stage"] = pose_pxs
                left_prev_hist["perplexity_per_stage"] = l_pxs
                right_prev_hist["perplexity_per_stage"] = r_pxs
        else:
            pose_prev_hist["probs"] = torch.zeros(pose_n_codes, dtype=torch.float32, device=device)
            pose_prev_hist["perplexity_from_hist"] = torch.tensor(float("nan"), device=device)
            left_prev_hist["probs"] = torch.zeros(left_n_codes, dtype=torch.float32, device=device)
            left_prev_hist["perplexity_from_hist"] = torch.tensor(float("nan"), device=device)
            right_prev_hist["probs"] = torch.zeros(right_n_codes, dtype=torch.float32, device=device)
            right_prev_hist["perplexity_from_hist"] = torch.tensor(float("nan"), device=device)

        return pose_prev_hist, left_prev_hist, right_prev_hist
    def _gather_slices(self, x: torch.Tensor, slices: Tuple[slice, ...]) -> torch.Tensor:
        parts = [x[..., s] for s in slices]
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    def forward(self, x: torch.Tensor, hand_valid_mask: Optional[torch.Tensor] = None, input_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        # x: (B,T,36) -> split into pose/hand/extra
        pose = self._gather_slices(x, self.pose_idx)  # (B,T,pose_dim)
        l_hand = self._gather_slices(x, self.dir_l_idx)   # (B,T,hand_dim)
        r_hand= self._gather_slices(x, self.dir_r_idx)   # (B,T,hand_dim)
        #extra = x[..., -self.extra_dim:]              # (B,T,extra_dim)

        out_pose = self.pose_vqvae(pose, hand_valid_mask=hand_valid_mask, input_length=input_length,no_return_loss=True)
        out_l_hand = self.left_vqvae(l_hand, hand_valid_mask=hand_valid_mask, input_length=input_length,no_return_loss=True)
        out_r_hand= self.right_vqvae(r_hand, hand_valid_mask=hand_valid_mask, input_length=input_length,no_return_loss=True)
        #out_extra = self.extra_vqvae(extra, hand_valid_mask=hand_valid_mask, input_length=input_length,no_return_loss=True)

        pose_recon_loss=F.smooth_l1_loss(out_pose['x_hat'],pose,reduction='none')
        hand_recon_loss=(F.smooth_l1_loss(out_l_hand['x_hat'],l_hand,reduction='none')+F.smooth_l1_loss(out_r_hand['x_hat'],r_hand,reduction='none'))/2.0
        #extra_recon_loss=F.smooth_l1_loss(out_extra['x_hat'],extra,reduction='none')
        if input_length is not None:
            mask=create_mask(input_length,max_len=x.shape[1]).unsqueeze(-1) #(B,T,1)
        else:
            mask=torch.ones(x.shape[0],x.shape[1],1,device=x.device)
        pose_recon_loss=(pose_recon_loss*mask).sum()/mask.sum()
        hand_recon_loss=(hand_recon_loss*mask).sum()/mask.sum()
        #extra_recon_loss=(extra_recon_loss*mask).sum()/mask.sum()

        pose_vq_loss=out_pose['loss_vq']
        hand_vq_loss=(out_l_hand['loss_vq']+out_r_hand['loss_vq'])/2.0
       # extra_vq_loss=out_extra['loss_vq']

        total_loss = self.loss_w.recon_pos * (pose_recon_loss + hand_recon_loss) + \
                     self.loss_w.vq * (pose_vq_loss + hand_vq_loss)
        perplexity=(out_pose['perplexity'] + out_l_hand['perplexity'] + out_r_hand['perplexity'])/3.0
        return {
            "loss_total": total_loss,
            "pose_recon_loss": pose_recon_loss.detach(),
            "hand_recon_loss": hand_recon_loss.detach(),
            "extra_recon_loss": torch.tensor(0.0, device=x.device),  # placeholder since extra is not used
            "pose_vq_loss": pose_vq_loss.detach(),
            "hand_vq_loss": hand_vq_loss.detach(),
            "extra_vq_loss": torch.tensor(0.0, device=x.device),  # placeholder since extra is not used
            "pose_codes": out_pose["codes"],
            "left_codes": out_l_hand["codes"],
            "right_codes": out_r_hand["codes"],
            "extra_codes": torch.tensor(0.0, device=x.device),  # placeholder since extra is not used
            "pose_x_hat": out_pose["x_hat"],
            "left_x_hat": out_l_hand["x_hat"],
            "right_x_hat": out_r_hand["x_hat"],
            "extra_x_hat":torch.tensor(0.0, device=x.device),  # placeholder since extra is not used
            "perplexity": perplexity.detach(),
        }
class VQVAETransformer1DEncoder(nn.Module):
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
            dir_idx: Tuple[slice, ...] = (slice(21, 27), slice(32, 38)),  # L(x,z) + R(x,z)
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
    def forward(self,x,padding_mask=None,attn_mask=None):
        B, T, C = x.shape
        assert C == self.in_dim

        h = self.in_proj(x)  # (B,T,d_model)
        pe = self.pos_emb(T, device=h.device, dtype=h.dtype)  # (1,T,d_model)
        h = h + pe
        h = self.encoder(h, src_key_padding_mask=padding_mask, mask=attn_mask)  # (B,T,d_model)
        # downsample
        h_t = h.transpose(1, 2).contiguous()   # (B,d_model,T)
        h_k = self.down(h_t).transpose(1, 2).contiguous()  # (B,K,d_model)
        return h_k
class VQVAETransformer1DDecoder(nn.Module):
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
            dir_idx: Tuple[slice, ...] = (slice(21, 27), slice(32, 38)),  # L(x,z) + R(x,z)
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

        # ---- Decoder ----
        self.from_code = nn.Linear(code_dim, d_model)

        # upsample in time (learned)
        # (B,K,d_model) -> (B,d_model,K) -> ConvTranspose1d stride -> (B,d_model,T)
        if stride > 1:
            self.up = nn.ConvTranspose1d(d_model, d_model, kernel_size=2 * stride, stride=stride, padding=stride // 2)
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
    @staticmethod
    def _safe_crop_or_pad_time(x: torch.Tensor, T: int) -> torch.Tensor:
        # x: (B, T_hat, C)
        T_hat = x.shape[1]
        if T_hat > T:
            return x[:, :T, :]
        if T_hat < T:
            return F.pad(x, (0, 0, 0, T - T_hat))
        return x

    def forward(self,z_q,T_out,padding_mask=None,attn_mask=None):
        h = self.from_code(z_q)  # (B,K,d_model)

        # upsample
        h_t = h.transpose(1, 2).contiguous()     # (B,d_model,K)
        h_up = self.up(h_t).transpose(1, 2).contiguous()  # (B,T_hat,d_model)

        h_up = self._safe_crop_or_pad_time(h_up, T_out)

        # add positional embedding at target length
        pe = self.pos_emb(T_out, device=h_up.device, dtype=h_up.dtype)
        h_up = h_up + pe
        h_up = self.decoder(h_up, src_key_padding_mask=padding_mask, mask=attn_mask)  # (B,T,d_model)
        h_up = self.dec_ln_out(h_up)
        x_hat = self.out_proj(h_up)
        return x_hat
class VQVAETransformer1DAggregated(nn.Module):
    def __init__(
            self,
            n_codes=1024,
            code_dim=128,
            pose_dim: int = 16,
            hand_dim: int = 42,
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
            pose_idx: Tuple[slice, ...] = (slice(0, 6), slice(6, 7), slice(27, 28)),  # body + Lwrist + Rwrist
            dir_l_idx: Tuple[slice, ...] = (slice(6, 27),),  # L(x,z) + R(x,z)
            dir_r_idx: Tuple[slice, ...] = (slice(27, 48),),
            tau=1.0,
    ):
        super().__init__()
        self.pose_idx = pose_idx
        self.dir_l_idx = dir_l_idx
        self.dir_r_idx = dir_r_idx
        #pose_idxからpose_dimを計算
        pose_dim=0
        for s in pose_idx:
            pose_dim+=(s.stop-s.start)*s.step if s.step is not None else (s.stop-s.start)
        pose_dim*=3
        hand_dim=0
        for s in dir_l_idx:
            hand_dim+=(s.stop-s.start)*s.step if s.step is not None else (s.stop-s.start)
        hand_dim*=3
        self.loss_w = loss_w
        self.pose_encoder = VQVAETransformer1DEncoder(
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
            pos_idx=pose_idx,
            dir_idx=dir_l_idx + dir_r_idx,
        )
        self.pose_decoder = VQVAETransformer1DDecoder(
            in_dim=pose_dim,
            d_model=pose_d_model,
            n_heads=n_pose_heads,
            n_layers_enc=n_pose_layers_enc,
            n_layers_dec=n_pose_layers_dec,
            ff_mult=ff_mult,
            dropout=dropout,
            code_dim=code_dim,
            n_codes=n_pose_codes,
            stride=stride,
            rvq_stages=rvq_stages,
            vq_beta=vq_beta,
            pos_idx=pose_idx,
            dir_idx=dir_l_idx + dir_r_idx,
        )
        self.left_encoder = VQVAETransformer1DEncoder(
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
            pos_idx=dir_l_idx,
            dir_idx=(),
        )
        self.left_decoder = VQVAETransformer1DDecoder(
            in_dim=hand_dim,
            d_model=hand_d_model,
            n_heads=n_hand_heads,
            n_layers_enc=n_hand_layers_enc,
            n_layers_dec=n_hand_layers_dec,
            ff_mult=ff_mult,
            dropout=dropout,
            code_dim=code_dim,
            n_codes=n_hand_codes,
            stride=stride,
            rvq_stages=rvq_stages,
            pos_idx=dir_l_idx,
            dir_idx=(),
        )
        self.right_encoder = VQVAETransformer1DEncoder(
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
            pos_idx=dir_r_idx,
            dir_idx=(),
        )
        self.right_decoder = VQVAETransformer1DDecoder(
            in_dim=hand_dim,
            d_model=hand_d_model,
            n_heads=n_hand_heads,
            n_layers_enc=n_hand_layers_enc,
            n_layers_dec=n_hand_layers_dec,
            ff_mult=ff_mult,
            dropout=dropout,
            code_dim=code_dim,
            n_codes=n_hand_codes,
            stride=stride,
            rvq_stages=rvq_stages,
            pos_idx=dir_r_idx,
            dir_idx=(),
        )
        self.stride=stride
        # ---- Quantizer ----
        self.to_code=nn.Linear(pose_d_model+hand_d_model*2,code_dim)
        if rvq_stages <= 1:
            self.quant = VectorQuantizerWithRestart(n_codes=n_codes, code_dim=code_dim, beta=vq_beta)
            self.is_rvq = False
        else:
            self.quant = ResidualVectorQuantizer(n_codes=n_codes, code_dim=code_dim, stages=rvq_stages, beta=vq_beta)
            self.is_rvq = True

    def _gather_slices(self, x: torch.Tensor, slices: Tuple[slice, ...]) -> torch.Tensor:
        #x:(B,T,C,2or3)
        parts = [x[:,:,s] for s in slices]
        return torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]

    def forward(self, x: torch.Tensor, hand_valid_mask: Optional[torch.Tensor] = None,
                input_length: Optional[torch.tensor] = None) -> Dict[str, torch.Tensor]:
        # x: (B,T,36) -> split into pose/hand/extra
        B,T,J,C=x.shape
        pose = self._gather_slices(x, self.pose_idx).reshape(B,T,-1) # (B,T,pose_dim)
        l_hand = self._gather_slices(x, self.dir_l_idx).reshape(B,T,-1)  # (B,T,hand_dim)
        r_hand = self._gather_slices(x, self.dir_r_idx).reshape(B,T,-1)  # (B,T,hand_dim)
        padding_mask=create_mask(input_length,max_len=x.shape[1]).bool() if input_length is not None else None
        #attn_mask=create_slide_window_mask(x.shape[1],window_size=7,device=x.device) if input_length is not None else None
        attn_mask=None
        # extra = x[..., -self.extra_dim:]              # (B,T,extra_dim)
        h_pose=self.pose_encoder(pose,padding_mask=padding_mask,attn_mask=attn_mask)  # (B,K_p,pose_d_model)
        h_l_hand=self.left_encoder(l_hand,padding_mask=padding_mask,attn_mask=attn_mask)  # (B,K_l,hand_d_model)
        h_r_hand=self.right_encoder(r_hand,padding_mask=padding_mask,attn_mask=attn_mask)  #(B,K_r,hand_d_model)
        # ---- aggregate ----
        z_e=torch.cat([h_pose,h_l_hand,h_r_hand],dim=-1) #(B,K,pose_d_model+hand_d_model*2) where K is the max of K_p,K_l,K_r (due to downsampling)
        z_e=self.to_code(z_e)  #(B,K,code_dim)
        q = self.quant(z_e)  # dict with 'z_q':(B,K,code_dim), 'loss_vq':scalar, 'perplexity':scalar, 'codes':(B,K) or (B,stages,K) if RVQ
        z_q = q["z_q"]  # (B,K,code_dim)
        x_hat_pose = self.pose_decoder(z_q, T_out=pose.shape[1],padding_mask=padding_mask,attn_mask=attn_mask)  # (B,T,pose_dim)
        x_hat_l_hand = self.left_decoder(z_q, T_out=l_hand.shape[1],padding_mask=padding_mask,attn_mask=attn_mask)  # (B,T,hand_dim)
        x_hat_r_hand = self.right_decoder(z_q, T_out=r_hand.shape[1],padding_mask=padding_mask,attn_mask=attn_mask)  # (B,T,hand_dim)
        x_hat = torch.zeros_like(x)
        # scatter back to original positions
        #loss
        pose_recon_loss=F.mse_loss(x_hat_pose,pose,reduction='none')
        right_recon_loss=F.mse_loss(x_hat_l_hand,l_hand,reduction='none')
        left_recon_loss=F.mse_loss(x_hat_r_hand,r_hand,reduction='none')
        #if hand_valid_mask is not None:
        #    right_recon_loss=right_recon_loss*hand_valid_mask[...,0:1]
        #    left_recon_loss=left_recon_loss*hand_valid_mask[...,1:2]
        hand_recon_loss=right_recon_loss+left_recon_loss
        if input_length is not None:
            mask=create_mask(input_length,max_len=x.shape[1]).unsqueeze(-1) #(B,T,1)
            pose_recon_loss=(pose_recon_loss*mask).sum()/mask.sum()
            hand_recon_loss=(hand_recon_loss*mask).sum()/mask.sum()
        else:
            pose_recon_loss=pose_recon_loss.mean()
            hand_recon_loss=hand_recon_loss.mean()
        recon_loss=pose_recon_loss+hand_recon_loss
        vq_loss=q['loss_vq']
        total_loss=self.loss_w.recon_pos*recon_loss+self.loss_w.vq*vq_loss
        return {
            "loss_total": total_loss,
            "pose_recon_loss": recon_loss.detach(),  # for simplicity, not separating pose/hand recon loss here since they are combined in the aggregate
            "hand_recon_loss": torch.tensor(0.0, device=x.device),  # placeholder since recon_loss is combined
            "extra_recon_loss": torch.tensor(0.0, device=x.device),  # placeholder since extra is not used
            "pose_vq_loss": vq_loss.detach(),  # for simplicity, not separating
            "hand_vq_loss": torch.tensor(0.0, device=x.device),  # placeholder since vq_loss is combined
            "extra_vq_loss": torch.tensor(0.0, device=x.device),
            "codes": q["codes"],
            "extra_codes": torch.tensor(0.0, device=x.device),  # placeholder
            "pose_x_hat": x_hat_pose,
            "left_x_hat": x_hat_l_hand,
            "right_x_hat": x_hat_r_hand,
            "extra_x_hat": torch.tensor(0.0, device=x.device),  #
            "perplexity": q["perplexity"],
            'z_e': z_e.detach(),
            'dist': q.get('dist', torch.tensor(0.0, device=x.device)),
        }
    def random_restart(self,z_e,threshold=None):
        B,K,C=z_e.shape
        z=z_e.view(B*K,C)
        self.quant.random_restart(z,threshold=threshold)
class VQVAETransformer1DAggregatedCategorical(VQVAETransformer1DAggregated):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # override quantizer with categorical version
        pose_d_model=kwargs.get('pose_d_model', 256)
        hand_d_model=kwargs.get('hand_d_model', 128)
        code_dim=kwargs.get('code_dim', 128)
        n_codes=kwargs.get('n_codes', 1024)
        vq_beta=kwargs.get('vq_beta', 0.25)
        tau=kwargs.get('tau', 1.0)

        self.to_code=nn.Linear(pose_d_model+hand_d_model*2,code_dim)
        self.quant=VectorQuantizerCategorical(n_codes=n_codes, code_dim=code_dim, beta=vq_beta, tau=tau)
    def random_restart(self,z_e,threshold=None):
        pass
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