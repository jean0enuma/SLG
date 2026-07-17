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
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import math


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


class VectorQuantizerEMA(nn.Module):
    """
    標準的な EMA-VQ (L2正規化なし、ユークリッド空間)
    - codebook は EMA で更新(勾配なし)
    - commit loss のみ encoder への勾配
    - dead code restart 内蔵(forward内で自動発火)
    """
    def __init__(self, n_codes: int, code_dim: int, beta: float = 0.25,
                 decay: float = 0.99, eps: float = 1e-5,
                 restart_threshold: float = 0.5,
                 restart_every: int = 500):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.restart_threshold = restart_threshold
        self.restart_every = restart_every

        # 小さめのN(0, 0.02^2)で初期化(VQ-VAE標準)
        emb = torch.randn(n_codes, code_dim) * 0.02
        # codebook は buffer で管理(勾配なし)
        self.register_buffer('codebook_buf', emb.clone())
        self.register_buffer('ema_cluster_size', torch.zeros(n_codes))
        self.register_buffer('ema_w', emb.clone())
        self.register_buffer('usage_count', torch.zeros(n_codes))
        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))
        # restart 用に直近 z_e のサンプルを保持
        self.register_buffer('last_z_for_restart', torch.zeros(0, code_dim))

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, K, C = z_e.shape
        assert C == self.code_dim
        z = z_e.reshape(-1, C)  # (N, C)

        # ユークリッド距離
        z2 = (z ** 2).sum(dim=1, keepdim=True)             # (N,1)
        e2 = (self.codebook_buf ** 2).sum(dim=1).unsqueeze(0)  # (1,M)
        dist = z2 + e2 - 2 * (z @ self.codebook_buf.t())   # (N,M)
        idx = torch.argmin(dist, dim=1)                    # (N,)

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.n_codes).float()
                cluster_size = onehot.sum(0)  # (M,)

                # usage は restart 判定用、cluster_size は EMA 更新用
                self.usage_count.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

                dw = onehot.t() @ z  # (M, C)
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                smoothed = (self.ema_cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
                self.codebook_buf.copy_(self.ema_w / smoothed.unsqueeze(1))

                # restart は一定step毎に自動発火
                self.step_count += 1
                if self.step_count.item() % self.restart_every == 0:
                    self._restart_dead_codes(z)

                # 直近の z をrestart用に保持(少量だけ)
                if self.last_z_for_restart.numel() == 0:
                    sample_n = min(1024, z.size(0))
                    self.last_z_for_restart = z[:sample_n].detach().clone()

        # 量子化
        z_q = F.embedding(idx, self.codebook_buf).view(B, K, C)

        # commit loss(encoderを z_q に近づける)
        # EMA を使うのでcodebook lossは不要、commitのみ
        loss_commit = self.beta * F.mse_loss(z_q.detach(), z_e)

        # straight-through
        z_q_st = z_e + (z_q - z_e).detach()

        # perplexity (現バッチ)
        avg = F.one_hot(idx, self.n_codes).float().mean(dim=0)
        perplexity = torch.exp(-(avg * (avg + 1e-10).log()).sum())

        return {
            "z_q": z_q_st,
            "codes": idx.view(B, K),
            "loss_vq": loss_commit,
            "perplexity": perplexity,
        }

    @torch.no_grad()
    def _restart_dead_codes(self, z_flat: torch.Tensor):
        """usage_count が低いコードを z_flat の値で再初期化"""
        # threshold は EMA定常状態での平均使用量に対する相対値
        # 平均期待値: N / n_codes(N=batch tokens)
        avg_usage = self.usage_count.mean()
        thresh = self.restart_threshold * avg_usage  # 平均の50%以下なら死
        dead = torch.where(self.usage_count < thresh)[0]
        if len(dead) == 0:
            return

        n_resets = min(len(dead), z_flat.size(0))
        picks = torch.randperm(z_flat.size(0), device=z_flat.device)[:n_resets]
        replacement = z_flat[picks].to(dtype=self.codebook_buf.dtype)

        target = dead[:n_resets]
        # 少しノイズを足して同じコードに集中しないように
        noise = 0.01 * replacement.std() * torch.randn_like(replacement)
        self.codebook_buf[target] = replacement + noise
        self.ema_w[target] = self.codebook_buf[target].clone()
        # 復活したコードがすぐまた死なないように、平均使用量に等しい値を入れる
        self.ema_cluster_size[target] = avg_usage
        self.usage_count[target] = avg_usage

    @torch.no_grad()
    def random_restart(self, z_flat, threshold=None):
        """外部から呼ばれた場合のためのインターフェース互換"""
        self._restart_dead_codes(z_flat)
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


class VectorQuantizerEMA_Normalized(nn.Module):
    """encoder出力が既にL2正規化されている前提のEMA-VQ"""

    def __init__(self, n_codes, code_dim, beta=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        emb = F.normalize(torch.randn(n_codes, code_dim), p=2, dim=1)
        self.register_buffer('codebook', emb)
        self.register_buffer('ema_cluster_size', torch.zeros(n_codes))
        self.register_buffer('ema_w', emb.clone())
        self.register_buffer('usage_count', torch.zeros(n_codes))

    def forward(self, z_e):
        """z_e は既に F.normalize 済み (B,K,C)"""
        B, K, C = z_e.shape
        z = z_e.reshape(-1, C)  # 既にunit norm

        # 距離: cos距離
        dist = 2 - 2 * (z @ self.codebook.t())
        idx = torch.argmin(dist, dim=1)

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.n_codes).float()
                cluster_size = onehot.sum(0)

                # Usage update for restart
                self.usage_count.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

                # EMA update
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                dw = onehot.t() @ z
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                smoothed = (self.ema_cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
                new_codebook = self.ema_w / smoothed.unsqueeze(1)
                self.codebook.copy_(F.normalize(new_codebook, p=2, dim=1))

        # Quantize: codebook entry も unit norm
        z_q = F.embedding(idx, self.codebook).view(B, K, C)

        # commit loss: 両方unit norm なので bounded ([0,4]の範囲)
        loss_commit = self.beta * F.mse_loss(z_q.detach(), z_e)

        # Straight-through
        z_q_st = z_e + (z_q - z_e).detach()

        # Perplexity
        avg = F.one_hot(idx, self.n_codes).float().mean(0)
        perplexity = torch.exp(-(avg * (avg + 1e-10).log()).sum())

        return {
            'z_q': z_q_st,
            'codes': idx.view(B, K),
            'loss_vq': loss_commit,
            'perplexity': perplexity,
        }

    @torch.no_grad()
    def random_restart(self, z_flat, threshold=None):
        if threshold is None:
            threshold = self.restart_threshold if hasattr(self, 'restart_threshold') else 0.5
        # z_flat も既に正規化されている前提
        dead = torch.where(self.usage_count < threshold)[0]
        if len(dead) > 0:
            n_resets = min(len(dead), z_flat.size(0))
            picks = torch.randperm(z_flat.size(0), device=z_flat.device)[:n_resets]
            replacement = z_flat[picks]  # 既にunit norm
            target = dead[:n_resets]
            self.codebook[target] = F.normalize(replacement + 0.01 * torch.randn_like(replacement), p=2, dim=1)
            self.ema_w[target] = self.codebook[target].clone()
            self.ema_cluster_size[target] = threshold
            self.usage_count[target] = threshold


class VectorQuantizerEMA_Plain(nn.Module):
    def __init__(self, n_codes, code_dim, beta=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_codes, self.code_dim = n_codes, code_dim
        self.beta, self.decay, self.eps = beta, decay, eps

        emb = torch.randn(n_codes, code_dim) * 0.02  # 小さい初期値
        self.register_buffer('codebook', emb)
        self.register_buffer('ema_cluster_size', torch.zeros(n_codes))
        self.register_buffer('ema_w', emb.clone())
        self.register_buffer('usage_count', torch.zeros(n_codes))

    def forward(self, z_e):
        B, K, C = z_e.shape
        z = z_e.reshape(-1, C)

        # 標準的なユークリッド距離
        z2 = (z ** 2).sum(1, keepdim=True)
        e2 = (self.codebook ** 2).sum(1).unsqueeze(0)
        dist = z2 + e2 - 2 * (z @ self.codebook.t())
        idx = torch.argmin(dist, dim=1)

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.n_codes).float()
                cluster_size = onehot.sum(0)
                self.usage_count.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                dw = onehot.t() @ z
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                n = self.ema_cluster_size.sum()
                smoothed = (self.ema_cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
                self.codebook.copy_(self.ema_w / smoothed.unsqueeze(1))  # 正規化なし

        z_q = F.embedding(idx, self.codebook).view(B, K, C)
        loss_commit = self.beta * F.mse_loss(z_q.detach(), z_e)
        z_q_st = z_e + (z_q - z_e).detach()

        avg = F.one_hot(idx, self.n_codes).float().mean(0)
        perplexity = torch.exp(-(avg * (avg + 1e-10).log()).sum())

        return {'z_q': z_q_st, 'codes': idx.view(B, K), 'loss_vq': loss_commit, 'perplexity': perplexity}
class VectorQuantizerEMA_Sphere(nn.Module):
    """
    encoder出力が unit norm である前提のEMA-VQ。
    内部で再正規化しない、L2正規化空間で全て完結。
    """
    def __init__(self, n_codes, code_dim, beta=0.25, decay=0.99, eps=1e-5,
                 restart_threshold=0.5, restart_every=500):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.restart_threshold = restart_threshold
        self.restart_every = restart_every

        # unit normで初期化
        emb = F.normalize(torch.randn(n_codes, code_dim), p=2, dim=1)
        self.register_buffer('codebook_buf', emb.clone())
        self.register_buffer('ema_cluster_size', torch.zeros(n_codes))
        self.register_buffer('ema_w', emb.clone())
        self.register_buffer('usage_count', torch.zeros(n_codes))
        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))

    def forward(self, z_e):
        """z_e: (B, K, C) で **既に L2 正規化されている前提** """
        B, K, C = z_e.shape
        z = z_e.reshape(-1, C)

        # コサイン距離(両方unit norm なので 2 - 2*z@e^T)
        dist = 2 - 2 * (z @ self.codebook_buf.t())
        idx = torch.argmin(dist, dim=1)

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.n_codes).float()
                cluster_size = onehot.sum(0)
                self.usage_count.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

                dw = onehot.t() @ z  # 既にunit norm の z で更新
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                n = self.ema_cluster_size.sum()
                smoothed = (self.ema_cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
                # 更新後に再正規化(unit sphere上に保つ)
                new_codebook = self.ema_w / smoothed.unsqueeze(1)
                self.codebook_buf.copy_(F.normalize(new_codebook, p=2, dim=1))

                self.step_count += 1
                if self.step_count.item() % self.restart_every == 0:
                    self._restart_dead_codes(z)

        z_q = F.embedding(idx, self.codebook_buf).view(B, K, C)

        # commit loss: 両方unit norm なので bounded
        # 値域は [0, 4]、典型的には学習が進むと 0 に収束
        loss_commit = self.beta * F.mse_loss(z_q.detach(), z_e)

        z_q_st = z_e + (z_q - z_e).detach()

        avg = F.one_hot(idx, self.n_codes).float().mean(dim=0)
        perplexity = torch.exp(-(avg * (avg + 1e-10).log()).sum())

        return {
            "z_q": z_q_st,
            "codes": idx.view(B, K),
            "loss_vq": loss_commit,
            "perplexity": perplexity,
        }

    @torch.no_grad()
    def _restart_dead_codes(self, z_flat):
        avg_usage = self.usage_count.mean()
        thresh = self.restart_threshold * avg_usage
        dead = torch.where(self.usage_count < thresh)[0]
        if len(dead) == 0:
            return
        n_resets = min(len(dead), z_flat.size(0))
        picks = torch.randperm(z_flat.size(0), device=z_flat.device)[:n_resets]
        replacement = z_flat[picks]  # 既にunit norm
        target = dead[:n_resets]
        # 少しノイズを足して再正規化
        noise = 0.01 * torch.randn_like(replacement)
        new_codes = F.normalize(replacement + noise, p=2, dim=1)
        self.codebook_buf[target] = new_codes
        self.ema_w[target] = new_codes.clone()
        self.ema_cluster_size[target] = avg_usage
        self.usage_count[target] = avg_usage

    @torch.no_grad()
    def random_restart(self, z_flat, threshold=None):
        # z_flat も呼び出し側で正規化されている想定
        self._restart_dead_codes(z_flat)


class VectorQuantizerWithRestart(nn.Module):
    def __init__(self, n_codes: int, code_dim: int, beta: float = 0.25,
                 restart_threshold: float = 1.0, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta
        self.restart_threshold = restart_threshold

        # EMA用のパラメータ
        self.decay = decay
        self.eps = eps

        self.codebook = nn.Embedding(n_codes, code_dim)
        # L2Normを利用するため、初期値も正規化しておく
        nn.init.normal_(self.codebook.weight)
        self.codebook.weight.data = F.normalize(self.codebook.weight.data, p=2, dim=1)

        # EMA更新用のバッファ
        self.register_buffer('usage_count', torch.zeros(n_codes))  # Restart判定用
        self.register_buffer('ema_cluster_size', torch.zeros(n_codes))
        self.register_buffer('ema_w', self.codebook.weight.data.clone())

        # --- 累積統計用バッファ(新規)---
        # usage_count は EMA で意味が違うので別名にする
        self.register_buffer('cumulative_usage', torch.zeros(n_codes))
        self.register_buffer('total_tokens', torch.zeros(1, dtype=torch.long))

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, K, C = z_e.shape
        z_flattened = z_e.reshape(-1, C)  # (N, C)

        # --- 1. L2 Normalization (Input & Codebook) ---
        # 単位球面上に投影することで、コサイン類似度に基づいた距離計算にする
        z_norm = F.normalize(z_flattened, p=2, dim=1)
        e_norm = F.normalize(self.codebook.weight, p=2, dim=1)

        # --- 2. 距離計算 (L2正規化済みなので内積で計算可能) ---
        # dist = ||z||^2 + ||e||^2 - 2*z·e  (正規化済みなので ||z||=1, ||e||=1)
        dist = 2 - 2 * (z_norm @ e_norm.t())
        idx = torch.argmin(dist, dim=1)

        # --- 3. EMA による統計情報の更新 (Training時のみ) ---
        if self.training:
            # One-hot作成
            encodings = F.one_hot(idx, num_classes=self.n_codes).float().detach()  # (N, n_codes)

            # 使用頻度の更新 (Restart判定用)
            current_usage = encodings.sum(0)
            self.usage_count.mul_(self.decay).add_(current_usage, alpha=1 - self.decay)

            # EMA更新: クラスタサイズと重みの更新
            self.ema_cluster_size.mul_(self.decay).add_(current_usage, alpha=1 - self.decay)

            # 各コードに割り当てられた入力ベクトルの総和
            dw = encodings.t() @ z_norm.detach()  # 正規化後のベクトルで更新
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # ラプラススムージングを用いたコードブックの更新
            n = self.ema_cluster_size.sum()
            smoothed_cluster_size = (
                    (self.ema_cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
            )
            updated_w = self.ema_w / smoothed_cluster_size.unsqueeze(1)

            # 常にL2正規化を維持して反映
            # 修正後（より安全な書き方）
            with torch.no_grad():
                normalized_w = F.normalize(updated_w, p=2, dim=1)
                self.codebook.weight.copy_(normalized_w.detach())
        # 累積統計の更新(train/eval 両方で実行)
        self._update_usage(idx.view(B, K))
        # --- 4. 量子化とLoss計算 ---
        z_q = self.codebook(idx).view(B, K, C)

        # Commitment Loss (z_eをz_qに近づける)
        # ※EMAを使う場合、Codebook Lossは不要になるためbeta * commitのみ
        loss_vq = self.beta * F.mse_loss(z_q.detach(), z_e)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Perplexity
        avg_probs = torch.bincount(idx, minlength=self.n_codes).float() / idx.shape[0]
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            "z_q": z_q_st,
            "codes": idx.view(B, K),
            "loss_vq": loss_vq,
            "perplexity": perplexity,
        }
    def indices_to_codes(self,indices: torch.Tensor) -> torch.Tensor:
        """整数インデックスからコードベクトルを取得するユーティリティ"""
        return self.codebook(indices)
    # ----------------------------------------------------------------
    # 累積 code usage 統計
    # ----------------------------------------------------------------
    @torch.no_grad()
    def _update_usage(self, indices: torch.Tensor):
        """
        累積使用量を更新する内部メソッド。
        indices: (B, K) 整数インデックス
        ※ VQ では code が不可分なベクトルなので per-dim 統計は持たない。
        """
        flat_indices = indices.reshape(-1).long()
        batch_count = torch.bincount(flat_indices, minlength=self.n_codes)
        self.cumulative_usage += batch_count.to(self.cumulative_usage.dtype)
        self.total_tokens += flat_indices.numel()

    @torch.no_grad()
    def reset_usage(self):
        """累積使用量をリセット (epoch 開始時などに呼ぶ)"""
        self.cumulative_usage.zero_()
        self.total_tokens.zero_()

    @torch.no_grad()
    def get_code_usage(self, normalize: bool = True) -> Dict[str, torch.Tensor]:
        """
        累積した code usage 統計を返す。
        """
        counts = self.cumulative_usage.float()
        total = counts.sum().clamp_min(1.0)
        probs = counts / total

        eps = 1e-10

        # 使われたコード数 / 比率
        used_mask = counts > 0
        used_codes = used_mask.sum()
        used_ratio = used_codes.float() / self.n_codes

        # 有意に使われたコード(平均カウントの 1% 以上)
        mean_count = counts.mean()
        effective_codes = (counts >= mean_count * 0.01).sum()

        # エントロピー / Perplexity (nat 単位)
        entropy = -(probs * torch.log(probs + eps)).sum()
        perplexity = torch.exp(entropy)

        # Top-k concentration
        sorted_probs, _ = torch.sort(probs, descending=True)
        top_k_concentration = {}
        for k in (1, 5, 10, 50, 100):
            if k <= self.n_codes:
                top_k_concentration[k] = sorted_probs[:k].sum().item()

        # Gini 係数 (0=均等, 1=完全集中)
        sorted_counts, _ = torch.sort(counts)
        n = sorted_counts.numel()
        cum = torch.cumsum(sorted_counts, dim=0)
        denom = cum[-1].clamp_min(eps)
        gini = (n + 1 - 2.0 * cum.sum() / denom) / n

        return {
            'counts': probs if normalize else counts,
            'probs': probs,
            'used_codes': used_codes,
            'used_ratio': used_ratio,
            'effective_codes': effective_codes,
            'perplexity': perplexity,
            'top_k_concentration': top_k_concentration,
            'gini': gini,
            'entropy': entropy,
            'total_tokens': self.total_tokens.clone(),
        }
    @torch.no_grad()
    def random_restart(self, z_flat, threshold=None):
        if threshold is None:
            threshold = self.restart_threshold

        # 使用率が低いインデックスを特定
        dead_indices = torch.where(self.usage_count < threshold)[0]

        if len(dead_indices) > 0:
            n_resets = min(len(dead_indices), z_flat.size(0))
            random_indices = torch.randperm(z_flat.size(0), device=z_flat.device)[:n_resets]

            # 入力も正規化してリセットに使う
            replacement = F.normalize(z_flat[random_indices], p=2, dim=1).to(dtype=self.codebook.weight.dtype)

            # コードブックおよびEMAバッファのリセット
            target_indices = dead_indices[:n_resets]
            self.codebook.weight.data[target_indices] = replacement + torch.randn_like(replacement) * 0.01
            self.ema_w[target_indices] = self.codebook.weight.data[target_indices].clone()
            self.ema_cluster_size[target_indices] = threshold  # すぐに消えないよう初期化
            self.usage_count[target_indices] = threshold

    @torch.no_grad()
    def usage_summary(self, verbose: bool = True) -> str:
        """
        人間が読める形式で usage 統計をサマリー化。
        verbose=True なら詳細(上位/下位コード、EMA状態など)を含める。
        """
        import math

        u = self.get_code_usage()
        counts = self.cumulative_usage.float()

        max_entropy = math.log(self.n_codes) if self.n_codes > 1 else 1.0

        lines = []
        lines.append("=== VQ Code Usage Summary ===")
        lines.append(f"Total tokens processed: {int(self.total_tokens.item()):,}")
        lines.append(f"Codebook size (n_codes): {self.n_codes}")
        lines.append(f"Code dim:               {self.code_dim}")

        # 使われたコード
        lines.append(
            f"Used codes:        {int(u['used_codes'].item())}/{self.n_codes} "
            f"({u['used_ratio'].item() * 100:.1f}%)"
        )
        lines.append(
            f"Effective codes:   {int(u['effective_codes'].item())}/{self.n_codes} "
            f"({u['effective_codes'].item() / self.n_codes * 100:.1f}%)  [>=1% of mean count]"
        )

        # Perplexity / Entropy
        lines.append(
            f"Cumulative perplexity: {u['perplexity'].item():.2f}  "
            f"(max possible: {self.n_codes})"
        )
        lines.append(
            f"Entropy:               {u['entropy'].item():.4f} nat  "
            f"(max: {max_entropy:.4f}, {u['entropy'].item() / max_entropy * 100:.1f}%)"
        )
        lines.append(
            f"Gini coefficient:      {u['gini'].item():.4f}  (0=uniform, 1=concentrated)"
        )

        # 上位 k 集中度
        lines.append("\nConcentration (top-k cumulative probability):")
        for k, ratio in u['top_k_concentration'].items():
            lines.append(f"  top-{k:>3}: {ratio * 100:6.2f}%")

        if verbose:
            # 上位/下位のコードを覗き見
            probs = u['probs']
            k_show = min(10, self.n_codes)

            top_vals, top_idx = torch.topk(probs, k_show, largest=True)
            bot_vals, bot_idx = torch.topk(probs, k_show, largest=False)

            lines.append(f"\nTop {k_show} codes (idx: prob):")
            lines.append(
                "  " + ", ".join(
                    f"{i.item()}:{v.item() * 100:.2f}%"
                    for i, v in zip(top_idx, top_vals)
                )
            )
            lines.append(f"Bottom {k_show} codes (idx: count):")
            lines.append(
                "  " + ", ".join(
                    f"{i.item()}:{int(counts[i].item())}"
                    for i in bot_idx
                )
            )

            # Restart 関連の状態(EMA usage_count)
            dead = (self.usage_count < self.restart_threshold).sum().item()
            lines.append(
                f"\nRestart status (EMA-based):"
            )
            lines.append(
                f"  dead codes (ema_usage < {self.restart_threshold}): "
                f"{dead}/{self.n_codes} ({dead / self.n_codes * 100:.1f}%)"
            )
            lines.append(
                f"  ema usage  min/mean/max: "
                f"{self.usage_count.min().item():.3f} / "
                f"{self.usage_count.mean().item():.3f} / "
                f"{self.usage_count.max().item():.3f}"
            )

        return "\n".join(lines)
# -------------------------
# Loss weights
# -------------------------
@dataclass
class VQLossWeights:
    recon_pos: float = 1.0
    recon_hand: float = 1.0
    recon_face: float = 1.0
    vq: float = 1.0


class FSQ(nn.Module):
    """
    Finite Scalar Quantization (Mentzer et al., 2023)

    Args:
        levels: 各次元のレベル数のリスト, e.g., [8, 5, 5, 5]
                総コードブックサイズ = prod(levels)
                code_dim = len(levels)
    """

    def __init__(self, levels: List[int],input_scale: float = 3.0):
        super().__init__()
        self.levels = levels
        self.code_dim = len(levels)
        self.n_codes = 1
        for l in levels:
            self.n_codes *= l

        # バッファとして保持
        _levels = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer('_levels', _levels)
        self.input_scale = input_scale

        # half_l = (L_i - 1) / 2
        self.register_buffer('_half_l', (_levels - 1) / 2)

        # 偶数レベルの対称化用 offset
        _offset = torch.where(_levels % 2 == 0,
                              torch.tensor(0.5),
                              torch.tensor(0.0))
        self.register_buffer('_offset', _offset)

        # tanh の shift (atanh(offset / half_l))
        # half_l == 0 の防止(レベル1は使わない前提)
        eps = 1e-8
        shift_arg = (_offset / (self._half_l + eps)).clamp(-0.9999, 0.9999)
        self.register_buffer('_shift', torch.atanh(shift_arg))

        # 各次元の basis (cumprod)
        _basis = torch.tensor([1] + levels[:-1], dtype=torch.long).cumprod(dim=0)
        self.register_buffer('_basis', _basis)

        # ─────────────────────────────────────
        # Code usage 累積カウント用バッファ
        # ─────────────────────────────────────
        # 全コード(combined index)の使用頻度
        self.register_buffer('usage_count', torch.zeros(self.n_codes, dtype=torch.long))
        # 各次元のレベル使用頻度(L_i ごとに分離して管理)
        # 形状は (code_dim, max(levels)) で、未使用の領域は無視する
        max_level = max(levels)
        self.register_buffer('per_dim_usage', torch.zeros(self.code_dim, max_level, dtype=torch.long))
        # 累積で処理したトークン数(平均などの計算に使用)
        self.register_buffer('total_tokens', torch.zeros(1, dtype=torch.long))
    def _bound(self, z):
        # スケールを掛けてから tanh
        return (z * self.input_scale + self._shift).tanh() * self._half_l - self._offset

    def quantize(self, z):
        """量子化(STE 付き)"""
        z_bounded = self._bound(z)
        z_rounded = z_bounded.round()
        # Straight-through estimator
        return z_bounded + (z_rounded - z_bounded).detach()

    def codes_to_indices(self, codes):
        """codes: (..., D) 量子化後の値 → indices: (...,) 整数"""
        # codes は [-half_l - offset, half_l - offset] にある
        # 各次元を [0, L_i - 1] に正規化
        codes_normalized = (codes + self._half_l + self._offset).round().long()
        indices = (codes_normalized * self._basis).sum(dim=-1)
        return indices

    def indices_to_codes(self, indices):
        """indices: (...,) → codes: (..., D)"""
        codes_normalized = (indices.unsqueeze(-1) // self._basis) % self._levels.long()
        codes = codes_normalized.float() - self._half_l - self._offset
        return codes

    def forward(self, z_e):
        """
        z_e: (B, K, D)  encoder出力
        D は self.code_dim と一致しないといけない
        """
        assert z_e.size(-1) == self.code_dim, \
            f"input dim {z_e.size(-1)} != code_dim {self.code_dim}"

        # 量子化
        z_q = self.quantize(z_e)

        with torch.no_grad():
            indices = self.codes_to_indices(z_q)

            # 累積使用量を更新(training/eval問わず常に記録)
            self._update_usage(z_q, indices)

        # Perplexity 計算(参考値)
        with torch.no_grad():
            flat_indices = indices.reshape(-1)
            unique, counts = torch.unique(flat_indices, return_counts=True)
            probs = counts.float() / counts.sum()
            perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())

        return {
            'z_q': z_q,
            'codes': indices,
            'loss_vq': torch.tensor(0.0, device=z_e.device),  # FSQ は VQ loss なし
            'perplexity': perplexity,
        }

    @torch.no_grad()
    def random_restart(self, z_flat, threshold=None):
        """互換性のために空実装(FSQ は restart 不要)"""
        pass


    @torch.no_grad()
    def _update_usage(self, z_q: torch.Tensor, indices: torch.Tensor):
        """
        累積使用量を更新する内部メソッド。
        z_q: (B, K, D) 量子化値
        indices: (B, K) 整数インデックス
        """
        # combined index の使用頻度
        flat_indices = indices.reshape(-1).long()
        # bincount は minlength を渡せば 0埋めしてくれる
        batch_count = torch.bincount(flat_indices, minlength=self.n_codes)
        self.usage_count += batch_count.to(self.usage_count.dtype)

        # 各次元ごとのレベル使用頻度
        # z_q を [0, L_i - 1] の整数に変換
        codes_normalized = (z_q + self._half_l + self._offset).round().long()  # (B, K, D)
        for d in range(self.code_dim):
            level_count = torch.bincount(
                codes_normalized[..., d].reshape(-1),
                minlength=int(self._levels[d].item())
            )
            self.per_dim_usage[d, :level_count.size(0)] += level_count.to(self.per_dim_usage.dtype)

        self.total_tokens += flat_indices.numel()


    @torch.no_grad()
    def reset_usage(self):
        """累積使用量をリセット(epoch開始時などに呼ぶ)"""
        self.usage_count.zero_()
        self.per_dim_usage.zero_()
        self.total_tokens.zero_()


    @torch.no_grad()
    def get_code_usage(self, normalize: bool = True) -> Dict[str, torch.Tensor]:
        """
        累積した code usage 統計を返す。

        Args:
            normalize: True なら確率に正規化、False なら生のカウント

        Returns:
            dict:
                'counts': (n_codes,) 各コードの累積使用回数
                'probs': (n_codes,) 各コードの使用確率
                'used_codes': スカラー、少なくとも1回使われたコード数
                'used_ratio': スカラー、used_codes / n_codes
                'effective_codes': スカラー、有意に使われたコード数(平均の1%以上)
                'perplexity': スカラー、累積分布のperplexity
                'top_k_concentration': dict(k -> 上位k個のコードが占める割合)
                'gini': スカラー、Gini係数(0=均等, 1=完全集中)
                'entropy': スカラー、累積分布のエントロピー(nat)
                'total_tokens': スカラー、これまでに処理した総トークン数
        """
        counts = self.usage_count.float()
        total = counts.sum().clamp_min(1)
        probs = counts / total

        # 使用されたコードの数
        used_codes = (counts > 0).sum()
        used_ratio = used_codes.float() / self.n_codes

        # 有意に使われたコード(平均の1%以上)
        mean_count = counts.mean().clamp_min(1)
        effective_codes = (counts > 0.01 * mean_count).sum()

        # Perplexity (累積分布)
        log_probs = (probs + 1e-10).log()
        entropy = -(probs * log_probs).sum()
        perplexity = entropy.exp()

        # 上位k個の集中度
        sorted_counts, _ = counts.sort(descending=True)
        cumsum = sorted_counts.cumsum(0)
        top_k_concentration = {}
        for k in [1, 10, 100, max(1, self.n_codes // 10)]:
            if k <= self.n_codes:
                top_k_concentration[f'top_{k}'] = (cumsum[k - 1] / total).item()

        # Gini 係数
        # G = 1 - 2*∫F(x)dx / max あるいは並べ替えた累積から計算
        sorted_probs, _ = probs.sort()
        n = self.n_codes
        idx = torch.arange(1, n + 1, device=counts.device, dtype=torch.float32)
        gini = (2 * (idx * sorted_probs).sum() / sorted_probs.sum().clamp_min(1e-10) - (n + 1)) / n

        result = {
            'counts': counts if not normalize else probs,
            'probs': probs,
            'used_codes': used_codes,
            'used_ratio': used_ratio,
            'effective_codes': effective_codes,
            'perplexity': perplexity,
            'entropy': entropy,
            'top_k_concentration': top_k_concentration,
            'gini': gini,
            'total_tokens': self.total_tokens.clone(),
        }
        return result


    @torch.no_grad()
    def get_per_dim_usage(self) -> Dict[str, torch.Tensor]:
        """
        各次元ごとのレベル使用頻度を返す。

        Returns:
            dict:
                'counts_per_dim': List of (L_i,) tensors, 各次元の使用カウント
                'probs_per_dim': List of (L_i,) tensors, 各次元の使用確率
                'entropy_per_dim': (D,) 各次元のエントロピー(nat)
                'perplexity_per_dim': (D,) 各次元のperplexity
                'used_levels_per_dim': (D,) 各次元で1回以上使われたレベル数
                'utilization_per_dim': (D,) used_levels / L_i の比率
        """
        counts_list = []
        probs_list = []
        entropy_list = []
        perplexity_list = []
        used_levels = []
        utilization = []

        for d in range(self.code_dim):
            L = int(self._levels[d].item())
            cnt = self.per_dim_usage[d, :L].float()  # 有効領域のみ取り出す
            total = cnt.sum().clamp_min(1)
            p = cnt / total

            log_p = (p + 1e-10).log()
            ent = -(p * log_p).sum()
            ppx = ent.exp()

            used = (cnt > 0).sum()

            counts_list.append(cnt)
            probs_list.append(p)
            entropy_list.append(ent)
            perplexity_list.append(ppx)
            used_levels.append(used)
            utilization.append(used.float() / L)

        return {
            'counts_per_dim': counts_list,
            'probs_per_dim': probs_list,
            'entropy_per_dim': torch.stack(entropy_list),
            'perplexity_per_dim': torch.stack(perplexity_list),
            'used_levels_per_dim': torch.stack(used_levels),
            'utilization_per_dim': torch.stack(utilization),
        }


    @torch.no_grad()
    def usage_summary(self, verbose: bool = True) -> str:
        """
        人間が読める形式で usage 統計をサマリー化。
        verbose=True なら詳細を含めた文字列を返す。
        """
        u = self.get_code_usage()
        pd = self.get_per_dim_usage()

        lines = []
        lines.append(f"=== FSQ Code Usage Summary ===")
        lines.append(f"Total tokens processed: {u['total_tokens'].item():,}")
        lines.append(f"Codebook size (n_codes): {self.n_codes}")
        lines.append(f"Used codes: {u['used_codes'].item()}/{self.n_codes} "
                     f"({u['used_ratio'].item() * 100:.1f}%)")
        lines.append(f"Effective codes (>1% of mean): {u['effective_codes'].item()}/{self.n_codes} "
                     f"({u['effective_codes'].item() / self.n_codes * 100:.1f}%)")
        lines.append(f"Cumulative perplexity: {u['perplexity'].item():.2f} "
                     f"(max possible: {self.n_codes})")
        lines.append(f"Entropy: {u['entropy'].item():.4f} nat "
                     f"(max: {math.log(self.n_codes):.4f})")
        lines.append(f"Gini coefficient: {u['gini'].item():.4f} (0=uniform, 1=concentrated)")

        lines.append(f"\nConcentration (top-k cumulative probability):")
        for k_str, ratio in u['top_k_concentration'].items():
            lines.append(f"  {k_str}: {ratio * 100:.2f}%")

        if verbose:
            lines.append(f"\nPer-dimension usage (levels={self.levels}):")
            for d in range(self.code_dim):
                L = self.levels[d]
                ppx = pd['perplexity_per_dim'][d].item()
                util = pd['utilization_per_dim'][d].item()
                used = int(pd['used_levels_per_dim'][d].item())
                lines.append(f"  dim {d} (L={L}): used {used}/{L} levels, "
                             f"perplexity={ppx:.2f}/{L}, utilization={util * 100:.1f}%")

        return '\n'.join(lines)


    @torch.no_grad()
    def random_restart(self, z_flat, threshold=None):
        """互換性のために空実装(FSQ は restart 不要)"""
        pass


class IndependentToCode(nn.Module):
    """各出力次元を独立した経路から計算"""

    def __init__(self, in_dim, code_dim, hidden_factor=2):
        super().__init__()
        self.code_dim = code_dim
        # 各次元ごとに小さなMLP
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim // hidden_factor),
                nn.GELU(),
                nn.Linear(in_dim // hidden_factor, 1),
            )
            for _ in range(code_dim)
        ])

    def forward(self, x):
        # x: (..., in_dim)
        outs = [head(x) for head in self.heads]
        return torch.cat(outs, dim=-1)  # (..., code_dim)

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
        in_dim: int = 50,
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
        levels: List[int] = [5,3,2,2],  # FSQ用のレベル設定
        loss_w: VQLossWeights = VQLossWeights(),
        # coarse split indices (same meaning as in previous code)
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.in_dim = in_dim
        self.d_model = d_model
        self.code_dim = code_dim
        self.n_codes = n_codes
        self.stride = stride
        self.loss_w = loss_w

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
        self.encoder=nn.TransformerEncoder(enc_layer, num_layers=n_layers_enc)
        self.down=nn.ModuleList()
        if stride>1:
            for i in range(int(math.log2(stride))):
                self.down.append(nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1))

        # ---- Quantizer ----
        if rvq_stages <= 1:

            fsq_levels=levels  # 例: code_dim=4で総コード数=8*5*5*5=1000
            code_dim=len(fsq_levels)  # FSQのコード次元はレベル数の長さ
            self.code_dim=code_dim  # 上書きしてFSQ用に合わせる
            self.quant = FSQ(levels=fsq_levels)  # FSQを使用
            self.n_codes=self.quant.n_codes  # FSQの総コード数を反映

            #self.quant=VectorQuantizerWithRestart(n_codes=n_codes, code_dim=code_dim, beta=vq_beta)
            self.is_rvq = False
        else:
            self.quant = ResidualVectorQuantizer(n_codes=n_codes, code_dim=code_dim, stages=rvq_stages, beta=vq_beta)
            self.is_rvq = True

        self.to_code = nn.Sequential(
            nn.Linear(d_model, code_dim),
                nn.LayerNorm(code_dim)  # 正規化を追加して安定化を
        )
        # ---- Decoder ----
        self.from_code = nn.Linear(code_dim, d_model)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder=nn.TransformerEncoder(dec_layer, num_layers=n_layers_dec)
        self.up=nn.ModuleList()
        if stride > 1:
            for i in range(int(math.log2(stride))):
                self.up.append(nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1, output_padding=0))

        self.dec_ln_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, in_dim)



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

    # ---------- encode/decode ----------
    def encode(self, x: torch.Tensor,src_mask=None) -> torch.Tensor:
        """
        x: (B,T,in_dim)
        returns z_e: (B,K,code_dim)
        """
        B, T, _ = x.shape
        h = self.in_proj(x)  # (B,T,d_model)
        pe = self.pos_emb(T, device=h.device, dtype=h.dtype)  # (1,T,d_model)
        h = h + pe
        h=self.encoder(h,mask=src_mask).transpose(1,2).contiguous()  # (B,T,d_model)
        for down in  self.down:
            h=down(h) # (B,T//stride,d_model) or (B,T,d_model) if stride=1

        # downsample
        z_e = self.to_code(h.transpose(1,2))              # (B,K,code_dim)
        #z_e=F.normalize(z_e, p=2, dim=-1)  # RVQ-Sphere用に正規化
        return z_e

    def decode(self, z_q: torch.Tensor, T_out: int,src_mask=None) -> torch.Tensor:
        """
        z_q: (B,K,code_dim)
        returns x_hat: (B,T_out,in_dim)
        """
        h = self.from_code(z_q)  # (B,K,d_model)
        h= self.decoder(h, mask=src_mask).transpose(1,2).contiguous()  # (B,K,d_model)
        # upsample
        for up in  self.up:
            h = up(h)

        h_up=self._safe_crop_or_pad_time(h.transpose(1,2), T_out)  # (B,T,d_model)
        h_up = self.dec_ln_out(h_up)
        x_hat = self.out_proj(h_up)
        return x_hat
    @torch.no_grad()
    def tokenize(self,x:torch.Tensor,src_mask=None)->torch.Tensor:
        z_e=self.encode(x,src_mask=src_mask)
        q=self.quant(z_e)
        return q["codes"]
    @torch.no_grad()
    def reconstruct(self,codes,T_out:int,src_mask=None)->torch.Tensor:
        if self.is_rvq:
            B,S,K=codes.shape
            z_q_sum=torch.zeros(B,self.code_dim,device=codes.device)
            for s in range(S):
                z_q_sum=z_q_sum+self.quant.vqs[s].codebook(codes[:,s,:])
        else:
            z_q = F.embedding(codes,self.quant.codebook_buf)  # (B, K, code_dim)
        x_hat=self.decode(z_q,T_out=T_out,src_mask=src_mask)
        return x_hat

    def soft_repetition_penalty(self,z_e,temperature=0.1)->torch.Tensor:
        """encoder出力 z_e に対する soft prob分布のentropy促進"""
        # z_e: (B, K, C), codebook: (M, C)
        codebook=self.quant.codebook.weight  # (M, code_dim)
        z_norm=F.normalize(z_e, p=2, dim=-1)  # (B, K, C)
        codebook_norm=F.normalize(codebook, p=2, dim=-1)
        dist = 2 - 2 * (z_norm @ codebook_norm.t())  # (B, K, M)
        probs = F.softmax(-dist / temperature, dim=-1)  # (B, K, M)
        # 隣接フレームの prob 分布のcosineが高すぎたらペナルティ
        cos_sim = (probs[:, 1:] * probs[:, :-1]).sum(-1)  # (B, K-1)
        return cos_sim.mean()

    @torch.no_grad()
    def kmeans_init_codebook(self, dataloader, n_samples=10000):
        """学習開始前に呼ぶ"""
        self.eval()
        z_collected = []
        for batch in dataloader:
            padded_cod_data,padded_mask, input_length_tensor, id_list,data_path=batch
            B, T, J, C = padded_cod_data.shape
            x =padded_cod_data.to(next(self.parameters()).device)
            x=x.reshape(B, T, J * C)
            z_e = self.encode(x)  # (B, K, C)
            z_collected.append(z_e.reshape(-1, z_e.size(-1)))
            if sum(z.size(0) for z in z_collected) >= n_samples:
                break
        z_all = torch.cat(z_collected, dim=0)[:n_samples]

        # k-meansでクラスタリング
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=self.quant.n_codes, batch_size=2048, n_init=3)
        km.fit(z_all.cpu().numpy())

        # codebookに代入
        centroids = torch.from_numpy(km.cluster_centers_).to(z_all.device).to(z_all.dtype)
        self.quant.codebook.weight.copy_(centroids)
        self.quant.ema_w.copy_(centroids)
        self.train()
    # ---------- forward & losses ----------
    def forward(
        self,
        x: torch.Tensor,  # (B,T,J,C)
        *,
        hand_valid_mask: Optional[torch.Tensor] = None,  # (B,T,2) bool
        input_length: Optional[int] = None,  # if provided, will crop/pad input to this length
        return_recon: bool = True,
        no_return_loss: bool = False,  # if True, skip loss computation and return empty dict (for inference)
    ) -> Dict[str, torch.Tensor]:
        B, T, J,C = x.shape
        assert J*C== self.in_dim
        x=x.reshape(B,T,J*C)
        #src_mask=create_slide_window_mask(T,window_size=9,device=x.device) if input_length is not None else None
        src_mask=None
        z_e = self.encode(x,src_mask=src_mask)              # (B,K,code_dim)
        if input_length is not None:
            mask=create_mask(input_length,max_len=T) #(B,T,1)
        else:
            mask=torch.ones(B,T,device=x.device)

        z_e = z_e
        q = self.quant(z_e)                # dict
        z_q = q["z_q"]                     # (B,K,code_dim)
        x_hat = self.decode(z_q, T_out=T,src_mask=src_mask)  # (B,T,36)
        if no_return_loss:
            return {"x_hat": x_hat, "codes": q["codes"],"perplexity": q["perplexity"].detach(), "loss_vq": q["loss_vq"]}

        # ---- losses: pos (L1/Huber), dir (cos), vel (small), vq ----
        x=x.reshape(B,T,J,C)
        x_hat=x_hat.reshape(B,T,J,C)
        x_pos = x[:,:,:8]# (B,T,8,C)
        h_pos = x_hat[:,:,:8]# (B,T,8,C)
        loss_recon_pos = F.smooth_l1_loss(h_pos, x_pos.detach(), reduction='none').mean(dim=[-1,-2])  # (B,T)

        x_hand= x[:,:,8:50]
        h_hand= x_hat[:,:,8:50]# (B,T,42)
        loss_recon_hand=F.smooth_l1_loss(h_hand,x_hand.detach(),reduction='none').mean(dim=[-1,-2])   # (B,T)
        #loss_recon_face=F.smooth_l1_loss(x_hat[:,:,50:],x[:,:,50:].detach(),reduction='none').mean(dim=[-1,-2])  # (B,T)
        mask=~(mask.bool())
        loss_recon_pos=(loss_recon_pos*mask).sum()/mask.sum()
        loss_recon_hand=(loss_recon_hand*mask).sum()/mask.sum()
        #loss_recon_face=(loss_recon_face*mask).sum()/mask.sum()


        loss_vq = q["loss_vq"]

        loss_total = (
            self.loss_w.recon_pos * loss_recon_pos
                + self.loss_w.recon_hand * loss_recon_hand
                #+ self.loss_w.recon_face * loss_recon_face
            + self.loss_w.vq * loss_vq
        )

        out = {
            "loss_total": loss_total,
            "loss_recon_pos": loss_recon_pos.detach(),
            'loss_recon_hand': loss_recon_hand.detach(),
            "loss_vq": loss_vq.detach(),
            "perplexity": q["perplexity"].detach(),
            "codes": q["codes"].detach(),   # (B,K) for VQ, (B,stages,K) for RVQ
            'z_e': z_e.detach(),
            'output': x_hat.detach(),
        }
        if return_recon:
            out["x_hat"] = x_hat
        return out
    @torch.no_grad()
    def random_restart(self, z_e: torch.Tensor,threshold=None):
        B, K, C = z_e.shape
        z = z_e.reshape(-1, C)
        self.quant.random_restart(z,threshold=threshold)

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
            no_return_loss=True,
        )
        codes = out["codes"]  # (B,K) or (B,S,K)
        predicted_poses = out["x_hat"]  # (B,T,in_dim)

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
    

class VQVAETransformer_separated(nn.Module):
    """
    VQVAETransformer1D を継承し、位置と手の情報を分離して処理するバージョン。
    位置と手で別々のエンコーダ/デコーダ/量子化器を持ち、最終的にコードを結合して出力する。
    これにより、位置と手の特徴がより独立して学習されることを期待。
    """
    def __init__(
        self,
        in_dim: int = 50,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers_enc: int = 2,
        n_layers_dec: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
        code_dim_pos: int = 64,
        code_dim_hand: int = 64,
        n_codes_pos: int = 512,
        n_codes_hand: int = 512,
        stride: int = 4,
        rvq_stages: int = 2,
        vq_beta: float = 0.25,
        levels_pos: List[int] = [5,3,2],  # FSQ用のレベル設定（位置）
        levels_hand: List[int] = [5,3,2,2],  # FSQ用のレベル設定（手）
        loss_w: VQLossWeights = VQLossWeights(),
    ):
        super().__init__()
        self.in_dim=in_dim
        self.pos_emb = SinusoidalPosEmb(d_model)

        # 上書きして位置と手で別々のモジュールを持つよう
        self.code_dim_pos = len(levels_pos) if rvq_stages <= 1 else code_dim_pos
        self.code_dim_hand = len(levels_hand) if rvq_stages <= 1 else code_dim_hand
        self.n_codes_pos = math.prod(levels_pos) if rvq_stages <= 1 else n_codes_pos
        self.n_codes_hand = math.prod(levels_hand) if rvq_stages <= 1 else n_codes_hand
        self.levels_pos = levels_pos
        self.levels_hand = levels_hand
        self.all_levels=levels_pos + levels_hand
        self.n_codes=math.prod(self.all_levels) if rvq_stages <= 1 else n_codes_pos * n_codes_hand
        # 位置用のエンコーダ/デコーダ/量子化器
        self.in_proj_pos = nn.Linear(8 * 3, d_model)
        enc_layer_pos = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder_pos = nn.TransformerEncoder(enc_layer_pos, num_layers=n_layers_enc)
        self.down_pos = nn.ModuleList()
        if stride > 1:
            for i in range(int(math.log2(stride))):
                self.down_pos.append(nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1))
        self.to_code_pos = nn.Sequential(
            nn.Linear(d_model, self.code_dim_pos),
            nn.LayerNorm(self.code_dim_pos)
        )
        self.from_code_pos = nn.Linear(self.code_dim_pos, d_model)
        dec_layer_pos = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder_pos = nn.TransformerEncoder(dec_layer_pos, num_layers=n_layers_dec)
        self.up_pos = nn.ModuleList()
        if stride > 1:
            for i in range(int(math.log2(stride))):
                self.up_pos.append(nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1, output_padding=0))
        # 手用のエンコーダ/デコーダ/量子化器   
        self.in_proj_hand = nn.Linear(42 * 3, d_model)
        enc_layer_hand = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        ) 
        self.encoder_hand = nn.TransformerEncoder(enc_layer_hand, num_layers=n_layers_enc)
        self.down_hand = nn.ModuleList()
        if stride > 1:
            for i in range(int(math.log2(stride))):
                self.down_hand.append(nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1))
        self.to_code_hand = nn.Sequential(
            nn.Linear(d_model, self.code_dim_hand),
            nn.LayerNorm(self.code_dim_hand)
        )
        if rvq_stages <= 1:
            self.quant= FSQ(levels=self.all_levels)  # FSQを使用
        else:
            self.quant = ResidualVectorQuantizer(n_codes=n_codes_hand, code_dim=code_dim_hand, stages=rvq_stages, beta=vq_beta)
        self.from_code_hand = nn.Linear(self.code_dim_hand, d_model)
        dec_layer_hand = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder_hand = nn.TransformerEncoder(dec_layer_hand, num_layers=n_layers_dec)
        self.up_hand = nn.ModuleList()
        if stride > 1:
            for i in range(int(math.log2(stride))):
                self.up_hand.append(nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1, output_padding=0))
        # 最終出力プロジェクション
        self.dec_ln_out_hand = nn.LayerNorm(d_model)
        self.dec_ln_out_pos = nn.LayerNorm(d_model)
        self.out_proj_pos= nn.Linear(d_model, 8 * 3)
        self.out_proj_hand = nn.Linear(d_model, 42 * 3)
        self.loss_w = loss_w
    # ---------- encode/decode ----------

    def encode(self, x: torch.Tensor, src_mask=None) -> torch.Tensor:
        """
        x: (B,T,in_dim)
        returns z_e: (B,K,code_dim)
        """
        B, T, _ = x.shape
        x_pos= x[:, :, :8 * 3]  # (B,T,24)
        x_hand = x[:, :, 8 * 3:]  # (B,T,126)
        # 位置エンコード
        h_pos = self.in_proj_pos(x_pos)  # (B,T,d_model)
        pe = self.pos_emb(T, device=h_pos.device, dtype=h_pos.dtype)  # (1,T,d_model)
        h_pos = h_pos + pe
        h_pos = self.encoder_pos(h_pos,mask=src_mask).transpose(1,2).contiguous()  # (B,T,d_model)
        for down in self.down_pos:
            h_pos = down(h_pos)  # (B,T//stride,d_model) or (B,T,d_model) if stride=1
        z_e_pos = self.to_code_pos(h_pos.transpose(1,2))  # (B,K,code_dim_pos)
        # 手エンコード
        h_hand = self.in_proj_hand(x_hand)  # (B,T,d_model)
        h_hand = h_hand + pe
        h_hand = self.encoder_hand(h_hand,mask=src_mask).transpose(1,2).contiguous()  # (B,T,d_model)
        for down in self.down_hand:
            h_hand = down(h_hand)  # (B,T//stride,d_model) or (B,T,d_model) if stride=1
        z_e_hand = self.to_code_hand(h_hand.transpose(1,2))  # (B,K,code_dim_hand)
        # コードの結合
        z_e = torch.cat([z_e_pos, z_e_hand], dim=-1)  # (B,K,code_dim_pos + code_dim_hand)
        return z_e
    def decode(self, z_q: torch.Tensor, T_out: int, src_mask=None) -> torch.Tensor:
        """
        z_q: (B,K,code_dim)
        returns x_hat: (B,T_out,in_dim)
        """
        # コードを位置と手に分割
        z_q_pos = z_q[:, :, :self.code_dim_pos]  # (B,K,code_dim_pos)
        z_q_hand = z_q[:, :, self.code_dim_pos:]  # (B,K,code_dim_hand)
        # 位置デコード
        h_pos = self.from_code_pos(z_q_pos)  # (B,K,d_model)
        h_pos = self.decoder_pos(h_pos, mask=src_mask).transpose(1,2).contiguous()  # (B,K,d_model)
        for up in self.up_pos:
            h_pos = up(h_pos)
        h_pos_up = VQVAETransformer1D._safe_crop_or_pad_time(h_pos.transpose(1,2), T_out)  # (B,T,d_model)
        # 手デコード
        h_hand = self.from_code_hand(z_q_hand)  # (B,K,d_model)
        h_hand = self.decoder_hand(h_hand, mask=src_mask).transpose(1,2).contiguous()  # (B,K,d_model)
        for up in self.up_hand:
            h_hand = up(h_hand)
        h_hand_up = VQVAETransformer1D._safe_crop_or_pad_time(h_hand.transpose(1,2), T_out)  # (B,T,d_model)
        # 最終出力プロジェクション
        h_pos_out = self.dec_ln_out_pos(h_pos_up)
        h_hand_out = self.dec_ln_out_hand(h_hand_up)
        x_pos_hat = self.out_proj_pos(h_pos_out)  # (B,T,24)
        x_hand_hat = self.out_proj_hand(h_hand_out)  # (B,T,126)
        x_hat = torch.cat([x_pos_hat, x_hand_hat], dim=-1)  # (B,T,150)
        return x_hat
    @torch.no_grad()
    def tokenize(self, x: torch.Tensor, src_mask=None) -> torch.Tensor:
        z_e = self.encode(x, src_mask=src_mask)
        q = self.quant(z_e)
        return q["codes"]

    @torch.no_grad()
    def reconstruct(self, codes, T_out: int, src_mask=None) -> torch.Tensor:
        if self.is_rvq:
            B, S, K = codes.shape
            z_q_sum = torch.zeros(B, self.code_dim_pos + self.code_dim_hand, device=codes.device)
            for s in range(S):
                z_q_sum = z_q_sum + self.quant.vqs[s].codebook(codes[:, s, :])
        else:
            z_q = self.quant.indices_to_codes(codes)  # (B, K, code_dim)
        x_hat = self.decode(z_q, T_out=T_out, src_mask=src_mask)
        return x_hat

    def soft_repetition_penalty(self, z_e, temperature=0.1) -> torch.Tensor:
        """encoder出力 z_e に対する soft prob分布のentropy促進"""
        # z_e: (B, K, C), codebook: (M, C)
        codebook = self.quant.codebook.weight  # (M, code_dim)
        z_norm = F.normalize(z_e, p=2, dim=-1)  # (B, K, C)
        codebook_norm = F.normalize(codebook, p=2, dim=-1)
        dist = 2 - 2 * (z_norm @ codebook_norm.t())  # (B, K, M)
        probs = F.softmax(-dist / temperature, dim=-1)  # (B, K, M)
        # 隣接フレームの prob 分布のcosineが高すぎたらペナルティ
        cos_sim = (probs[:, 1:] * probs[:, :-1]).sum(-1)  # (B, K-1)
        return cos_sim.mean()
    def forward( self,
        x: torch.Tensor,  # (B,T,J,C)
        *,
        hand_valid_mask: Optional[torch.Tensor] = None,  # (B,T,2) bool
        input_length: Optional[int] = None,  # if provided, will crop/pad input to this length
        return_recon: bool = True,
        no_return_loss: bool = False,  # if True, skip loss computation and return empty dict (for inference)
    ) -> Dict[str, torch.Tensor]:
        B, T, J,C = x.shape
        assert J*C== self.in_dim
        x=x.reshape(B,T,J*C)
        #src_mask=create_slide_window_mask(T,window_size=9,device=x.device) if input_length is not None else None
        src_mask=None
        if input_length is not None:
            mask=create_mask(input_length,max_len=T) #(B,T,1)
        else:
            mask=torch.ones(B,T,device=x.device)
        z_e = self.encode(x,src_mask=src_mask)              # (B,K,code_dim)
        z_e = z_e
        q = self.quant(z_e)                # dict
        z_q = q["z_q"]                     # (B,K,code_dim)
        x_hat = self.decode(z_q, T_out=T,src_mask=src_mask)  # (B,T,36)

        # ---- losses: pos (L1/Huber), dir (cos), vel (small), vq ----
        x=x.reshape(B,T,J,C)
        x_hat=x_hat.reshape(B,T,J,C)
        if no_return_loss:
            return {"x_hat": x_hat, "codes": q["codes"], "perplexity": q["perplexity"].detach(),
                    "loss_vq": q["loss_vq"]}
        x_pos = x[:,:,:8]# (B,T,8,C)
        h_pos = x_hat[:,:,:8]# (B,T,8,C)
        loss_recon_pos = F.smooth_l1_loss(h_pos, x_pos.detach(), reduction='none').mean(dim=[-1,-2])  # (B,T)
        x_hand= x[:,:,8:50]
        h_hand= x_hat[:,:,8:50]# (B,T,42)
        loss_recon_hand=F.smooth_l1_loss(h_hand,x_hand.detach(),reduction='none').mean(dim=[-1,-2])   # (B,T)
        #loss_recon_face=F.smooth_l1_loss(x_hat[:,:,50:],x[:,:,50:].detach(),reduction='none').mean(dim=[-1,-2])  # (B,T)
        mask=~(mask.bool())
        loss_recon_pos=(loss_recon_pos*mask).sum()/mask.sum()
        loss_recon_hand=(loss_recon_hand*mask).sum()/mask.sum()
        #loss_recon_face=(loss_recon_face*mask).sum()/mask.sum()
        loss_vq = q["loss_vq"]
        loss_total = (
            self.loss_w.recon_pos * loss_recon_pos
                + self.loss_w.recon_hand * loss_recon_hand
                #+ self.loss_w.recon_face * loss_recon_face
            + self.loss_w.vq * loss_vq
        )
        out = {
            "loss_total": loss_total,
            "loss_recon_pos": loss_recon_pos.detach(),
            'loss_recon_hand': loss_recon_hand.detach(),
            "loss_vq": loss_vq.detach(),
            "perplexity": q["perplexity"].detach(),
            "codes": q["codes"].detach(),   # (B,K) for VQ, (B,stages,K) for RVQ
            'z_e': z_e.detach(),
            'output': x_hat.detach(),
        }
        if return_recon:
            out["x_hat"] = x_hat
        return out

    @torch.no_grad()
    def random_restart(self, z_e: torch.Tensor,threshold=None):
        B, K, C = z_e.shape
        z = z_e.reshape(-1, C)
        self.quant.random_restart(z,threshold=threshold)

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
            no_return_loss=True,
        )
        codes = out["codes"]  # (B,K) or (B,S,K)
        predicted_poses = out["x_hat"]  # (B,T,in_dim)

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