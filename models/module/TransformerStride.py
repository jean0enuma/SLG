import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class PoseEncoderStride(nn.Module):
    def __init__(self,  hidden_dim):
        super().__init__()
        # 1D畳み込みで時間方向に圧縮（例：長さが1/4になる）
        self.temporal_compress = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (batch, time, joints) -> (batch, joints, time)
        x = x.transpose(1, 2)
        x = self.temporal_compress(x)
        # (batch, hidden, compressed_time) -> (batch, compressed_time, hidden)
        x = x.transpose(1, 2)

        return x
class PoseDecoderStride(nn.Module):
    def __init__(self,  hidden_dim):
        super().__init__()
        self.temporal_decompress = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1)
        )

    def forward(self, z):
        # (batch, hidden, compressed_time) -> (batch, output_dim, time)
        z = z.transpose(1, 2)
        z = self.temporal_decompress(z)
        # (batch, output_dim, time) -> (batch, time, output_dim)
        z = z.transpose(1, 2)
        return z
class PerceiverResampler(nn.Module):
    def __init__(self, dim, num_queries=64, num_heads=8, ff_mult=4):
        super().__init__()
        # 1. 学習可能な固定クエリ（これが最終的なトークン数になる）
        self.queries = nn.Parameter(torch.randn(num_queries, dim))

        # 2. Cross-Attention層
        # クエリがVAEの出力(KV)から情報を取得する
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        # 3. Feed Forward Network (FFN)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: VAEからの出力 (batch_size, seq_len, dim)
        """
        b = x.shape[0]

        # クエリをバッチサイズ分複製 (batch_size, num_queries, dim)
        queries = self.queries.unsqueeze(0).repeat(b, 1, 1)

        # --- Cross-Attention ---
        # Query: Latent Queries
        # Key/Value: VAE outputs (x)
        attn_out, _ = self.attn(queries, x, x)
        x_res = self.norm1(queries + attn_out)

        # --- FFN ---
        ff_out = self.ff(x_res)
        out = self.norm2(x_res + ff_out)

        return out  # (batch_size, num_queries, dim)

class PerceiverDecoder(nn.Module):
    def __init__(self, dim,num_heads=8):
        super().__init__()

        # 2. Cross-Attention層 (Transformer Decoderのコア)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1=nn.LayerNorm(dim)
        self.norm2=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )


    def get_sinusoidal_embeddings(self,n_seq, d_model):
        """サイン・コサインによる位置エンコーディングの生成"""
        pe = torch.zeros(n_seq, d_model)
        position = torch.arange(0, n_seq, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (n_seq, d_model)

    # デコーダのforward内
    def forward(self, z_fixed, target_len):
        device = z_fixed.device
        b = z_fixed.shape[0]
        dim=z_fixed.shape[2]

        # 毎回、必要な長さ分だけの「時間の座標」を生成する
        pos_enc = self.get_sinusoidal_embeddings(target_len, dim).to(device)
        queries = pos_enc.unsqueeze(0).repeat(b, 1, 1)  # (b, target_len, dim)

        # --- Cross-Attention (復元) ---
        # Query: 復元したい各時間のID (queries)
        # Key/Value: 情報が詰まった固定長ベクトル (z_fixed)
        res=queries
        queries=self.norm1(queries)
        attn_out, _ = self.attn(queries, z_fixed, z_fixed)
        queries=res+attn_out
        res=queries
        queries=self.norm2(queries)
        ff_out=self.ffn(queries)
        queries=res+ff_out
        return queries
