"""
hand_gcn_autoencoder.py
========================
MediaPipe Hands (21関節) の手骨格に対する ST-GCN ベース AutoEncoder.

入力テンソル形状:
    1 サンプル : (T, C, J) = (フレーム数, 3, 21)
    バッチ     : (B, T, C, J)
    C = 3  (x, y, z),  J = 21 (MediaPipe Hands ランドマーク)

設計概要:
    - 空間方向: グラフ畳み込み (ST-GCN の spatial partitioning, K=3)
    - 時間方向: 1D 時間畳み込み (encoder=stride downsample / decoder=transposed upsample)
    - ボトルネック: (N, latent_c, T', V) の時空間特徴マップ
                    (時間情報を保持したまま圧縮 — 動作再構成に適した形)
    - 出力: 入力と同じ (B, T, C, J) を線形回帰で復元 (入力長 T は厳密に一致)
"""
from __future__ import annotations
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. グラフ定義 (MediaPipe Hands のトポロジー)
# ----------------------------------------------------------------------
MEDIAPIPE_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 親指   thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # 人差指 index
    (5, 9), (9, 10), (10, 11), (11, 12),     # 中指   middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # 薬指   ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # 小指   pinky
    (0, 17),                                 # 手のひら外周 palm
]


def get_hop_distance(num_node: int, edges, max_hop: int = 1) -> np.ndarray:
    """各ノード対のホップ距離を返す (到達不能は inf)."""
    A = np.zeros((num_node, num_node))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    hop_dis = np.full((num_node, num_node), np.inf)
    transfer = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive = np.stack(transfer) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive[d]] = d
    return hop_dis


def normalize_digraph(A: np.ndarray) -> np.ndarray:
    """列方向の次数正規化 A * D^-1 (ST-GCN 準拠)."""
    Dl = np.sum(A, axis=0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** -1
    return A @ Dn


class HandGraph:
    """MediaPipe Hands 用の隣接行列 A: (K, V, V) を構築.

    strategy:
        'uniform'  : 1 サブセット (正規化隣接行列のみ)
        'distance' : ホップ距離ごとにサブセット分割
        'spatial'  : ST-GCN の空間配置分割 (root / 求心 / 遠心), K=3
    center: 重心とみなす基準ノード (手首=0)
    """

    def __init__(self, num_node: int = 21, edges=MEDIAPIPE_HAND_EDGES,
                 strategy: str = "spatial", max_hop: int = 1,
                 dilation: int = 1, center: int = 0):
        self.num_node = num_node
        self.max_hop = max_hop
        self.dilation = dilation
        self.center = center
        self.hop_dis = get_hop_distance(num_node, edges, max_hop)
        self.A = self._build(strategy)  # (K, V, V) float32

    def _build(self, strategy: str) -> np.ndarray:
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        norm = normalize_digraph(adjacency)

        if strategy == "uniform":
            A = norm[np.newaxis]

        elif strategy == "distance":
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = norm[self.hop_dis == hop]

        elif strategy == "spatial":
            subsets = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_far = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] != hop:
                            continue
                        di = self.hop_dis[i, self.center]
                        dj = self.hop_dis[j, self.center]
                        if dj == di:
                            a_root[j, i] = norm[j, i]
                        elif dj > di:
                            a_close[j, i] = norm[j, i]
                        else:
                            a_far[j, i] = norm[j, i]
                if hop == 0:
                    subsets.append(a_root)
                else:
                    subsets.append(a_root + a_close)
                    subsets.append(a_far)
            A = np.stack(subsets)
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        return A.astype(np.float32)


# ----------------------------------------------------------------------
# 2. グラフ畳み込み (空間方向)
# ----------------------------------------------------------------------
class SpatialGraphConv(nn.Module):
    """ST-GCN の空間グラフ畳み込み.  x:(N,C,T,V), A:(K,V,V)."""

    def __init__(self, in_channels: int, out_channels: int, K: int):
        super().__init__()
        self.K = K
        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        N, KC, T, V = x.shape
        x = x.view(N, self.K, KC // self.K, T, V)
        # 各サブセット k の隣接で関節間集約: sum_v x[k,v] * A[k,v,w]
        x = torch.einsum("nkctv,kvw->nctw", x, A)
        return x.contiguous()


# ----------------------------------------------------------------------
# 3. Encoder / Decoder ブロック
# ----------------------------------------------------------------------
class STGCNBlock(nn.Module):
    """Encoder 用: 空間 GCN + 時間 Conv (stride で時間方向ダウンサンプル)."""

    def __init__(self, in_channels, out_channels, K, t_kernel=9,
                 stride=1, residual=True, dropout=0.0):
        super().__init__()
        pad = (t_kernel - 1) // 2
        self.gcn = SpatialGraphConv(in_channels, out_channels, K)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (t_kernel, 1),
                      (stride, 1), (pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda _: 0.0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.act(x)


class STGCNUpBlock(nn.Module):
    """Decoder 用: 空間 GCN + 時間 TransposedConv (時間方向アップサンプル)."""

    def __init__(self, in_channels, out_channels, K, t_kernel=9,
                 stride=1, residual=True, dropout=0.0):
        super().__init__()
        pad = (t_kernel - 1) // 2
        self.gcn = SpatialGraphConv(in_channels, out_channels, K)
        if stride > 1:
            tconv = nn.ConvTranspose2d(
                out_channels, out_channels, (t_kernel, 1), (stride, 1),
                (pad, 0), output_padding=(stride - 1, 0))
        else:
            tconv = nn.Conv2d(out_channels, out_channels, (t_kernel, 1),
                              (stride, 1), (pad, 0))
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            tconv,
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        self.use_res = residual
        if residual:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, A):
        out = self.gcn(x, A)
        out = self.tcn(out)
        if self.use_res:
            res = self.res_conv(x)
            if res.shape[2] != out.shape[2]:  # 時間長を合わせてからスキップ接続
                res = F.interpolate(res, size=out.shape[2:], mode="nearest")
            out = out + res
        return self.act(out)


class Encoder(nn.Module):
    def __init__(self, in_channels, K, channels=(64, 128, 256),
                 strides=(1, 2, 2), t_kernel=9):
        super().__init__()
        blocks, c_prev = [], in_channels
        for c, s in zip(channels, strides):
            blocks.append(STGCNBlock(c_prev, c, K, t_kernel, stride=s))
            c_prev = c
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = c_prev

    def forward(self, x, A):
        for blk in self.blocks:
            x = blk(x, A)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_channels, out_channels, K,
                 channels=(128, 64), strides=(2, 2), t_kernel=9):
        super().__init__()
        blocks, c_prev = [], latent_channels
        for c, s in zip(channels, strides):
            blocks.append(STGCNUpBlock(c_prev, c, K, t_kernel, stride=s))
            c_prev = c
        self.blocks = nn.ModuleList(blocks)
        # 出力ヘッド: 活性化なし = 座標を線形回帰で復元
        self.head = SpatialGraphConv(c_prev, out_channels, K)

    def forward(self, x, A):
        for blk in self.blocks:
            x = blk(x, A)
        return self.head(x, A)


# ----------------------------------------------------------------------
# 4. AutoEncoder 本体
# ----------------------------------------------------------------------
class HandGCNAutoEncoder(nn.Module):
    """ST-GCN ベースの手骨格 AutoEncoder.

    入出力: (B, T, C, J) または (T, C, J)
    enc_strides の積 == dec_strides の積 となるよう設定すれば時間長が概ね一致し、
    端数は forward 内で線形補間して厳密に入力長へ揃える。
    """

    def __init__(self, in_channels: int = 3, num_joints: int = 21,
                 graph_strategy: str = "spatial",
                 enc_channels: Sequence[int] = (64, 128, 256),
                 enc_strides: Sequence[int] = (1, 2, 2),
                 dec_channels: Sequence[int] = (128, 64),
                 dec_strides: Sequence[int] = (2, 2),
                 latent_channels: int | None = None,
                 t_kernel: int = 9):
        super().__init__()
        self.in_channels = in_channels
        self.num_joints = num_joints

        graph = HandGraph(num_node=num_joints, strategy=graph_strategy)
        self.register_buffer("A", torch.from_numpy(graph.A))  # (K,V,V)
        self.K = self.A.shape[0]

        # 入力正規化 (C*V チャネルの BatchNorm1d)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.encoder = Encoder(in_channels, self.K, enc_channels,
                               enc_strides, t_kernel)
        enc_out = self.encoder.out_channels
        self.latent_channels = latent_channels or enc_out
        self.to_latent = nn.Conv2d(enc_out, self.latent_channels, 1)

        self.decoder = Decoder(self.latent_channels, in_channels, self.K,
                               dec_channels, dec_strides, t_kernel)
        self._init_weights()

    # ---- 形状変換ユーティリティ ----------------------------------
    def _to_internal(self, x):
        squeeze = x.dim() == 3
        if squeeze:                       # (T,C,J) -> (1,T,C,J)
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B,T,C,J) -> (B,C,T,V)
        return x, squeeze

    def _to_external(self, x, squeeze):
        x = x.permute(0, 2, 1, 3).contiguous()   # (B,C,T,V) -> (B,T,C,J)
        return x.squeeze(0) if squeeze else x

    def _data_norm(self, x):
        N, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        return x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

    def _match_time(self, x, T_target):
        if x.shape[2] == T_target:
            return x
        N, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)
        x = F.interpolate(x, size=T_target, mode="linear", align_corners=False)
        return x.reshape(N, C, V, T_target).permute(0, 1, 3, 2).contiguous()

    # ---- 公開 API -----------------------------------------------
    def encode(self, x):
        """生入力 (B,T,C,J) -> 潜在 (B, latent_c, T', V)."""
        x, _ = self._to_internal(x)
        x = self._data_norm(x)
        return self.to_latent(self.encoder(x, self.A))

    def decode(self, z, T_target=None):
        """潜在 z -> 復元 (B,T,C,J).  T_target 指定で時間長を合わせる."""
        out = self.decoder(z, self.A)
        if T_target is not None:
            out = self._match_time(out, T_target)
        return self._to_external(out, squeeze=False)

    def forward(self, x):
        x_in, squeeze = self._to_internal(x)
        T_in = x_in.shape[2]
        h = self._data_norm(x_in)
        z = self.to_latent(self.encoder(h, self.A))
        out = self.decoder(z, self.A)
        out = self._match_time(out, T_in)
        return self._to_external(out, squeeze)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ----------------------------------------------------------------------
# 5. 再構成損失
# ----------------------------------------------------------------------
def reconstruction_loss(pred, target, mask=None):
    """(B,T,C,J) 同士の MSE.
    mask: (B,T,J) 有効関節=1 / 欠損・低信頼=0 を与えると masked-MSE.
    MediaPipe は欠損関節が出やすいため、補間/可視性を反映した学習に有用.
    """
    if mask is None:
        return F.mse_loss(pred, target)
    mask = mask.unsqueeze(2)  # (B,T,1,J)
    se = ((pred - target) ** 2) * mask
    denom = mask.sum() * pred.shape[2] + 1e-8
    return se.sum() / denom


def make_mask(input_length: torch.Tensor, input_data: torch.Tensor) -> torch.Tensor:
    """
    input_length: (B,)         各バッチの有効フレーム長
    input_data:   (B, T, C, J) 座標データ．全チャンネルが0の点を欠損とみなす
    return:
        mask: (B, T, J) bool. True=有効(欠損でも無く、pad範囲でもない), False=無効
    """
    B, T, C, J = input_data.shape
    device = input_data.device

    # 欠損マスク: 全チャンネルが0なら欠損 -> (B, T, J)
    valid_value_mask = (input_data != 0).any(dim=2)  # (B, T, J)

    # 長さマスク: t < input_length -> (B, T)
    time_idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    valid_length_mask = time_idx < input_length.unsqueeze(1)  # (B, T)
    valid_length_mask = valid_length_mask.unsqueeze(-1).expand(-1, -1, J)  # (B, T, J)

    mask = valid_value_mask & valid_length_mask
    return mask

# ----------------------------------------------------------------------
# 6. 動作確認
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    model = HandGCNAutoEncoder()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"graph A: {tuple(model.A.shape)}  (K,V,V)")
    print(f"params : {n_params/1e6:.2f} M")

    # バッチ入力 (B,T,C,J)
    x = torch.randn(4, 64, 3, 21)
    y = model(x)
    z = model.encode(x)
    print(f"input  : {tuple(x.shape)}")
    print(f"latent : {tuple(z.shape)}")
    print(f"output : {tuple(y.shape)}")
    assert y.shape == x.shape

    # 奇数長・単一サンプル (T,C,J) も時間長を厳密復元
    x2 = torch.randn(37, 3, 21)
    y2 = model(x2)
    print(f"single : {tuple(x2.shape)} -> {tuple(y2.shape)}")
    assert y2.shape == x2.shape

    # 損失計算
    loss = reconstruction_loss(y, x)
    mask = (torch.rand(4, 64, 21) > 0.1).float()
    mloss = reconstruction_loss(y, x, mask)
    print(f"mse={loss.item():.4f}  masked_mse={mloss.item():.4f}")
    loss.backward()
    print("backward OK")