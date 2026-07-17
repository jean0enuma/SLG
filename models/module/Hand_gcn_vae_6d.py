"""
hand_gcn_vae_6d.py
===================
HAND_BONES の 6D 回転表現を入出力とする ST-GCN ベース VAE.

入力テンソル形状:
    1 サンプル : (T, C, J) = (フレーム数, 6, 20)   C=6 (6D回転), J=20 (ボーン数)
    バッチ     : (B, T, C, J)

hand_gcn_autoencoder.py からの主な変更点:
    1. グラフ: 関節グラフ (21ノード) -> ボーングラフ (20ノード)
       - 関節を共有するボーン同士を接続
       - spatial partitioning の求心/遠心判定はキネマティック深さで行う
    2. AE -> VAE: to_latent を to_mu / to_logvar に分岐し再パラメータ化
    3. 損失: 6D-MSE + 復元回転行列の Frobenius 損失 + beta * KL
    4. 前処理: (T, 3, 21) 関節座標 -> (T, 6, 20) 6D 特徴の変換関数を同梱

依存: 同ディレクトリの hand_gcn_autoencoder.py (Encoder / Decoder を再利用)
"""
from __future__ import annotations
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.STGCNHand import  Encoder, Decoder, normalize_digraph
from loader.coordinate_preprocess import HAND_BONES,BODY_BONES,joints_to_rotation_inputs,build_rotation_from_joints,ALL_BONES

def bones_to_joint_parents(bones, num_joints=None):
    """(parent, child, aux) のボーンリスト -> JOINT_PARENTS (rootと未使用は-1).
    関節番号が疎(歯抜け)でも動作する. 検証はボーンに登場する関節のみ対象."""
    used = {p for p, c, *_ in bones} | {c for p, c, *_ in bones}
    if num_joints is None:
        num_joints = max(used) + 1
    parents = [-1] * num_joints          # 未使用インデックスも -1 のまま

    children = set()
    for p, c, *_ in bones:               # auxは無視(運動学的接続ではない)
        if c in children:
            raise ValueError(f"joint {c} has multiple parents")
        children.add(c)
        parents[c] = p

    # 検証: rootが(使用関節の中で)1つだけか
    roots = sorted(used - children)
    if len(roots) != 1:
        raise ValueError(f"expected 1 root, got {roots}")

    # 検証: 使用関節がすべてrootに到達可能か(ループ検出)
    for j in used:
        cur, steps = j, 0
        while parents[cur] != -1:
            cur = parents[cur]
            steps += 1
            if steps > len(used):
                raise ValueError(f"cycle detected from joint {j}")
    return parents
# ----------------------------------------------------------------------
# 1. ボーン定義 (parent, child, aux)  ※過去の会話で設計した隣MCP方式
# ----------------------------------------------------------------------
NUM_HAND_BONES = len(HAND_BONES)  # 20
NUM_BODY_BONES = len(BODY_BONES)  # 19
NUM_ALL_BONES=len(ALL_BONES)    # 39
# 関節の親配列 (ボーン深さの計算に使用)
HAND_JOINT_PARENTS = bones_to_joint_parents(HAND_BONES)
BODY_JOINT_PARENTS=bones_to_joint_parents(BODY_BONES)
ALL_JOINT_PARENTS=bones_to_joint_parents(ALL_BONES)


def _joint_depth(parents: Sequence[int]) -> list[int]:
    depth = [0] * len(parents)
    for j, p in enumerate(parents):
        d, cur = 0, j
        while parents[cur] != -1:
            cur = parents[cur]
            d += 1
        depth[j] = d
    return depth


def build_bone_edges(bones=HAND_BONES) -> list[tuple[int, int]]:
    """関節を共有するボーン同士を接続 (auxは接続判定に使わない)."""
    edges = []
    for i in range(len(bones)):
        for j in range(i + 1, len(bones)):
            if {bones[i][0], bones[i][1]} & {bones[j][0], bones[j][1]}:
                edges.append((i, j))
    return edges


class BoneGraph:
    """ボーンをノードとする隣接行列 A: (K, V, V) を構築.

    spatial strategy では ST-GCN の hop_dis[i, center] の代わりに
    キネマティック深さ (bone_depth = 子関節のツリー深さ) を使って
    root / 求心 / 遠心 のサブセットに分割する。
    """

    def __init__(self, bones=HAND_BONES, strategy: str = "spatial",
                 max_hop: int = 1):
        self.num_node = len(bones)
        self.edges = build_bone_edges(bones)
        if bones==HAND_BONES:
            jd_parents = HAND_JOINT_PARENTS
        elif bones==BODY_BONES:
            jd_parents = BODY_JOINT_PARENTS
        elif bones==ALL_BONES:
            jd_parents = ALL_JOINT_PARENTS
        else:
            raise ValueError(f"Unknown bones: {bones}")
        jd = _joint_depth(jd_parents)
        self.depth = np.array([jd[c] for (_, c, _) in bones])  # (V,)
        self.hop_dis = self._hop_distance(max_hop)
        self.A = self._build(strategy, max_hop)

    def _hop_distance(self, max_hop: int) -> np.ndarray:
        n = self.num_node
        A = np.zeros((n, n))
        for i, j in self.edges:
            A[i, j] = A[j, i] = 1
        hop_dis = np.full((n, n), np.inf)
        transfer = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive = np.stack(transfer) > 0
        for d in range(max_hop, -1, -1):
            hop_dis[arrive[d]] = d
        return hop_dis

    def _build(self, strategy: str, max_hop: int) -> np.ndarray:
        valid_hop = range(0, max_hop + 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        norm = normalize_digraph(adjacency)

        if strategy == "uniform":
            A = norm[np.newaxis]

        elif strategy == "distance":
            A = np.zeros((len(list(valid_hop)), self.num_node, self.num_node))
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
                        di, dj = self.depth[i], self.depth[j]
                        if dj == di:
                            a_root[j, i] = norm[j, i]
                        elif dj > di:      # j が末端側 -> 求心
                            a_close[j, i] = norm[j, i]
                        else:              # j が root 側 -> 遠心
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
# 2. 6D 回転の前処理 / 復元ユーティリティ
# ----------------------------------------------------------------------
def build_rotation_from_joints(p_parent, p_child, p_aux, eps: float = 1e-6):
    """3点から局所回転行列 (..., 3, 3) を構成 (列基底). 併せて退化マスクも返す."""
    x = F.normalize(p_child - p_parent, dim=-1, eps=eps)
    v = p_aux - p_parent
    y_raw = v - (v * x).sum(-1, keepdim=True) * x
    valid = y_raw.norm(dim=-1) > 1e-4          # False = 共線退化
    y = F.normalize(y_raw, dim=-1, eps=eps)
    z = torch.cross(x, y, dim=-1)
    R = torch.stack([x, y, z], dim=-1)         # 列ベクトルとして積む
    return R, valid


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3). Gram-Schmidt で正規直交化 (列基底)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1,
                     dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def hand_joints_to_6d(x: torch.Tensor, bones=HAND_BONES,
                      return_valid: bool = False):
    """
    関節座標 -> 6D 回転特徴.
    x: (T, 3, 21) or (B, T, 3, 21)
    return:
        d6:    (..., T, 6, 20)   モデル入力形状
        valid: (..., T, 20)      True=非退化 (return_valid=True のとき)
    """
    pos = x.movedim(-2, -1)                                  # (..., T, 21, 3)
    idx = torch.as_tensor(bones, device=x.device)            # (20, 3)
    p_par = pos[..., idx[:, 0], :]
    p_chi = pos[..., idx[:, 1], :]
    p_aux = pos[..., idx[:, 2], :]
    R, valid = build_rotation_from_joints(p_par, p_chi, p_aux)
    d6 = R[..., :, :2].transpose(-1, -2).reshape(*R.shape[:-2], 6)
    d6 = d6.movedim(-1, -2).contiguous()                     # (..., T, 6, 20)
    return (d6, valid) if return_valid else d6


def make_bone_mask(joint_mask: torch.Tensor, bones=HAND_BONES,
                   valid_rot: torch.Tensor | None = None) -> torch.Tensor:
    """
    関節マスク (B, T, 21) -> ボーンマスク (B, T, 20).
    parent / child / aux の3関節すべて有効なボーンのみ True.
    valid_rot: hand_joints_to_6d の退化マスク (B, T, 20) を AND できる.
    """
    idx = torch.as_tensor(bones, device=joint_mask.device)   # (20, 3)
    m = (joint_mask[..., idx[:, 0]]
         & joint_mask[..., idx[:, 1]]
         & joint_mask[..., idx[:, 2]])                       # (B, T, 20)
    if valid_rot is not None:
        m = m & valid_rot
    return m

def reconstruct_joints_from_6d(x_ref, d6, bones,bone_mask=None, num_joints=None, eps=1e-8):
    """
    モデル出力の6D表現から骨格座標を復元する(FK).
    ボーン長・root軌跡は参照座標 x_ref から抽出する.

    Args:
        x_ref: (B, T, 3, J) or (T, 3, J)  元の骨格座標(復元パラメータの供給源)
        d6:    (B, T, 6, Nb) or (T, 6, Nb) モデル出力の6D表現(非直交でも可)
        bones: [(parent, child, aux), ...]  Nb 本(auxは未使用)
        num_joints: 出力の関節数 J(省略時は x_ref から取得)
    Returns:
        x_rec: (B, T, 3, J)  復元座標(入力が3次元なら (T, 3, J))
    """
    squeeze = x_ref.dim() == 3
    if squeeze:
        x_ref, d6 = x_ref.unsqueeze(0), d6.unsqueeze(0)
    B, T, _, J = x_ref.shape
    J = num_joints or J
    pos_ref = x_ref.movedim(-2, -1)  # (B, T, J, 3)
    idx = torch.as_tensor([b[:2] for b in bones], device=x_ref.device)
    vec = pos_ref[..., idx[:, 1], :] - pos_ref[..., idx[:, 0], :]
    #L=compute_bone_lengths(x_ref,bones,bone_mask=bone_mask)               # (B, Nb)
    # --- ① ボーン長: median を外してフレームごとの長さをそのまま使う ---

    # 変更前:
    # L = vec.norm(dim=-1).median(dim=1).values      # (B, Nb)
    # 変更後:
    L = vec.norm(dim=-1)  # (B, T, Nb)  フレームごと

    # --- ② seg のブロードキャスト形状を変更 ---
    # 変更前:
    # seg = R[..., :, 0] * L.view(B, 1, -1, 1)       # (B,1,Nb,1) をT方向に展開
    # 変更後:


    # --- ② root関節: どのボーンのchildにも現れない関節 ---
    children = {c for _, c, *_ in bones}
    used = {p for p, _, *_ in bones} | children
    roots = sorted(used - children)
    assert len(roots) == 1, f"expected 1 root, got {roots}"
    root = roots[0]

    # --- ③ 6D -> 回転行列 -> ボーン方向 × 長さ ---
    R = rotation_6d_to_matrix(d6.movedim(-2, -1))            # (B, T, Nb, 3, 3)
    #seg = R[..., :, 0] * L.view(B, 1, -1, 1)                 # 第1列 = ボーン方向
    seg = R[..., :, 0] * L.unsqueeze(-1)  # (B,T,Nb,1) 各フレーム固有の長さ

    # --- ④ FK: rootから木の順に位置を積算(トポロジカル順で解決) ---
    joints = pos_ref.clone()          # boneに含まれない関節は参照値を保持
    joints[..., root, :] = pos_ref[..., root, :]             # root軌跡は参照から
    placed = {root}
    remaining = list(enumerate(bones))
    while remaining:
        rest = []
        for b, (p, c, *_) in remaining:
            if p in placed:
                joints[..., c, :] = joints[..., p, :] + seg[..., b, :]
                placed.add(c)
            else:
                rest.append((b, (p, c)))
        if len(rest) == len(remaining):
            raise ValueError(f"unreachable bones: {[bc for _, bc in rest]}")
        remaining = rest

    x_rec = joints.movedim(-1, -2).contiguous()              # (B, T, 3, J)
    return x_rec.squeeze(0) if squeeze else x_rec

# ----------------------------------------------------------------------
# 3. VAE 本体
# ----------------------------------------------------------------------
class HandGCNVAE(nn.Module):
    """6D 回転特徴 (B, T, 6, 20) を入出力とする ST-GCN VAE.

    潜在: 時空間特徴マップ z ~ N(mu, sigma^2), 形状 (B, latent_c, T', V)
    ※ 6D 成分は概ね [-1, 1] に正規化済みのため data_bn はデフォルト無効。
    """

    def __init__(self, in_channels: int = 6, bones=HAND_BONES,
                 graph_strategy: str = "spatial",
                 enc_channels: Sequence[int] = (64, 128, 256),
                 enc_strides: Sequence[int] = (1, 2, 2),
                 dec_channels: Sequence[int] = (128, 64),
                 dec_strides: Sequence[int] = (2, 2),
                 latent_channels: int = 64,
                 t_kernel: int = 9,
                 use_data_bn: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.num_nodes = len(bones)
        self.latent_channels = latent_channels
        self.t_stride = int(np.prod(enc_strides))  # 時間方向の圧縮率

        graph = BoneGraph(bones, strategy=graph_strategy)
        self.register_buffer("A", torch.from_numpy(graph.A))  # (K, V, V)
        self.K = self.A.shape[0]

        self.data_bn = (nn.BatchNorm1d(in_channels * self.num_nodes)
                        if use_data_bn else nn.Identity())

        self.encoder = Encoder(in_channels, self.K, enc_channels,
                               enc_strides, t_kernel)
        enc_out = self.encoder.out_channels
        self.to_mu = nn.Conv2d(enc_out, latent_channels, 1)
        self.to_logvar = nn.Conv2d(enc_out, latent_channels, 1)

        self.decoder = Decoder(latent_channels, in_channels, self.K,
                               dec_channels, dec_strides, t_kernel)
        self._init_weights()
        # 学習初期の KL 爆発を防ぐため logvar ヘッドは 0 付近から開始
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.zeros_(self.to_logvar.bias)

    # ---- 形状変換ユーティリティ ----------------------------------
    def _to_internal(self, x):
        squeeze = x.dim() == 3
        if squeeze:                                   # (T,C,J) -> (1,T,C,J)
            x = x.unsqueeze(0)
        return x.permute(0, 2, 1, 3).contiguous(), squeeze  # -> (B,C,T,V)

    def _to_external(self, x, squeeze):
        x = x.permute(0, 2, 1, 3).contiguous()        # (B,C,T,V) -> (B,T,C,J)
        return x.squeeze(0) if squeeze else x

    def _data_norm(self, x):
        if isinstance(self.data_bn, nn.Identity):
            return x
        N, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        return x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

    def _match_time(self, x, T_target):
        if x.shape[2] == T_target:
            return x
        N, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)
        x = F.interpolate(x, size=T_target, mode="linear",
                          align_corners=False)
        return x.reshape(N, C, V, T_target).permute(0, 1, 3, 2).contiguous()

    # ---- VAE コア -------------------------------------------------
    def encode(self, x):
        """(B,T,6,20) -> mu, logvar 各 (B, latent_c, T', V)."""
        x, _ = self._to_internal(x)
        h = self.encoder(self._data_norm(x), self.A)
        return self.to_mu(h), self.to_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, T_target: int | None = None):
        """z: (B, latent_c, T', V) -> (B, T, 6, 20)."""
        out = self.decoder(z, self.A)
        if T_target is not None:
            out = self._match_time(out, T_target)
        return self._to_external(out, squeeze=False)

    def forward(self, x):
        x_in, squeeze = self._to_internal(x)
        T_in = x_in.shape[2]
        h = self.encoder(self._data_norm(x_in), self.A)
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        z = self.reparameterize(mu, logvar) if self.training else mu
        out = self._match_time(self.decoder(z, self.A), T_in)
        return self._to_external(out, squeeze), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, T: int, device=None):
        """事前分布 N(0, I) からサンプリング生成. T は出力フレーム数."""
        device = device or self.A.device
        T_lat = max(1, T // self.t_stride)
        z = torch.randn(n, self.latent_channels, T_lat, self.num_nodes,
                        device=device)
        return self.decode(z, T_target=T)

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
# 4. 損失関数
# ----------------------------------------------------------------------
def bones_used_joints(bones, include_aux=True):
    """bonesに登場する関節インデックスをソート済みリストで返す.

    include_aux=True : parent/child/aux すべて (座標の切り出し用)
    include_aux=False: parent/child のみ (キネマティクス構造のみ)
    """
    used = set()
    for b in bones:
        used.update(b[:3] if include_aux else b[:2])
    return sorted(used)
def kl_divergence(mu, logvar, free_bits: float = 0.0):
    """KL(q||N(0,I)). 要素ごとに計算しサンプル毎に総和 -> バッチ平均.
    free_bits > 0 で次元ごとの KL を下限クランプ (posterior collapse 対策)."""
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, C, T', V)
    if free_bits > 0:
        kl = kl.clamp_min(free_bits)
    return kl.flatten(1).sum(-1).mean()

def fk_loss(pred_6d, x_gt, bones, joint_mask=None, joint_weight=None):
    """
    pred_6d: (B, T, 6, Nb)  モデル出力
    x_gt:    (B, T, 3, J)   GT座標(勾配不要のリファレンス)
    joint_mask:   (B, T, J)  有効関節
    joint_weight: (J,)       関節別重み(指先を重くする等)
    """
    joints=reconstruct_joints_from_6d(x_gt, pred_6d, bones)

    bone_idx=bones_used_joints(bones, include_aux=True)
    joints=joints[...,bone_idx]
    pos_gt=x_gt[...,bone_idx]

    # FK積算(rootはGT位置)
    se = (joints - pos_gt).pow(2).sum(-1)        # (B, T, J)
    if joint_weight is not None:
        se = se * joint_weight.view(1, 1, -1)
    if joint_mask is not None:
        m = joint_mask.float()
        m=m[...,:3]
        return (se * m).sum() / (m.sum() + 1e-8)
    return se.mean()
def vae_loss(recon, target, mu, logvar, mask=None,
             beta: float = 1e-4, w_6d: float = 1.0, w_mat: float = 1.0,w_vel: float = 0.0,
             free_bits: float = 0.05,target_cod=None,w_fk:float=1.0,bones=HAND_BONES,j_mask=None):
    """
    recon, target: (B, T, 6, 20)
    mask:          (B, T, 20)  True=有効ボーン (make_bone_mask の出力)
    returns: total, dict(各項)

    - 6D 空間の MSE (勾配の主経路)
    - Gram-Schmidt 復元後の回転行列 Frobenius 損失 (SO(3) 上の整合)
    - beta * KL
    """
    if mask is not None:
        m6 = mask.unsqueeze(2).float()                       # (B,T,1,20)
        denom6 = m6.sum() * recon.shape[2] + 1e-8
        loss_6d = (((recon - target) ** 2) * m6).sum() / denom6
    else:
        loss_6d = F.mse_loss(recon, target)

    # (B,T,6,20) -> (B,T,20,6) -> (B,T,20,3,3)
    R_pred = rotation_6d_to_matrix(recon.movedim(-2, -1))
    R_gt = rotation_6d_to_matrix(target.movedim(-2, -1))
    mat_se = (R_pred - R_gt).pow(2).sum((-2, -1))            # (B,T,20)
    if mask is not None:
        mf = mask.float()
        loss_mat = (mat_se * mf).sum() / (mf.sum() + 1e-8)
    else:
        loss_mat = mat_se.mean()

    loss_kl = kl_divergence(mu, logvar, free_bits)
    if target_cod!=None:
        loss_fk=fk_loss(recon, target_cod, bones,joint_mask=j_mask)
    else:
        loss_fk=torch.tensor(0.0, device=recon.device)
    if mask is not None:
        loss_vel = (F.mse_loss(recon[:, 1:] - recon[:, :-1],
                              target[:, 1:] - target[:, :-1],reduction="mean"))
    else:
        loss_vel = F.mse_loss(recon[:, 1:] - recon[:, :-1],
                              target[:, 1:] - target[:, :-1])

    total = w_6d * loss_6d + w_mat * loss_mat + beta * loss_kl+ w_fk * loss_fk+ w_vel * loss_vel
    return total, {"6d": loss_6d.item(), "mat": loss_mat.item(),
                   "kl": loss_kl.item(), "total": total.item()}


def vae_loss_cod(recon, target, mu, logvar, mask=None,
             beta: float = 1e-3, w_3d: float = 1.0,
             free_bits: float = 0.0):
    """
    recon, target: (B, T, 6, 20)
    mask:          (B, T, 20)  True=有効ボーン (make_bone_mask の出力)
    returns: total, dict(各項)

    - 6D 空間の MSE (勾配の主経路)
    - Gram-Schmidt 復元後の回転行列 Frobenius 損失 (SO(3) 上の整合)
    - beta * KL
    """
    if mask is not None:
        m6 = mask.unsqueeze(2).float()                       # (B,T,1,20)
        denom6 = m6.sum() * recon.shape[2] + 1e-8
        loss_6d = (((recon - target) ** 2) * m6).sum() / denom6
    else:
        loss_6d = F.mse_loss(recon, target)

    loss_kl = kl_divergence(mu, logvar, free_bits)
    total = w_3d * loss_6d + beta * loss_kl
    return total, {"6d": loss_6d.item(),
                   "kl": loss_kl.item(), "total": total.item()}

# ----------------------------------------------------------------------
# 5. 動作確認
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    model = HandGCNVAE()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"bone graph A: {tuple(model.A.shape)}  (K,V,V)")
    print(f"params      : {n_params / 1e6:.2f} M")

    # --- 前処理: 関節座標 (B,T,3,21) -> 6D (B,T,6,20) ---
    joints = torch.randn(4, 64, 3, 21)
    d6, valid = hand_joints_to_6d(joints, return_valid=True)
    print(f"joints {tuple(joints.shape)} -> 6d {tuple(d6.shape)}, "
          f"valid {tuple(valid.shape)}")

    # --- マスク: 関節マスク + 退化マスク -> ボーンマスク ---
    joint_mask = torch.rand(4, 64, 21) > 0.05
    bone_mask = make_bone_mask(joint_mask, valid_rot=valid)
    print(f"bone_mask: {tuple(bone_mask.shape)}  "
          f"valid ratio={bone_mask.float().mean():.3f}")

    # --- 学習ステップ ---
    model.train()
    recon, mu, logvar = model(d6)
    print(f"recon {tuple(recon.shape)}  latent {tuple(mu.shape)}")
    assert recon.shape == d6.shape

    total, logs = vae_loss(recon, d6, mu, logvar, mask=bone_mask,
                           beta=1e-3, free_bits=0.02)
    total.backward()
    print(f"loss: {logs}")
    print("backward OK")

    # --- 単一サンプル (T,C,J), 奇数長 ---
    d6_single = hand_joints_to_6d(torch.randn(37, 3, 21))
    r, mu1, lv1 = model(d6_single)
    print(f"single: {tuple(d6_single.shape)} -> {tuple(r.shape)}")
    assert r.shape == d6_single.shape

    # --- 事前分布からの生成 ---
    model.eval()
    gen = model.sample(n=2, T=64)
    R_gen = rotation_6d_to_matrix(gen.movedim(-2, -1))
    det = torch.det(R_gen)
    print(f"sampled: {tuple(gen.shape)}, det(R) mean={det.mean():.4f}")