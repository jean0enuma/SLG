import copy
import torch
import jiwer
import numpy as np
from math import isinf
from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from tqdm import tqdm
from typing import List, Optional, Callable
from inspect import getfullargspec
from sacrebleu.metrics import CHRF
from scipy.spatial.distance import euclidean
from scipy import linalg  # FID の共分散平方根に使用

from utils.external_metrics.sacrebleu import raw_corpus_bleu
from utils.external_metrics.rouge import calc_score as rouge_calc_score
# Apply DTW
def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

def mse(y_true, y_pred):
    if isinstance(y_true, list) and isinstance(y_pred, list):
        # numpy join
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def token_accuracy(hypotheses: list, references: list):
    """
    Calculate the token accuracy between hypotheses and references.

    Args:
    - hypotheses (list): List of hypothesis sequences.
    - references (list): List of reference sequences.

    Returns:
    - token_accuracy (float): Token accuracy.
    """
    token_accuracy = 0
    n_seq = len(hypotheses)

    for h, r in zip(hypotheses, references):
        token_accuracy += (h == r).sum() / len(r) / n_seq

    return token_accuracy


def mpjpe(gt_poses, hypo_poses):
    """
    Calculate the Mean Per Joint Position Error (MPJPE) between ground truth and hypothesis poses.

    Args:
    - gt_poses (torch.Tensor): Ground truth poses with shape (num_samples, num_joints, 3).
    - hypo_poses (torch.Tensor): Hypothesis poses with shape (num_samples, num_joints, 3).

    Returns:
    - mean_mpjpe (float): Mean Per Joint Position Error.
    """

    def cal_mpjpe(a, b):
        # Check if the input tensors have the same shape
        assert (
                a.shape == b.shape
        ), "Ground truth and hypothesis poses must have the same shape."

        # Calculate the Euclidean distance between corresponding joints
        joint_distances = torch.norm(a - b, dim=2)

        # Calculate the mean over all joints and samples
        mean_mpjpe = torch.mean(joint_distances)
        return mean_mpjpe
    mpjpe_list=[]
    for gt_pose, hypo_pose in zip(gt_poses, hypo_poses):
        gt_pose=_to_TJ3(gt_pose.detach().cpu())
        hypo_pose=_to_TJ3(hypo_pose.detach().cpu())
        mpjpe_dist=cal_mpjpe(gt_pose, hypo_pose)
        mpjpe_list.append(mpjpe_dist)
    return torch.mean(torch.tensor(mpjpe_list)).item()


def pose_error_align_mje(hyps: list = None, gt_pose: torch.Tensor = None):
    """Coverts the codebook indexs to poses and calculates the MPJPE error"""

    def euclidean_distance(x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        return torch.sqrt(torch.sum((x - y) ** 2))

    def dtw_align_data(a: list, b: list, dist_fn=euclidean_distance):
        align_a = []
        align_b = []
        for _a, _b in tqdm(zip(a, b), total=len(a), desc="Aligning A to B:"):
            #  skip blank sequences
            if _a is None or _b is None:
                continue
            if len(_a) == 0 or len(_b) == 0:
                continue
            dist, path = dtw(_a.flatten(1, -1), _b.flatten(1, -1), dist=dist_fn)
            a_path, b_path = zip(*path)
            _a = _a[list(a_path)]
            _b = _b[list(b_path)]
            assert _a.shape == _b.shape
            align_a.append(_a)
            align_b.append(_b)
        return align_a, align_b

    h_gt, r_gt = dtw_align_data(hyps, gt_pose)
    mpjper_gt = mpjpe(h_gt, r_gt)

    return mpjper_gt


def bleu(hypotheses: List[str], references: List[str], **sacrebleu_cfg) -> dict:
    """
    Raw corpus BLEU from sacrebleu (without tokenization)
    cf. https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/bleu.py

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return: bleu score
    """
    bleu_scores = raw_corpus_bleu(
        sys_stream=hypotheses, ref_streams=[references]
    ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores["bleu" + str(n + 1)] = bleu_scores[n]
    return scores


def wer(hypotheses: list, references: list):
    hypotheses = copy.deepcopy(hypotheses)
    references = copy.deepcopy(references)
    transforms = jiwer.Compose(
        [
            jiwer.ExpandCommonEnglishContractions(),
            # jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )
    wer = jiwer.wer(
        references,
        hypotheses,
        reference_transform=transforms,
        hypothesis_transform=transforms,
    )
    return wer * 100


def rouge(hypotheses: list, references: list):
    rouge_score = 0
    n_seq = len(hypotheses)

    for h, r in zip(hypotheses, references):
        rouge_score += rouge_calc_score(hypotheses=[h], references=[r]) / n_seq

    return rouge_score * 100


def chrf(hypotheses: List[str], references: List[str], **sacrebleu_cfg) -> float:
    """
    Character F-score from sacrebleu
    cf. https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return: character f-score (0 <= chf <= 1)
             see Breaking Change in sacrebleu v2.0
    """
    kwargs = {}
    if sacrebleu_cfg:
        valid_keys = getfullargspec(CHRF).args
        for k, v in sacrebleu_cfg.items():
            if k in valid_keys:
                kwargs[k] = v

    metric = CHRF(**kwargs)
    score = metric.corpus_score(hypotheses=hypotheses, references=[references]).score

    return score

# ======================================================================
# DTW-MJE  (Dynamic Time Warping - Mean Joint Error)
# ======================================================================
def _to_TJ3(seq):
    """任意の骨格系列を (T, J, 3) の torch.Tensor(float) に整える.
    受理形状: (T, J, 3) / (T, J*3) / (T, 3, J)."""
    if not torch.is_tensor(seq):
        seq = torch.as_tensor(np.asarray(seq))
    seq = seq.float()
    if seq.dim() == 2:                       # (T, J*3)
        T, F = seq.shape
        assert F % 3 == 0, f"feature dim {F} is not divisible by 3"
        seq = seq.reshape(T, F // 3, 3)
    elif seq.dim() == 3 and seq.shape[1] == 3 and seq.shape[2] != 3:
        seq = seq.permute(0, 2, 1)           # (T, 3, J) -> (T, J, 3)
    assert seq.dim() == 3 and seq.shape[-1] == 3, \
        f"expected (T, J, 3)-like, got {tuple(seq.shape)}"
    return seq


def dtw_mje(
    hypotheses: list,
    references: list,
    joint_mask: Optional[list] = None,
    radius: int = 10,
    reduction: str = "mean",
):
    """DTW で時間整列した後の Mean Joint Error (関節位置の平均ユークリッド誤差).

    生成手話は GT と長さ・速度が一致しないため, フレーム単位の MPJPE は
    使えない. DTW で最適整列してから関節誤差を測る. Text-to-Sign 生成
    (Saunders et al., Progressive Transformers 等) の標準指標.

    Args:
        hypotheses: list of 生成系列. 各要素は (T_h, J, 3) 等 (_to_TJ3 が受理する形).
        references: list of GT 系列. 各要素は (T_r, J, 3) 等. T_r != T_h でよい.
        joint_mask: 省略可. list of (J,) bool. 評価に含める関節を指定
                    (例: bones_used_joints で復元対象のみ). 各系列で共通なら1つでも可.
        radius:     fastdtw の探索半径 (大きいほど厳密, 遅い).
        reduction:  "mean" で全系列平均, "none" で系列ごとの誤差リストを返す.

    Returns:
        reduction="mean": float (系列平均 DTW-MJE)
        reduction="none": (errors: list[float], lengths: list[int] 整列後フレーム数)
    """
    def frame_dist(a, b):
        # a, b: (J*3,) numpy. 整列コスト = 関節平均ユークリッド距離
        d = (a - b).reshape(-1, 3)
        return float(np.sqrt((d ** 2).sum(-1)).mean())

    errors, lengths = [], []
    for i, (h, r) in enumerate(
        tqdm(zip(hypotheses, references), total=len(hypotheses),
             desc="DTW-MJE")
    ):
        if h is None or r is None:
            continue
        h = _to_TJ3(h)
        r = _to_TJ3(r)
        if h.shape[0] == 0 or r.shape[0] == 0:
            continue
        assert h.shape[1:] == r.shape[1:], \
            f"joint/coord mismatch: {tuple(h.shape)} vs {tuple(r.shape)}"

        # 関節マスク (指定関節のみで整列・評価)
        if joint_mask is not None:
            m = joint_mask[i] if isinstance(joint_mask, (list, tuple)) \
                else joint_mask
            m = torch.as_tensor(m).bool()
            h, r = h[:, m], r[:, m]

        h_np = h.reshape(h.shape[0], -1).cpu().numpy()
        r_np = r.reshape(r.shape[0], -1).cpu().numpy()

        # DTW 整列 (fastdtw は (距離, パス) を返す)
        dtw_dist, path = dtw(h_np, r_np, radius=radius, dist=frame_dist)
        h_idx, r_idx = zip(*path)
        h_al = h[list(h_idx)]                 # (L, J, 3)
        r_al = r[list(r_idx)]

        # 整列後の関節平均誤差
        je = torch.norm(h_al - r_al, dim=-1).mean().item()
        errors.append(dtw_dist)
        lengths.append(len(path))

    if reduction == "none":
        return errors, lengths
    if len(errors) == 0:
        return float("nan")
    return float(np.mean(errors))


# ======================================================================
# FID  (Frechet Inception Distance) — 骨格運動用
# ======================================================================
def _frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6):
    """2つのガウス N(mu1,sigma1), N(mu2,sigma2) 間の Frechet 距離の2乗.
    ||mu1-mu2||^2 + Tr(s1 + s2 - 2 sqrt(s1 s2))."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        # 特異行列対策: 対角に微小量を加えて再計算
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # 数値誤差由来の虚部を除去
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def _gaussian_stats(feats: np.ndarray):
    """特徴量 (N, D) -> (平均 mu (D,), 共分散 sigma (D, D))."""
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    if sigma.ndim == 0:                       # D=1 のときスカラーになる
        sigma = sigma.reshape(1, 1)
    return mu, sigma


def _default_motion_features(seq):
    """特徴抽出器が無い場合の簡易フォールバック特徴 (1系列 -> (D,)).
    位置・速度・加速度の各関節統計を連結した手作り特徴.
    注意: 学習済みのモーションエンコーダ (行動認識モデル等) がある場合は
    そちらを feature_fn に渡す方が指標として信頼できる (画像FIDの Inception 相当)."""
    x = _to_TJ3(seq)                          # (T, J, 3)
    x = x.reshape(x.shape[0], -1)             # (T, J*3)
    if x.shape[0] < 3:
        vel = torch.zeros_like(x)
        acc = torch.zeros_like(x)
    else:
        vel = x[1:] - x[:-1]
        acc = vel[1:] - vel[:-1]

    def stats(t):                             # 時間方向の平均・標準偏差
        if t.shape[0] == 0:
            d = x.shape[1]
            return torch.zeros(2 * d)
        return torch.cat([t.mean(0), t.std(0)], dim=0)

    return torch.cat([stats(x), stats(vel), stats(acc)], dim=0).cpu().numpy()


def compute_motion_features(
    sequences: list,
    feature_fn: Optional[Callable] = None,
    batched: bool = False,
):
    """系列リスト -> 特徴行列 (N, D).

    Args:
        sequences: list of 骨格系列.
        feature_fn: 1系列 -> (D,) or (D,)相当 を返す特徴抽出器.
                    None なら _default_motion_features (手作り統計特徴) を使用.
                    学習済みモーションエンコーダを渡すのが望ましい.
        batched:    feature_fn がリスト一括入力 (N,)->(N,D) に対応する場合 True.
    """
    fn = feature_fn or _default_motion_features
    if batched:
        feats = np.asarray(fn(sequences))
    else:
        feats = np.stack([np.asarray(fn(s)).reshape(-1)
                          for s in sequences if s is not None])
    return feats


def fid(
    hypotheses: list,
    references: list,
    feature_fn: Optional[Callable] = None,
    batched: bool = False,
    return_stats: bool = False,
):
    """骨格運動の FID (Frechet Distance between real / generated feature dists).

    画像 FID の Inception 特徴を「モーション特徴」に置き換えたもの. 生成分布と
    実分布の特徴空間でのガウス近似間 Frechet 距離. 低いほど良い. 対応する
    ペアは不要 (分布同士の比較なので hypotheses と references の要素数・順序は
    無関係でよい).

    Args:
        hypotheses: 生成系列のリスト.
        references: 実 (GT) 系列のリスト. hypotheses とペアである必要はない.
        feature_fn: モーション特徴抽出器 (None で手作り統計特徴).
                    信頼できる FID には行動認識等の学習済みエンコーダを推奨.
        batched:    feature_fn が一括入力対応なら True.
        return_stats: True で (fid, (mu_h, sig_h), (mu_r, sig_r)) を返す.

    Returns:
        fid_value (float), または上記 return_stats 付きタプル.
    """
    feat_h = compute_motion_features(hypotheses, feature_fn, batched)
    feat_r = compute_motion_features(references, feature_fn, batched)
    if feat_h.shape[0] < 2 or feat_r.shape[0] < 2:
        raise ValueError(
            "FID には各分布に最低2系列必要です "
            f"(got {feat_h.shape[0]} / {feat_r.shape[0]})")

    mu_h, sig_h = _gaussian_stats(feat_h)
    mu_r, sig_r = _gaussian_stats(feat_r)
    value = _frechet_distance(mu_h, sig_h, mu_r, sig_r)
    if return_stats:
        return value, (mu_h, sig_h), (mu_r, sig_r)
    return value
def fid_noextract(
    hypotheses_mu: list,
    hypotheses_logvar: list,
    references_mu: list,
    references_logvar: list,
    feature_fn: Optional[Callable] = None,
    batched: bool = False,
    return_stats: bool = False,
):
    """骨格運動の FID (Frechet Distance between real / generated feature dists).

    画像 FID の Inception 特徴を「モーション特徴」に置き換えたもの. 生成分布と
    実分布の特徴空間でのガウス近似間 Frechet 距離. 低いほど良い. 対応する
    ペアは不要 (分布同士の比較なので hypotheses と references の要素数・順序は
    無関係でよい).

    Args:
        hypotheses: 生成系列の特徴量行列 (N, D) のリスト.
        references: 実 (GT) 系列の特徴量行列 (N, D) のリスト. hypotheses とペアである必要はない.
        batched:    feature_fn が一括入力対応なら True.
        return_stats: True で (fid, (mu_h, sig_h), (mu_r, sig_r)) を返す.

    Returns:
        fid_value (float), または上記 return_stats 付きタプル.
    """
    feat_h_mu= np.asarray(hypotheses_mu)
    feat_r_mu= np.asarray(references_mu)
    feat_h_logvar= np.asarray(hypotheses_logvar)
    feat_r_logvar= np.asarray(references_logvar)
    value = _frechet_distance(feat_h_mu, np.exp(feat_h_logvar), feat_r_mu, np.exp(feat_r_logvar))
    if return_stats:
        return value, (feat_h_mu, np.exp(feat_h_logvar)), (feat_r_mu, np.exp(feat_r_logvar))
    return value
# ======================================================================
# MPJAE  (Mean Per Joint Angle Error) — 関節角度誤差 [degree]
# ======================================================================
def _geodesic_angle_from_matrices(R_pred, R_gt, eps: float = 1e-7):
    """回転行列間の測地線角 [rad]. R_*: (..., 3, 3) -> (...,).
    theta = arccos((tr(R_pred^T R_gt) - 1) / 2)."""
    rel = R_pred.transpose(-1, -2) @ R_gt               # (..., 3, 3)
    tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)         # (...,)
    cos = ((tr - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    return torch.arccos(cos)                            # (...,) rad


def _bone_vectors(pose, bones):
    """座標 (T, J, 3) と bones から各ボーンの単位方向ベクトル (T, Nb, 3) を返す."""
    pose = _to_TJ3(pose)                                # (T, J, 3)
    idx = torch.as_tensor([b[:2] for b in bones]).to(pose.device)  # (Nb, 2)
    vec = pose[:, idx[:, 1], :] - pose[:, idx[:, 0], :]  # (T, Nb, 3)
    return torch.nn.functional.normalize(vec, dim=-1, eps=1e-8)


def mpjae(
    hypotheses: list,
    references: list,
    mode: str = "matrix",
    bones: Optional[list] = None,
    joint_mask: Optional[list] = None,
    align_dtw: bool = False,
    radius: int = 10,
    reduction: str = "mean",
    rot6d_fn: Optional[Callable] = None,
):
    """Mean Per Joint Angle Error — 関節角度の平均誤差 [degree]. 低いほど良い.

    位置指標 (MPJPE/DTW-MJE) と相補的で, ボーンの向き・回転の誤りを捉える.
    手話では手指の向き (palm orientation) が音韻的に重要なため角度指標が効く.

    3 つの入力モード:
      mode="matrix": hypotheses/references が回転行列系列 (T, Nb, 3, 3).
                     ボーン毎の測地線角誤差を測る (最も厳密).
      mode="6d":     6D 回転表現 (T, Nb, 6) or (T, 6, Nb). 内部で回転行列へ復元.
                     ※ rotation_6d_to_matrix をこのモジュールに渡すか, 事前に
                       matrix へ変換して mode="matrix" を使う (下記 rot6d_fn 参照).
      mode="coord":  骨格座標 (T, J, 3) 等. bones からボーン方向ベクトルを作り,
                     hyp と gt のボーン方向の成す角を測る (回転の twist は測れない).

    Args:
        hypotheses, references: 上記モードに応じた系列のリスト. 長さ T は
            align_dtw=False なら hyp/gt で一致が必要, True なら DTW 整列する.
        mode:       "matrix" / "6d" / "coord".
        bones:      mode="coord" で必須. [(parent, child, ...), ...].
        joint_mask: (Nb,) bool. 評価対象のボーン/関節を限定 (list or 単一).
        align_dtw:  True で位置ベース DTW 整列後に角度誤差 (可変長生成向け).
                    整列は座標 (coord) またはボーン方向 (matrix/6d は先頭列) で行う.
        radius:     fastdtw 探索半径.
        reduction:  "mean" or "none".

    Returns:
        reduction="mean": float (平均 MPJAE, degree)
        reduction="none": list[float]
    """
    if mode == "6d":
        if rot6d_fn is None:
            raise ValueError(
                "mode='6d' には rotation_6d_to_matrix が必要です. "
                "metrics.mpjae._rot6d_fn = rotation_6d_to_matrix を設定するか, "
                "事前に matrix へ変換して mode='matrix' を使ってください.")

    def to_matrices(seq):
        """seq -> 回転行列 (T, Nb, 3, 3) の torch.Tensor."""
        t = seq if torch.is_tensor(seq) else torch.as_tensor(np.asarray(seq))
        t = t.float()
        if mode == "matrix":
            assert t.shape[-2:] == (3, 3), \
                f"matrix mode expects (T, Nb, 3, 3), got {tuple(t.shape)}"
            return t
        # 6d: (T, 6, Nb) を (T, Nb, 6) に直してから復元
        if t.dim() == 3 and t.shape[1] == 6 and t.shape[2] != 6:
            t = t.movedim(1, 2)
        return rot6d_fn(t)                       # (T, Nb, 3, 3)

    errors = []
    for i, (h, r) in enumerate(
        tqdm(zip(hypotheses, references), total=len(hypotheses), desc="MPJAE")
    ):
        if h is None or r is None:
            continue

        if mode == "coord":
            assert bones is not None, "mode='coord' には bones が必要"
            h=_to_TJ3(h)
            r=_to_TJ3(r)
            hv = h                # (T_h, Nb, 3)
            rv = r               # (T_r, Nb, 3)
        else:
            hv = to_matrices(h)                         # (T_h, Nb, 3, 3)
            rv = to_matrices(r)

        if hv.shape[0] == 0 or rv.shape[0] == 0:
            continue

        # 時間整列
        if align_dtw:
            # 位置的な整列キー: coord はボーン方向, matrix/6d は第1列 (ボーン向き)
            key_h = hv if mode == "coord" else hv[..., 0]   # (T, Nb, 3)
            key_r = rv if mode == "coord" else rv[..., 0]
            hk = key_h.reshape(key_h.shape[0], -1).cpu().numpy()
            rk = key_r.reshape(key_r.shape[0], -1).cpu().numpy()

            def fdist(a, b):
                return float(np.linalg.norm(a - b))
            _, path = dtw(hk, rk, radius=radius, dist=fdist)
            hi, ri = zip(*path)
            hv, rv = hv[list(hi)], rv[list(ri)]
        else:
            assert hv.shape[0] == rv.shape[0], \
                (f"length mismatch {hv.shape[0]} vs {rv.shape[0]}: "
                 "align_dtw=True を使うか長さを揃えてください")

        # ボーンマスク
        if joint_mask is not None:
            m = joint_mask[i] if isinstance(joint_mask, (list, tuple)) \
                else joint_mask
            m = torch.as_tensor(m).bool()
            hv, rv = hv[:, m], rv[:, m]

        # 角度誤差
        if mode == "coord":
            cos = (hv * rv).sum(-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            ang = torch.arccos(cos)                     # (T, Nb) rad
        else:
            ang = _geodesic_angle_from_matrices(hv, rv)  # (T, Nb) rad

        errors.append(torch.rad2deg(ang).mean().item())

    if reduction == "none":
        return errors
    if len(errors) == 0:
        return float("nan")
    return float(np.mean(errors))


# ======================================================================
# DTW-6D  (Dynamic Time Warping on 6D rotation representation)
# ======================================================================
def _to_TNb6(seq):
    """任意の 6D 回転系列を (T, Nb, 6) の torch.Tensor(float) に整える.
    受理形状: (T, Nb, 6) / (T, 6, Nb) / (T, Nb*6).
    mpjae の 6d 規約 (T,6,Nb -> movedim) と互換."""
    if not torch.is_tensor(seq):
        seq = torch.as_tensor(np.asarray(seq))
    seq = seq.float()
    if seq.dim() == 2:  # (T, Nb*6)
        T, Fdim = seq.shape
        assert Fdim % 6 == 0, f"feature dim {Fdim} is not divisible by 6"
        seq = seq.reshape(T, Fdim // 6, 6)
    elif seq.dim() == 3 and seq.shape[1] == 6 and seq.shape[2] != 6:
        seq = seq.movedim(1, 2)  # (T, 6, Nb) -> (T, Nb, 6)
    assert seq.dim() == 3 and seq.shape[-1] == 6, \
        f"expected (T, Nb, 6)-like, got {tuple(seq.shape)}"
    return seq


def dtw_6d(
        hypotheses: list,
        references: list,
        dist: str = "geodesic",
        bone_mask: Optional[list] = None,
        radius: int = 10,
        reduction: str = "mean",
        rot6d_fn=None,
):
    """6D 回転表現の系列を DTW で時間整列し, 回転誤差を測る.

    生成手話は GT と長さ・速度が一致しないため, DTW で最適整列してから
    誤差を測る (dtw_mje の 6D 回転版). 6D 回転は VAE 潜在の直前表現なので,
    座標に復元せずボーンの回転そのものの誤りを評価できる.

    整列コストと最終誤差は dist で選ぶ:
      dist="geodesic" (推奨): 6D を回転行列へ復元し, 測地線角 [degree] で
          整列・評価. 回転として正しい距離 (6D の冗長表現による歪みがない).
          rot6d_fn (6D->行列) が必要. 未指定なら mpjae._rot6d_fn を探す.
      dist="l2": 6D ベクトルの L2 距離で整列・評価 (復元関数不要, 高速).
          6D 空間の距離は回転角に厳密比例しないが, 目安として使える.

    Args:
        hypotheses: 生成系列のリスト. 各要素は (T_h, Nb, 6) 等 (_to_TNb6 が受理).
        references: GT 系列のリスト. 各要素は (T_r, Nb, 6) 等. T_r != T_h でよい.
        dist:       "geodesic" (角度[deg]) or "l2" (6Dベクトル距離).
        bone_mask:  省略可. (Nb,) bool の list or 単一. 評価対象ボーンを限定.
        radius:     fastdtw 探索半径 (大きいほど厳密, 遅い).
        reduction:  "mean" で全系列平均, "none" で (errors, lengths) を返す.
        rot6d_fn:   6D(...,6)->行列(...,3,3) の関数. dist="geodesic" で必要.
                    None なら metrics.mpjae._rot6d_fn を使う.

    Returns:
        reduction="mean": float
            dist="geodesic" -> 平均角度誤差 [degree]
            dist="l2"       -> 平均 6D-L2 誤差
        reduction="none": (errors: list[float], lengths: list[int])
    """
    if dist == "geodesic":
        if rot6d_fn is None:
            # metrics モジュール内の注入口を探す (循環 import 回避)
            import sys
            m = sys.modules.get("metrics") or sys.modules.get(__name__)
            rot6d_fn = getattr(getattr(m, "mpjae", None), "_rot6d_fn", None)
        if rot6d_fn is None:
           rot6d_fn = mpjae._rot6d_fn

    def geodesic_deg(Ra, Rb):
        # Ra, Rb: (..., 3, 3) 測地線角 [deg]
        rel = Ra.transpose(-1, -2) @ Rb
        tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        return torch.rad2deg(torch.arccos(cos))

    def frame_dist_l2(a, b):
        # a,b: (Nb*6,) numpy. 6D ベクトルの平均 L2
        d = (a - b).reshape(-1, 6)
        return float(np.sqrt((d ** 2).sum(-1)).mean())

    def frame_dist_geo(a, b):
        # a,b: (Nb*6,) numpy -> 行列復元 -> 平均測地線角[deg]
        ta = torch.as_tensor(a, dtype=torch.float32).reshape(-1, 6)
        tb = torch.as_tensor(b, dtype=torch.float32).reshape(-1, 6)
        Ra = rot6d_fn(ta)  # (Nb, 3, 3)
        Rb = rot6d_fn(tb)
        return float(geodesic_deg(Ra, Rb).mean().item())

    frame_dist = frame_dist_geo if dist == "geodesic" else frame_dist_l2

    errors, lengths = [], []
    for i, (h, r) in enumerate(
            tqdm(zip(hypotheses, references), total=len(hypotheses),
                 desc=f"DTW-6D({dist})")
    ):
        if h is None or r is None:
            continue
        h = _to_TNb6(h)
        r = _to_TNb6(r)
        if h.shape[0] == 0 or r.shape[0] == 0:
            continue
        assert h.shape[1:] == r.shape[1:], \
            f"bone/6d mismatch: {tuple(h.shape)} vs {tuple(r.shape)}"

        # ボーンマスク (指定ボーンのみで整列・評価)
        if bone_mask is not None:
            m = bone_mask[i] if isinstance(bone_mask, (list, tuple)) \
                else bone_mask
            m = torch.as_tensor(m).bool()
            h, r = h[:, m], r[:, m]

        h_np = h.reshape(h.shape[0], -1).cpu().numpy()
        r_np = r.reshape(r.shape[0], -1).cpu().numpy()

        # DTW 整列 (フレーム距離は dist に応じて角度 or L2)
        dtw_dist, path = dtw(h_np, r_np, radius=radius, dist=frame_dist)
        h_idx, r_idx = zip(*path)
        h_al = h[list(h_idx)]  # (L, Nb, 6)
        r_al = r[list(r_idx)]

        # 整列後の誤差
        if dist == "geodesic":
            Ra = rot6d_fn(h_al.reshape(-1, 6)).reshape(h_al.shape[0], -1, 3, 3)
            Rb = rot6d_fn(r_al.reshape(-1, 6)).reshape(r_al.shape[0], -1, 3, 3)
            err = geodesic_deg(Ra, Rb).mean().item()  # [deg]
        else:
            err = torch.norm(h_al - r_al, dim=-1).mean().item()  # 6D-L2
        errors.append(dtw_dist)
        lengths.append(len(path))

    if reduction == "none":
        return errors, lengths
    if len(errors) == 0:
        return float("nan")
    return float(np.mean(errors))
def calculate_dtw(references, hypotheses):
    """
    Calculate the DTW costs between a list of references and hypotheses

    :param references: list of reference sequences to compare against
    :param hypotheses: list of hypothesis sequences to fit onto the reference

    :return: dtw_scores: list of DTW costs
    """
    # Euclidean norm is the cost function, difference of coordinates
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    dtw_scores = []


    # Remove the BOS frame from the hypothesis
    #hypotheses = hypotheses[:, 1:]

    # For each reference in the references list
    for i, ref in enumerate(references):
        # Cut the reference down to the max count value
        _ , ref_max_idx = torch.max(ref[:, -1], 0)
        if ref_max_idx == 0: ref_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        ref_count = ref[:ref_max_idx,:-1].cpu().numpy()

        # Cut the hypothesis down to the max count value
        hyp = hypotheses[i]
        print("hypotheses shape:",hyp.shape)
        _, hyp_max_idx = torch.max(hyp[:, -1], 0)
        if hyp_max_idx == 0: hyp_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        hyp_count = hyp[:hyp_max_idx,:-1].cpu().numpy()

        # Calculate DTW of the reference and hypothesis, using euclidean norm
        d, cost_matrix, acc_cost_matrix, path = dtw(ref_count, hyp_count, dist=euclidean_norm)

        # Normalise the dtw cost by sequence length
        d = d/acc_cost_matrix.shape[0]

        dtw_scores.append(d)

    # Return dtw scores and the hypothesis with altered timing
    return dtw_scores

def calculate_dtw_nocounter(references, hypotheses):
  dist = lambda x, y: np.mean(np.abs(x - y))
  scores = []
  for ref, hyp in zip(references, hypotheses):
      d, _, _, path = dtw(ref.cpu().numpy(), hyp.cpu().numpy(), dist=dist)
      scores.append(d / len(path[0]))
  return scores

# mode="6d" 用の 6D->行列 変換器の差し込み口 (循環 import 回避のため属性で注入).
#   例: from models.module.Hand_gcn_vae_6d import rotation_6d_to_matrix
#       metrics.mpjae._rot6d_fn = rotation_6d_to_matrix
mpjae._rot6d_fn = None