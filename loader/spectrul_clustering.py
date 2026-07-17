"""
GAK (Global Alignment Kernel) によるクラスタリング パイプライン
================================================================
(B, T, F) の時系列データセット（例: 身体アンカー正規化済みのキーポイント列）を、
GAK の Gram 行列を経由して spectral clustering / kernel k-means でグループ分けする。

理論上の要点:
  - GAK は min(DTW) を全経路の soft-min(=和) に置き換えた PSD カーネル。
  - tslearn の gak は正規化版で gak(x,x)=1, gak(x,y)∈[0,1]（コサイン正規化済み = 対角優勢対策込み）。
  - PSD なので spectral でも kernel k-means でも理論保証が崩れない。

依存: numpy, scikit-learn, tslearn
  pip install tslearn scikit-learn numpy
"""
from sklearn.metrics import adjusted_rand_score

import numpy as np
from tslearn.metrics import sigma_gak, cdist_gak
from tslearn.utils import to_time_series_dataset
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from tslearn.utils import to_time_series_dataset
from sklearn.cluster import KMeans
def sweep_sigma(X, base):
    for s in base * np.logspace(-1.5, 1.5, 13):
        K = build_gram(X, sigma=s)
        off = K[~np.eye(len(K), dtype=bool)]
        d = K.sum(1); dis = 1/np.sqrt(np.clip(d, 1e-12, None))
        L = np.eye(len(K)) - dis[:, None]*K*dis[None, :]
        gap = np.diff(np.sort(np.linalg.eigvalsh(L))[:6])[1:].max()
        print(f"sigma={s:8.3f}  std/mean={off.std()/off.mean():.3f}  eigengap={gap:.4f}")

# ----------------------------------------------------------------------
# 0. データ準備
# ----------------------------------------------------------------------
def load_dataset():
    """
    返り値 X: tslearn 形式の時系列データセット。
      - 固定長なら ndarray (B, T, F)
      - 可変長なら list[ndarray(T_b, F)] を to_time_series_dataset に通す（NaN パディング）
        GAK は可変長をネイティブに扱えるので長さを無理に揃えなくてよい。
    ここは自分のローダに差し替える。下はダミー。
    """
    rng = np.random.default_rng(0)
    series = []
    for b in range(120):                      # B = 120 本
        T = rng.integers(40, 80)              # 可変長 T
        F = 84                                # 例: 42 keypoints × (x, y)
        series.append(rng.standard_normal((T, F)).cumsum(axis=0))
    X = to_time_series_dataset(series)        # (B, T_max, F), 余白は NaN
    return X

def score_sigma(X, base, n_grid=15, span=(-2.0, 1.0)):
    """各σで (コントラスト, eigengap) を出し、両者を正規化して合成スコア化。"""
    sigmas = base * np.logspace(span[0], span[1], n_grid)
    rows = []
    for s in sigmas:
        K = build_gram(X, sigma=s)
        off = K[~np.eye(len(K), dtype=bool)]
        contrast = off.std() / (off.mean() + 1e-12)
        d = K.sum(1); dis = 1/np.sqrt(np.clip(d, 1e-12, None))
        L = np.eye(len(K)) - dis[:, None]*K*dis[None, :]
        ev = np.sort(np.linalg.eigvalsh(L))[:8]
        eigengap = np.diff(ev)[1:].max()
        rows.append((s, contrast, eigengap))
    S = np.array(rows)
    # コントラストと eigengap をそれぞれ [0,1] 正規化して幾何平均で合成
    def norm(v): return (v - v.min()) / (v.ptp() + 1e-12)
    combined = np.sqrt(norm(S[:,1]) * norm(S[:,2]))
    order = np.argsort(-combined)
    return S, sigmas[order[:3]]
def select_k_for_sigma(K, k_range=range(2, 16)):
    n = K.shape[0]
    d = K.sum(1); dis = 1/np.sqrt(np.clip(d, 1e-12, None))
    L = np.eye(n) - dis[:, None]*K*dis[None, :]
    ev = np.sort(np.linalg.eigvalsh(L))

    # 信号1・3: eigengap（絶対・相対）
    gaps_abs = np.diff(ev[:max(k_range)+2])
    k_abs = int(np.argmax(gaps_abs[1:]) + 2)
    rel = ev[2:max(k_range)+2] / (ev[1:max(k_range)+1] + 1e-9)
    k_rel = int(np.argmax(rel) + 2)

    # 信号2: silhouette
    D = 1.0 - K; np.fill_diagonal(D, 0.0)
    sil = {}
    for k in k_range:
        lab = SpectralClustering(n_clusters=k, affinity="precomputed",
                                 assign_labels="cluster_qr", random_state=0).fit_predict(K)
        if len(np.unique(lab)) < 2: continue
        sil[k] = silhouette_score(D, lab, metric="precomputed")
    k_sil = max(sil, key=sil.get) if sil else None
    return {"eigengap_abs": k_abs, "eigengap_rel": k_rel,
            "silhouette": k_sil, "sil_curve": sil}

def stability(X, sigma, k, n_boot=20, frac=0.8, random_state=0):
    rng = np.random.default_rng(random_state)
    n = len(X); base_lab = None; aris = []
    K_full = build_gram(X, sigma=sigma)
    ref = SpectralClustering(n_clusters=k, affinity="precomputed",
                             random_state=0).fit_predict(K_full)
    for _ in range(n_boot):
        idx = rng.choice(n, size=int(frac*n), replace=False)
        Ksub = K_full[np.ix_(idx, idx)]
        lab = SpectralClustering(n_clusters=k, affinity="precomputed",
                                 random_state=0).fit_predict(Ksub)
        aris.append(adjusted_rand_score(ref[idx], lab))
    return np.mean(aris), np.std(aris)

def joint_select(X, base):
    S, sigma_cands = score_sigma(X, base)
    results = []
    for s in sigma_cands:
        K = build_gram(X, s)
        picks = select_k_for_sigma(K)
        agree = (picks["eigengap_abs"] == picks["silhouette"])   # 合意フラグ
        results.append((s, picks, agree))
        print(f"sigma={s:.3f}  eigengap_k={picks['eigengap_abs']} "
              f"sil_k={picks['silhouette']}  agree={agree}")
    # 合意している組を優先。無ければ silhouette 曲線が最も鋭い σ を採る
    agreed = [(s, p) for s, p, a in results if a]
    return agreed if agreed else results
def finalize_sigma_k(X, base):
    picked = joint_select(X, base)          # 合意リスト or 全結果

    # 返り値の形を (sigma, k) の候補列に正規化する
    candidates = []
    if picked and len(picked[0]) == 2:      # agreed: [(s, picks), ...]
        for s, p in picked:
            candidates.append((s, p["eigengap_abs"]))   # 合意時は eigengap=silhouette
    else:                                   # results: [(s, picks, agree), ...]
        for s, p, _ in picked:
            k = p["silhouette"] or p["eigengap_abs"]     # 合意なしは silhouette 優先
            candidates.append((s, k))

    # 各候補を安定性 ARI で採点し、最良を選ぶ
    best = None
    for s, k in candidates:
        if k is None or k < 2:
            continue
        mean_ari, std_ari = stability(X, s, k)
        print(f"sigma={s:.3f}  k={k}  stability_ARI={mean_ari:.3f}±{std_ari:.3f}")
        if best is None or mean_ari > best[2]:
            best = (s, k, mean_ari)

    if best is None:
        raise ValueError("有効な (sigma, k) 候補なし → 構造が無い可能性。外れ値検出へ。")

    sigma_star, k_star, ari_star = best
    print(f"\n選択: sigma={sigma_star:.4f}, k={k_star} (stability ARI={ari_star:.3f})")
    return sigma_star, k_star
#-----------------------------
# 1. σ 推定（オーバーフロー対策つき）
# ----------------------------------------------------------------------
def estimate_sigma(X, n_samples=200, shrink_if_long=True, random_state=0):
    """
    sigma_gak: フレーム対距離の中央値 × sqrt(系列長) × 定数 のヒューリスティック。
    長い系列だと σ が大きくなり log-sum-exp が膨らんで overflow しうるので、
    その場合は控えめに縮める。最終的には silhouette でこの周辺を探索するのが安全。
    """

    sigma = sigma_gak(X, n_samples=n_samples, random_state=random_state)
    #sweep_sigma(X,sigma)
    T_med = np.median([np.sum(~np.isnan(x).any(axis=-1)) for x in X])
    if shrink_if_long and T_med > 60:
        sigma *= 0.5                          # 長系列は控えめに
    print(f"[sigma] estimated={sigma:.4f} (median length={T_med:.0f})")
    return float(sigma)


# ----------------------------------------------------------------------
# 2. GAK Gram 行列を一度だけ計算（k 探索で使い回す）
# ----------------------------------------------------------------------
def build_gram(X, sigma, n_jobs=-1):
    """K: (B, B) 正規化済み GAK 行列。diag≈1, 値∈[0,1], 対称 PSD。"""
    K = cdist_gak(X, sigma=sigma, n_jobs=n_jobs)
    K = 0.5 * (K + K.T)                       # 数値誤差で生じうる非対称を均す
    np.fill_diagonal(K, 1.0)
    return K


# ----------------------------------------------------------------------
# 3. eigengap で k 候補を出す
# ----------------------------------------------------------------------
def eigengap_suggestion(K, k_max=10):
    """正規化ラプラシアン L_sym = I - D^{-1/2} K D^{-1/2} の固有値ギャップから k を示唆。"""
    d = K.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.clip(d, 1e-12, None))
    L = np.eye(K.shape[0]) - (d_inv_sqrt[:, None] * K * d_inv_sqrt[None, :])
    eigvals = np.sort(np.linalg.eigvalsh(L))[:k_max + 1]
    gaps = np.diff(eigvals)
    if len(gaps)<=1:
        k_star = 2
    else:
        k_star = int(np.argmax(gaps[1:]) + 2)     # k=1 は除外して最大ギャップ
    print(f"[eigengap] smallest eigenvalues={np.round(eigvals[:k_max], 4)}")
    print(f"[eigengap] suggested k={k_star}")
    return k_star


# ----------------------------------------------------------------------
# 4. spectral clustering（precomputed affinity = K）
# ----------------------------------------------------------------------
def cluster_spectral(K, k, random_state=0):
    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",               # K をそのまま親和性行列として使う
        assign_labels="cluster_qr",           # kmeans より安定なことが多い
        random_state=random_state,
    )
    return sc.fit_predict(K)


# ----------------------------------------------------------------------
# 5. silhouette で k を選ぶ（距離 = 1 - K）
# ----------------------------------------------------------------------
def select_k(K, k_grid):
    """各 k で spectral を回し、precomputed silhouette が最大の k を選ぶ。"""
    D = 1.0 - K                               # diag=0, 非類似度として使用
    np.fill_diagonal(D, 0.0)
    best = (None, -1.0, None)
    for k in k_grid:
        #labels = cluster_spectral(K, k)
        labels, T, centroids_embed, reps = spectral_with_centroids(K, k)
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(D, labels, metric="precomputed")
        print(f"[select_k] k={k:2d}  silhouette={s:.4f}")
        if s > best[1]:
            best = (k, s, labels)
    print(f"[select_k] -> best k={best[0]} (silhouette={best[1]:.4f})")
    return best[0], best[2]


# ----------------------------------------------------------------------
# 6. 評価
# ----------------------------------------------------------------------
def evaluate(K, labels, y_true=None):
    D = 1.0 - K
    np.fill_diagonal(D, 0.0)
    print(f"[eval] silhouette(1-K)={silhouette_score(D, labels, metric='precomputed'):.4f}")
    if y_true is not None:                    # gloss など一部ラベルがあれば外部指標も
        print(f"[eval] ARI={adjusted_rand_score(y_true, labels):.4f}")
        print(f"[eval] NMI={normalized_mutual_info_score(y_true, labels):.4f}")


# ----------------------------------------------------------------------
# 大規模 B 向け: Nyström 近似（B^2 を B*m に削減）
# ----------------------------------------------------------------------
def nystrom_embedding(X, sigma, n_landmarks=64, random_state=0, n_jobs=-1):
    """
    ランドマーク m 本だけ厳密に GAK を計算し、spectral 埋め込みを近似。
    返り値 Z は (B, m) の特徴で、これに普通の KMeans をかければよい。
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(n_landmarks, len(X)), replace=False)
    X_land = X[idx]
    K_bm = cdist_gak(X, X_land, sigma=sigma, n_jobs=n_jobs)      # (B, m)
    K_mm = cdist_gak(X_land, sigma=sigma, n_jobs=n_jobs)         # (m, m)
    K_mm = 0.5 * (K_mm + K_mm.T)
    # K_mm^{-1/2}（固有分解、負の数値ノイズはクリップ）
    w, V = np.linalg.eigh(K_mm)
    w = np.clip(w, 1e-10, None)
    K_mm_inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    Z = K_bm @ K_mm_inv_sqrt                                     # (B, m)
    return Z

def cluster_medoids(K, labels):
    diagK = np.diag(K); medoids = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        sub = K[np.ix_(idx, idx)]
        cross  = sub.sum(1) / len(idx)          # ⟨φ(x_i), m_c⟩  (i∈c)
        normsq = sub.sum() / len(idx)**2        # ‖m_c‖²
        dist2  = diagK[idx] - 2*cross + normsq
        medoids[c] = idx[dist2.argmin()]        # 重心に最も近い実サンプル
    return medoids
def spectral_with_centroids(K, k, random_state=0):
    d   = K.sum(1)
    dis = 1.0 / np.sqrt(np.clip(d, 1e-12, None))
    L   = np.eye(K.shape[0]) - dis[:, None] * K * dis[None, :]   # L_sym
    w, V = np.linalg.eigh(L)
    U = V[:, :k]                                                 # 最小k固有ベクトル
    T = U / np.clip(np.linalg.norm(U, axis=1, keepdims=True), 1e-12, None)  # 行正規化(NJW)

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(T)
    labels          = km.labels_
    centroids_embed = km.cluster_centers_       # ← 埋め込み空間の重心（明示的・k次元）

    # 重心に最も近い実サンプル（= 元系列の代表）
    reps = {}
    for c in range(k):
        idx = np.where(labels == c)[0]
        reps[c] = idx[np.linalg.norm(T[idx] - centroids_embed[c], axis=1).argmin()]
    return labels, T, centroids_embed, reps
# ----------------------------------------------------------------------
# 実行
# ----------------------------------------------------------------------
if __name__ == "__main__":
    X = load_dataset()
    print(f"dataset: B={len(X)}  T_max={X.shape[1]}  F={X.shape[2]}")

    sigma = estimate_sigma(X)

    # --- 標準ルート: Gram を1回計算 → eigengap で当たり → silhouette で k 確定 ---
    K = build_gram(X, sigma)
    k_hint = eigengap_suggestion(K, k_max=10)
    k_grid = sorted(set(range(2, 11)) | {k_hint})
    k_best, labels = select_k(K, k_grid)
    evaluate(K, labels, y_true=None)          # gloss ラベルがあれば y_true= に渡す

    # --- 最終確認に kernel k-means を併用したい場合 ---
    from tslearn.clustering import KernelKMeans
    km = KernelKMeans(n_clusters=k_best, kernel="gak",
                       kernel_params={"sigma": sigma}, random_state=0)
    labels_kkm = km.fit_predict(X)

    # --- B が大きいとき: Nyström + KMeans ---
    # from sklearn.cluster import KMeans
    # Z = nystrom_embedding(X, sigma, n_landmarks=64)
    # labels_ny = KMeans(n_clusters=k_best, n_init=10, random_state=0).fit_predict(Z)

    print("done. labels:", labels[:20], "...")
    print("done. labels_kkm:", labels_kkm[:20], "...")
