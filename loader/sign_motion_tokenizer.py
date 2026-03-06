import numpy as np
import os
EPS = 1e-8

# ----------------------------
# Basic utils
# ----------------------------
def safe_norm(x, axis=-1, keepdims=False, eps=EPS):
    return np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims) + eps)

def safe_normalize(x, axis=-1, eps=EPS):
    n = safe_norm(x, axis=axis, keepdims=True, eps=eps)
    return x / n

def rolling_mean_1d(x, win=3):
    """x: (T, d)"""
    if win <= 1:
        return x
    T = x.shape[0]
    pad = win // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(x)
    for t in range(T):
        out[t] = xp[t:t+win].mean(axis=0)
    return out

def make_valid_mask_from_hand(hand_xyz):
    """
    hand_xyz: (T,21,3)
    Return valid: (T,) bool, True if not all zeros.
    """
    return ~(np.all(hand_xyz == 0, axis=(1,2)))

# ----------------------------
# KMeans (numpy)
# ----------------------------
def kmeans_pp_init(X, k, rng, eps=1e-12):
    """Robust k-means++ init. X:(N,D)"""
    N, D = X.shape
    centers = np.zeros((k, D), dtype=np.float32)

    # 1) pick first center
    i0 = int(rng.integers(0, N))
    centers[0] = X[i0]

    # 2) initial dist^2
    dist2 = np.sum((X - centers[0]) ** 2, axis=1).astype(np.float64)
    dist2 = np.clip(dist2, 0.0, np.inf)

    for ci in range(1, k):
        s = float(dist2.sum())

        # If all points are identical to current centers (or numerical issues),
        # fall back to uniform sampling.
        if (not np.isfinite(s)) or s <= eps:
            idx = int(rng.integers(0, N))
            centers[ci] = X[idx]
            # update dist2
            new_dist2 = np.sum((X - centers[ci]) ** 2, axis=1).astype(np.float64)
            new_dist2 = np.clip(new_dist2, 0.0, np.inf)
            dist2 = np.minimum(dist2, new_dist2)
            continue

        probs = dist2 / s
        # Numerical cleanup (important!)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        ps = float(probs.sum())
        if ps <= eps:
            idx = int(rng.integers(0, N))
        else:
            probs = probs / ps
            idx = int(rng.choice(N, p=probs))

        centers[ci] = X[idx]
        new_dist2 = np.sum((X - centers[ci]) ** 2, axis=1).astype(np.float64)
        new_dist2 = np.clip(new_dist2, 0.0, np.inf)
        dist2 = np.minimum(dist2, new_dist2)

    return centers.astype(np.float32)

def kmeans_fit(X, k, max_iter=50, seed=0, tol=1e-4):
    """
    X: (N,D) float32
    returns centers: (k,D)
    """
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32)
    centers = kmeans_pp_init(X, k, rng)
    prev_inertia = None

    for it in range(max_iter):
        # assign
        # dist^2 = ||x||^2 + ||c||^2 - 2 x c
        x2 = np.sum(X*X, axis=1, keepdims=True)          # (N,1)
        c2 = np.sum(centers*centers, axis=1, keepdims=True).T  # (1,k)
        dist2 = x2 + c2 - 2.0 * (X @ centers.T)          # (N,k)
        labels = np.argmin(dist2, axis=1)

        # update
        new_centers = np.zeros_like(centers)
        counts = np.zeros((k,), dtype=np.int64)
        for j in range(k):
            m = (labels == j)
            counts[j] = int(m.sum())
            if counts[j] > 0:
                new_centers[j] = X[m].mean(axis=0)
            else:
                # re-seed empty cluster
                new_centers[j] = X[rng.integers(0, X.shape[0])]

        centers = new_centers

        inertia = float(np.mean(np.min(dist2, axis=1)))
        if prev_inertia is not None and abs(prev_inertia - inertia) < tol * max(1.0, prev_inertia):
            break
        prev_inertia = inertia

    return centers

def kmeans_predict(X, centers):
    """X: (N,D), centers:(k,D) -> labels:(N,)"""
    X = X.astype(np.float32)
    centers = centers.astype(np.float32)
    x2 = np.sum(X*X, axis=1, keepdims=True)
    c2 = np.sum(centers*centers, axis=1, keepdims=True).T
    dist2 = x2 + c2 - 2.0 * (X @ centers.T)
    return np.argmin(dist2, axis=1).astype(np.int64)

# ----------------------------
# Feature builders (factorized)
# ----------------------------
def compute_shoulder_mid_and_scale(pose_xyz, left_sh_id=11, right_sh_id=12):
    """
    pose_xyz: (T,33,3)
    return mid:(T,3), scale:(T,1) where scale ~ shoulder width
    """
    pL = pose_xyz[:, left_sh_id, :]
    pR = pose_xyz[:, right_sh_id, :]
    mid = 0.5 * (pL + pR)
    scale = safe_norm(pL - pR, axis=-1, keepdims=True)  # (T,1)
    scale = np.clip(scale, 1e-3, None)
    return mid, scale

def handshape_feat(hand_xyz, valid, use_xy_only=False):
    """
    hand_xyz: (T,21,3) centered already (mid subtracted)
    valid: (T,) bool
    Return H: (T, Dhs) float32, invalid rows = 0
    Handshape = (hand - wrist) / palm_scale  flattened
    """
    T = hand_xyz.shape[0]
    wrist = hand_xyz[:, 0, :]  # (T,3)
    rel = hand_xyz - wrist[:, None, :]  # (T,21,3)

    if use_xy_only:
        rel = rel[..., :2]  # (T,21,2)

    # palm scale ~ ||index_mcp - pinky_mcp||
    # mediapipe: index_mcp=5, pinky_mcp=17
    palm = hand_xyz[:, 5, :] - hand_xyz[:, 17, :]
    palm_s = safe_norm(palm, axis=-1, keepdims=True)  # (T,1)
    palm_s = np.clip(palm_s, 1e-3, None)

    reln = rel / palm_s[:, None, :]  # broadcast
    H = reln.reshape(T, -1).astype(np.float32)  # (T, 21*2 or 21*3)

    H[~valid] = 0.0
    return H

def location_feat(wrist_L, wrist_R, valid_L, valid_R, scale, use_xy_only=False):
    """
    wrist_*: (T,3) already centered
    scale: (T,1) shoulder width
    Return (T, Dloc) where Dloc = 6 or 4
    """
    if use_xy_only:
        wL = wrist_L[:, :2] / scale
        wR = wrist_R[:, :2] / scale
        feat = np.concatenate([wL, wR], axis=-1)  # (T,4)
    else:
        wL = wrist_L / scale
        wR = wrist_R / scale
        feat = np.concatenate([wL, wR], axis=-1)  # (T,6)

    # invalid -> 0
    feat = feat.astype(np.float32)
    feat[~valid_L, 0:(feat.shape[1]//2)] = 0.0
    feat[~valid_R, (feat.shape[1]//2):] = 0.0
    return feat

def movement_feat(wrist, valid, fps=25, smooth_win=3, use_xy_only=True):
    """
    wrist: (T,3) centered
    Return (T, Dmov) where Dmov = 3 (dx,dy,speed) if xy, else 4 (dx,dy,dz,speed)
    - Use delta per frame; include speed scalar
    """
    w = wrist[:, :2] if use_xy_only else wrist  # (T,2 or 3)
    w = rolling_mean_1d(w.astype(np.float32), win=smooth_win)
    T = w.shape[0]
    d = np.zeros_like(w, dtype=np.float32)
    # compute delta only when consecutive valid
    vmask = valid[1:] & valid[:-1]
    if np.any(vmask):
        d[1:][vmask] = (w[1:][vmask] - w[:-1][vmask]).astype(np.float32)
    speed = safe_norm(d, axis=-1, keepdims=True).astype(np.float32)  # (T,1)

    # normalize speed by fps? (optional)
    # Here keep per-frame displacement; you can multiply by fps if you want per-second.
    feat = np.concatenate([d, speed], axis=-1)  # (T, 2+1 or 3+1)

    feat[~valid] = 0.0
    return feat

def relation_feat(wL, wR, vL, vR, valid_L, valid_R, scale, use_xy_only=True):
    """
    wL,wR: (T,3) centered
    vL,vR: (T,2 or 3) deltas (same basis as movement_feat without speed)
    Return (T, Drel) small.
      - distance between wrists
      - relative vector (normalized)
      - speed similarity (abs diff)
      - sync (||vL - vR||)
      - hold_L/hold_R (optional outside)
    """
    both = valid_L & valid_R
    if use_xy_only:
        a = wL[:, :2] / scale
        b = wR[:, :2] / scale
    else:
        a = wL / scale
        b = wR / scale

    rel = np.zeros_like(a, dtype=np.float32)
    dist = np.zeros((a.shape[0], 1), dtype=np.float32)
    if np.any(both):
        rel[both] = (a[both] - b[both]).astype(np.float32)
        dist[both, 0] = safe_norm(rel[both], axis=-1, keepdims=False).astype(np.float32)

    rel_dir = safe_normalize(rel, axis=-1).astype(np.float32)  # (T,2 or 3)

    # speed stats from deltas
    sL = safe_norm(vL, axis=-1, keepdims=False).astype(np.float32)
    sR = safe_norm(vR, axis=-1, keepdims=False).astype(np.float32)
    sdiff = np.abs(sL - sR).reshape(-1, 1).astype(np.float32)

    sync = safe_norm((vL - vR).astype(np.float32), axis=-1, keepdims=True).astype(np.float32)

    feat = np.concatenate([dist, rel_dir, sdiff, sync], axis=-1)  # (T, 1+(2/3)+1+1)
    feat[~both] = 0.0
    return feat

def hold_indicator_from_speed(speed, valid, thresh):
    """
    speed: (T,1) or (T,) displacement magnitude
    """
    s = speed.reshape(-1)
    hold = (s < thresh) & valid
    return hold.astype(np.float32)
def body_location_feat(skel_c, scale, body_joint_ids, use_xy_only=True):
    """
    skel_c: (T,J,3) すでに shoulder-mid 原点で中心化済み
    scale:  (T,1)   shoulder width 等
    body_joint_ids: tuple/list of joint indices to use (e.g., (0,11,12,13,14))
    Return:
      f_body: (T, Db)  Db = len(ids)*2 (xy) or len(ids)*3 (xyz)
    """
    pts = skel_c[:, body_joint_ids, :]  # (T,nb,3)
    if use_xy_only:
        pts = pts[..., :2]              # (T,nb,2)
    # normalize by scale
    pts = pts / scale[:, None, :]       # broadcast
    f_body = pts.reshape(pts.shape[0], -1).astype(np.float32)
    return f_body
# ----------------------------
# Motion Tokenizer (fit + encode)
# ----------------------------
class SignMotionTokenizer:
    def __init__(
        self,
        k_bodyloc=64,          # ★追加
        k_loc=64,
        k_mov=64,
        k_hs=128,
        k_rel=32,
        fps=25,
        use_xy_only=True,
        hold_speed_thresh=0.01,
        seed=0,
        # ★追加：bodyに使う関節（あなたの skeleton に合わせて変える）
        body_joint_ids=(0, 11, 12, 13, 14),
        left_sh_id=11,
        right_sh_id=12

    ):
        self.k_bodyloc = k_bodyloc
        self.k_loc = k_loc
        self.k_mov = k_mov
        self.k_hs = k_hs
        self.k_rel = k_rel
        self.fps = fps
        self.use_xy_only = use_xy_only
        self.hold_speed_thresh = hold_speed_thresh
        self.seed = seed
        self.body_joint_ids = tuple(body_joint_ids)
        self.left_sh_id = left_sh_id
        self.right_sh_id = right_sh_id

        # codebooks
        self.cb_bodyloc = None   # ★追加
        self.cb_loc = None
        self.cb_mov = None
        self.cb_hs = None
        self.cb_rel = None

    def _extract_features_one(self, pose_xyz, left_hand, right_hand):
        """
        Inputs:
          pose_xyz: (T,33,3)
          left_hand,right_hand: (T,21,3)
        Returns dict of per-frame features and masks.
        """
        pose_xyz = np.asarray(pose_xyz, dtype=np.float32)
        left_hand = np.asarray(left_hand, dtype=np.float32)
        right_hand = np.asarray(right_hand, dtype=np.float32)
        T = pose_xyz.shape[0]

        mid, scale = compute_shoulder_mid_and_scale(pose_xyz,left_sh_id=self.left_sh_id,right_sh_id=self.right_sh_id) # (T,3), (T,1)

        # center
        pose_c = pose_xyz - mid[:, None, :]
        Lc = left_hand - mid[:, None, :]
        Rc = right_hand - mid[:, None, :]

        valid_L = make_valid_mask_from_hand(left_hand)
        valid_R = make_valid_mask_from_hand(right_hand)

        wL = Lc[:, 0, :]
        wR = Rc[:, 0, :]

        # location
        f_loc = location_feat(wL, wR, valid_L, valid_R, scale, use_xy_only=self.use_xy_only)

        # movement (need deltas + speed)
        f_mov_L = movement_feat(wL, valid_L, fps=self.fps, use_xy_only=self.use_xy_only)  # (T,2+1) or (T,3+1)
        f_mov_R = movement_feat(wR, valid_R, fps=self.fps, use_xy_only=self.use_xy_only)
        # skel_c: (T,J,3) centered by shoulder-mid
        # scale:  (T,1)

        # split delta and speed for relation
        dL = f_mov_L[:, :-1]
        dR = f_mov_R[:, :-1]
        sL = f_mov_L[:, -1:]
        sR = f_mov_R[:, -1:]

        # handshape
        f_hs_L = handshape_feat(Lc, valid_L, use_xy_only=self.use_xy_only)
        f_hs_R = handshape_feat(Rc, valid_R, use_xy_only=self.use_xy_only)

        # relation
        f_rel = relation_feat(wL, wR, dL, dR, valid_L, valid_R, scale, use_xy_only=self.use_xy_only)

        # hold indicators (optional tokens or extra features)
        hold_L = hold_indicator_from_speed(sL, valid_L, self.hold_speed_thresh)
        hold_R = hold_indicator_from_speed(sR, valid_R, self.hold_speed_thresh)

        f_body = body_location_feat(
            skel_c=pose_c,
            scale=scale,
            body_joint_ids=self.body_joint_ids,
            use_xy_only=self.use_xy_only,
        )
        return {
            "bodyloc": f_body,  # ★追加
            "loc": f_loc,
            "mov_L": f_mov_L,
            "mov_R": f_mov_R,
            "hs_L": f_hs_L,
            "hs_R": f_hs_R,
            "rel": f_rel,
            "valid_L": valid_L,
            "valid_R": valid_R,
            "hold_L": hold_L,
            "hold_R": hold_R,
        }
    def fit_codebooks(self, dataset, sample_frames=200000, max_iter=50):
        """
        dataset: iterable of dicts or tuples providing (pose_xyz, left_hand, right_hand)
          - if dict: keys {"pose","left","right"}
          - if tuple: (pose,left,right)
        sample_frames: subsample total frames across dataset for speed
        """
        rng = np.random.default_rng(self.seed)

        X_bodyloc = []  # ★追加
        X_loc, X_mov, X_hs, X_rel = [], [], [], []
        total = 0

        for item in dataset:
            if isinstance(item, dict):
                pose, L, R = item["pose"], item["left"], item["right"]
            else:
                pose, L, R = item

            feats = self._extract_features_one(pose, L, R)
            T = feats["loc"].shape[0]

            # choose frames to sample
            if total >= sample_frames:
                break

            # sample indices uniformly
            n_take = min(T, max(1, int(sample_frames / max(1, len(X_loc)+1))))
            idx = rng.choice(T, size=min(n_take, T), replace=False)

            # collect valid-ish rows (optional: filter by masks)
            X_loc.append(feats["loc"][idx])
            X_mov.append(np.concatenate([feats["mov_L"][idx], feats["mov_R"][idx]], axis=-1))  # joint mov
            X_hs.append(np.concatenate([feats["hs_L"][idx], feats["hs_R"][idx]], axis=-1))     # joint hs
            X_rel.append(feats["rel"][idx])
            X_bodyloc.append(feats["bodyloc"][idx])  # ★追加

            total += len(idx)

        X_bodyloc = np.concatenate(X_bodyloc, axis=0).astype(np.float32)  # ★追加
        X_loc = np.concatenate(X_loc, axis=0).astype(np.float32)
        X_mov = np.concatenate(X_mov, axis=0).astype(np.float32)
        X_hs = np.concatenate(X_hs, axis=0).astype(np.float32)
        X_rel = np.concatenate(X_rel, axis=0).astype(np.float32)


        # (optional) standardize features lightly
        def standardize(X):
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True) + 1e-4
            return (X - mu) / sd, (mu, sd)

            # standardize
        X_bodyloc_s, self.bodyloc_stats = standardize(X_bodyloc)  # ★追加
        X_loc_s, self.loc_stats = standardize(X_loc)
        X_mov_s, self.mov_stats = standardize(X_mov)
        X_hs_s, self.hs_stats = standardize(X_hs)
        X_rel_s, self.rel_stats = standardize(X_rel)

        # fit kmeans
        self.cb_bodyloc = kmeans_fit(X_bodyloc_s, self.k_bodyloc, max_iter=max_iter, seed=self.seed + 0)  # ★追加
        self.cb_loc = kmeans_fit(X_loc_s, self.k_loc, max_iter=max_iter, seed=self.seed + 1)
        self.cb_mov = kmeans_fit(X_mov_s, self.k_mov, max_iter=max_iter, seed=self.seed + 2)
        self.cb_hs = kmeans_fit(X_hs_s, self.k_hs, max_iter=max_iter, seed=self.seed + 3)
        self.cb_rel = kmeans_fit(X_rel_s, self.k_rel, max_iter=max_iter, seed=self.seed + 4)
        return {
            "bodyloc_centers": self.cb_bodyloc,  # ★追加
            "loc_centers": self.cb_loc,
            "mov_centers": self.cb_mov,
            "hs_centers": self.cb_hs,
            "rel_centers": self.cb_rel,
        }

    def encode(self, pose_xyz, left_hand, right_hand, pad_id=-1, return_features=False):
        """
        Return tokens per frame:
          loc_id: (T,)
          mov_id: (T,)  (joint L+R movement token)
          hs_id:  (T,)  (joint L+R handshape token)
          rel_id: (T,)
          hold_L, hold_R: (T,) float(0/1)  (optional auxiliary)
        """
        assert self.cb_loc is not None, "Call fit_codebooks first"

        feats = self._extract_features_one(pose_xyz, left_hand, right_hand)

        # standardize
        def apply_stats(X, stats):
            mu, sd = stats
            return (X - mu) / sd

        X_loc = apply_stats(feats["loc"], self.loc_stats)
        X_mov = apply_stats(np.concatenate([feats["mov_L"], feats["mov_R"]], axis=-1), self.mov_stats)
        X_hs  = apply_stats(np.concatenate([feats["hs_L"], feats["hs_R"]], axis=-1), self.hs_stats)
        X_rel = apply_stats(feats["rel"], self.rel_stats)
        X_bodyloc = apply_stats(feats["bodyloc"], self.bodyloc_stats)  # ★追加

        bodyloc_id = kmeans_predict(X_bodyloc, self.cb_bodyloc)  # ★追加
        loc_id = kmeans_predict(X_loc, self.cb_loc)
        mov_id = kmeans_predict(X_mov, self.cb_mov)
        hs_id  = kmeans_predict(X_hs,  self.cb_hs)
        rel_id = kmeans_predict(X_rel, self.cb_rel)

        # You can optionally mask invalid frames with pad_id:
        # - loc/mov/hs: if one hand invalid, still defined (zeros). If you prefer pad, do it here.
        # - rel: only meaningful when both valid; you may pad when not both.
        both = feats["valid_L"] & feats["valid_R"]
        rel_id = np.where(both, rel_id, pad_id).astype(np.int64)

        out = {
            "bodyloc": bodyloc_id.astype(np.int64),  # ★追加
            "loc": loc_id.astype(np.int64),
            "mov": mov_id.astype(np.int64),
            "hs": hs_id.astype(np.int64),
            "rel": rel_id,
            "valid_L": feats["valid_L"],
            "valid_R": feats["valid_R"],
            "hold_L": feats["hold_L"].astype(np.float32),
            "hold_R": feats["hold_R"].astype(np.float32),
        }
        if return_features:
            out["features"] = feats
        return out

    def _require_trained(self):
        """Check if codebooks exist."""
        if self.cb_loc is None or self.cb_mov is None or self.cb_rel is None:
            raise RuntimeError("Tokenizer is not trained. Call fit_codebooks() before saving.")

    @staticmethod
    def _pack_stats(stats):
        """
        stats: (mu, sd) each (1,D) float arrays
        return mu, sd
        """
        if stats is None:
            return None, None
        mu, sd = stats
        return np.asarray(mu, dtype=np.float32), np.asarray(sd, dtype=np.float32)

    @staticmethod
    def _unpack_stats(mu, sd):
        if mu is None or sd is None:
            return None
        return (np.asarray(mu, dtype=np.float32), np.asarray(sd, dtype=np.float32))

    def save(self, path: str):
        """
        Save trained tokenizer to a .npz file.

        Saved:
          - config (k_* etc.)
          - codebooks (centers)
          - standardize stats (mu, sd) for each feature
        """
        self._require_trained()

        # config
        cfg = dict(
            k_bodyloc=int(getattr(self, "k_bodyloc", 0)),
            k_loc=int(self.k_loc),
            k_mov=int(self.k_mov),
            k_hs=int(getattr(self, "k_hs", 0)),
            k_rel=int(self.k_rel),
            fps=int(self.fps),
            use_xy_only=bool(self.use_xy_only),
            hold_speed_thresh=float(self.hold_speed_thresh),
            seed=int(self.seed),
            body_joint_ids=np.array(getattr(self, "body_joint_ids", ()), dtype=np.int32),
        )

        # stats
        bodyloc_mu, bodyloc_sd = self._pack_stats(getattr(self, "bodyloc_stats", None))
        loc_mu, loc_sd = self._pack_stats(getattr(self, "loc_stats", None))
        mov_mu, mov_sd = self._pack_stats(getattr(self, "mov_stats", None))
        hs_mu, hs_sd = self._pack_stats(getattr(self, "hs_stats", None))
        rel_mu, rel_sd = self._pack_stats(getattr(self, "rel_stats", None))

        # codebooks
        arrs = {
            # config
            "cfg_k_bodyloc": np.array(cfg["k_bodyloc"], dtype=np.int32),
            "cfg_k_loc": np.array(cfg["k_loc"], dtype=np.int32),
            "cfg_k_mov": np.array(cfg["k_mov"], dtype=np.int32),
            "cfg_k_hs": np.array(cfg["k_hs"], dtype=np.int32),
            "cfg_k_rel": np.array(cfg["k_rel"], dtype=np.int32),
            "cfg_fps": np.array(cfg["fps"], dtype=np.int32),
            "cfg_use_xy_only": np.array(int(cfg["use_xy_only"]), dtype=np.int32),
            "cfg_hold_speed_thresh": np.array(cfg["hold_speed_thresh"], dtype=np.float32),
            "cfg_seed": np.array(cfg["seed"], dtype=np.int32),
            "cfg_body_joint_ids": cfg["body_joint_ids"],

            # centers (some may be None depending on your setup)
            "cb_bodyloc": np.asarray(getattr(self, "cb_bodyloc", None)) if getattr(self, "cb_bodyloc",
                                                                                   None) is not None else np.array([],
                                                                                                                   dtype=np.float32),
            "cb_loc": np.asarray(self.cb_loc, dtype=np.float32),
            "cb_mov": np.asarray(self.cb_mov, dtype=np.float32),
            "cb_hs": np.asarray(getattr(self, "cb_hs", None)) if getattr(self, "cb_hs", None) is not None else np.array(
                [], dtype=np.float32),
            "cb_rel": np.asarray(self.cb_rel, dtype=np.float32),

            # stats
            "bodyloc_mu": bodyloc_mu if bodyloc_mu is not None else np.array([], dtype=np.float32),
            "bodyloc_sd": bodyloc_sd if bodyloc_sd is not None else np.array([], dtype=np.float32),
            "loc_mu": loc_mu if loc_mu is not None else np.array([], dtype=np.float32),
            "loc_sd": loc_sd if loc_sd is not None else np.array([], dtype=np.float32),
            "mov_mu": mov_mu if mov_mu is not None else np.array([], dtype=np.float32),
            "mov_sd": mov_sd if mov_sd is not None else np.array([], dtype=np.float32),
            "hs_mu": hs_mu if hs_mu is not None else np.array([], dtype=np.float32),
            "hs_sd": hs_sd if hs_sd is not None else np.array([], dtype=np.float32),
            "rel_mu": rel_mu if rel_mu is not None else np.array([], dtype=np.float32),
            "rel_sd": rel_sd if rel_sd is not None else np.array([], dtype=np.float32),
        }

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, **arrs)

    @classmethod
    def load(cls, path: str):
        """
        Load tokenizer from .npz and return a SignMotionTokenizer instance.
        """
        z = np.load(path, allow_pickle=False)

        # read config
        k_bodyloc = int(z["cfg_k_bodyloc"])
        k_loc = int(z["cfg_k_loc"])
        k_mov = int(z["cfg_k_mov"])
        k_hs = int(z["cfg_k_hs"])
        k_rel = int(z["cfg_k_rel"])
        fps = int(z["cfg_fps"])
        use_xy_only = bool(int(z["cfg_use_xy_only"]))
        hold_speed_thresh = float(z["cfg_hold_speed_thresh"])
        seed = int(z["cfg_seed"])
        body_joint_ids = tuple(z["cfg_body_joint_ids"].astype(np.int32).tolist())

        # instantiate
        tok = cls(
            k_bodyloc=k_bodyloc if k_bodyloc > 0 else 0,
            k_loc=k_loc,
            k_mov=k_mov,
            k_hs=k_hs if k_hs > 0 else 0,
            k_rel=k_rel,
            fps=fps,
            use_xy_only=use_xy_only,
            hold_speed_thresh=hold_speed_thresh,
            seed=seed,
            body_joint_ids=body_joint_ids if len(body_joint_ids) > 0 else (0, 11, 12, 13, 14),
        )

        # codebooks
        cb_bodyloc = z["cb_bodyloc"]
        tok.cb_bodyloc = cb_bodyloc.astype(np.float32) if cb_bodyloc.size > 0 else None

        tok.cb_loc = z["cb_loc"].astype(np.float32)
        tok.cb_mov = z["cb_mov"].astype(np.float32)

        cb_hs = z["cb_hs"]
        tok.cb_hs = cb_hs.astype(np.float32) if cb_hs.size > 0 else None

        tok.cb_rel = z["cb_rel"].astype(np.float32)

        # stats
        def read_stats(mu_key, sd_key):
            mu = z[mu_key]
            sd = z[sd_key]
            if mu.size == 0 or sd.size == 0:
                return None
            return (mu.astype(np.float32), sd.astype(np.float32))

        tok.bodyloc_stats = read_stats("bodyloc_mu", "bodyloc_sd")
        tok.loc_stats = read_stats("loc_mu", "loc_sd")
        tok.mov_stats = read_stats("mov_mu", "mov_sd")
        tok.hs_stats = read_stats("hs_mu", "hs_sd")
        tok.rel_stats = read_stats("rel_mu", "rel_sd")

        return tok
# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # dummy
    T = 100
    N=70
    pose = np.random.randn(N,T,33,3).astype(np.float32)
    L = np.random.randn(N,T,21,3).astype(np.float32)
    R = np.random.randn(N,T,21,3).astype(np.float32)

    # dataset format: list of (pose,L,R)
    dataset = [pose, L, R]

    tok = SignMotionTokenizer(
        k_loc=64, k_mov=64, k_hs=128, k_rel=32,
        fps=25, use_xy_only=True, hold_speed_thresh=0.01, seed=0
    )
    for p,l,r in zip(pose,L,R):
        sample_frame=p.shape[0]
        tok.fit_codebooks([(p,l,r)], sample_frames=sample_frame, max_iter=30)
        feature=tok._extract_features_one(p,l,r)
        out = tok.encode(p,l,r)
        #print("loc:", feature["loc"].shape, "mov_L", feature["mov_L"].shape, "mov_R", feature["mov_R"].shape, "hs_L", feature["hs_L"].shape, "hs_R", feature["hs_R"].shape, "rel", feature["rel"].shape,"bodyloc",feature["bodyloc"].shape)
        print(out["loc"].shape, out["mov"].shape, out["hs"].shape, out["rel"].shape,out['bodyloc'].shape)
    #print("centers:", tok.cb_loc, tok.cb_mov, tok.cb_hs, tok.cb_rel,tok.cb_bodyloc)
    #demo save and load
    tok.save("/home/caffe/work/SLG/loader/tokenizer_demo.npz")
    tok2 = SignMotionTokenizer.load("/home/caffe/work/SLG/loader/tokenizer_demo.npz")
    #print("Loaded centers:", tok2.cb_loc.shape, tok2.cb_mov.shape, tok2.cb_hs.shape, tok2.cb_rel.shape,tok2.cb_bodyloc.shape)