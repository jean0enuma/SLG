import numpy as np

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
EPS = 1e-8

def _safe_norm(x, axis=-1, keepdims=True, eps=EPS):
    return np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims) + eps)
def compute_interaction_event_block(
    left_hand_c,   # (T,21,3) mid原点にセンタリング済み（wrist=0）
    right_hand_c,  # (T,21,3) 同上
    left_valid,    # (T,) bool そのフレームで左手が有効
    right_valid,   # (T,) bool
    *,
    wrist_id=0,
    hold_speed_thresh=0.015,  # ★要調整（あなたの座標スケールに依存）
    hold_min_frames=3,        # 短いノイズholdを抑制したい場合
    return_raw=False,
):
    """
    Returns:
      feat: (T, F) where F=5 by default [d_lr, dv_lr, hold_L, hold_R, sync]
    Notes:
      - 速度はフレーム差分。t=0は0扱い。
      - invalid handの特徴は0埋め（マスクを別途使うなら、損失側で無視が理想）
    """
    T = left_hand_c.shape[0]
    assert right_hand_c.shape[0] == T

    wL = left_hand_c[:, wrist_id, :2]   # (T,3)
    wR = right_hand_c[:, wrist_id, :2]  # (T,3)

    # ---------- distance ----------
    # 両手が有効なときだけ計算し、他は0
    both = left_valid & right_valid
    d_lr = np.zeros((T,), dtype=np.float32)
    if np.any(both):
        new_dlr=_safe_norm(wL[both] - wR[both], axis=-1).astype(np.float32).squeeze(-1)
        d_lr[both] = new_dlr

    # ---------- velocity ----------
    # v[t] = ||w[t]-w[t-1]||, t=0は0
    vL = np.zeros((T,), dtype=np.float32)
    vR = np.zeros((T,), dtype=np.float32)

    # 速度は「連続して有効な区間」でのみ計算（t-1も有効）
    valid_vel_L = left_valid[1:] & left_valid[:-1]
    valid_vel_R = right_valid[1:] & right_valid[:-1]

    if np.any(valid_vel_L):
        diff = wL[1:][valid_vel_L] - wL[:-1][valid_vel_L]
        vL[1:][valid_vel_L] = _safe_norm(diff, axis=-1).astype(np.float32).squeeze(-1)
    if np.any(valid_vel_R):
        diff = wR[1:][valid_vel_R] - wR[:-1][valid_vel_R]
        vR[1:][valid_vel_R] = _safe_norm(diff, axis=-1).astype(np.float32).squeeze(-1)

    # 両手速度差（スカラー）
    dv_lr = np.zeros((T,), dtype=np.float32)
    if np.any(both):
        dv_lr[both] = np.abs(vL[both] - vR[both]).astype(np.float32)

    # 同期度（ベクトル速度差のノルム）
    # sync[t] = || (wL[t]-wL[t-1]) - (wR[t]-wR[t-1]) ||
    sync = np.zeros((T,), dtype=np.float32)
    valid_sync = both[1:] & both[:-1]
    if np.any(valid_sync):
        vvecL = wL[1:][valid_sync] - wL[:-1][valid_sync]
        vvecR = wR[1:][valid_sync] - wR[:-1][valid_sync]
        sync[1:][valid_sync] = _safe_norm(vvecL - vvecR, axis=-1).astype(np.float32).squeeze(-1)

    # ---------- hold indicators ----------
    # hold = (速度が小さい) を基本に、短い島を除去（任意）
    hold_L = (vL < hold_speed_thresh) & left_valid
    hold_R = (vR < hold_speed_thresh) & right_valid

    # 短いhold区間を除去したい場合（モルフォロジー的な簡易処理）
    if hold_min_frames > 1:
        def suppress_short_runs(b):
            b = b.astype(np.bool_)
            out = b.copy()
            i = 0
            while i < len(b):
                if not b[i]:
                    i += 1
                    continue
                j = i
                while j < len(b) and b[j]:
                    j += 1
                if (j - i) < hold_min_frames:
                    out[i:j] = False
                i = j
            return out

        hold_L = suppress_short_runs(hold_L)
        hold_R = suppress_short_runs(hold_R)

    # float化（0/1）
    hold_L_f = hold_L.astype(np.float32)
    hold_R_f = hold_R.astype(np.float32)
    #必要ならdv_lrを追加
    feat = np.stack([d_lr, hold_L_f, hold_R_f, sync], axis=-1)  # (T,4)

    if return_raw:
        return feat, {"vL": vL, "vR": vR, "both": both}
    return feat

def _safe_normalize(x, axis=-1, eps=EPS):
    return x / _safe_norm(x, axis=axis, keepdims=True, eps=eps)

def _cos_sim(a, b, axis=-1, eps=EPS):
    a_n = a / _safe_norm(a, axis=axis, keepdims=True, eps=eps)
    b_n = b / _safe_norm(b, axis=axis, keepdims=True, eps=eps)
    return np.sum(a_n * b_n, axis=axis)

def _orthonormal_hand_frame(wrist, index_mcp, pinky_mcp, alt_axis=None, eps=EPS):
    """
    Build hand frame from wrist/index_mcp/pinky_mcp.
    Returns: x, y, z (each (...,3)), and a valid mask (...,)
      x: wrist -> index_mcp direction
      y: in-plane direction
      z: palm normal
    alt_axis: optional (...,3) alternative vector to stabilize z when cross is near zero
    """
    v1 = index_mcp - wrist  # (...,3)
    v2 = pinky_mcp - wrist  # (...,3)

    # x axis
    x = _safe_normalize(v1, eps=eps)

    # provisional y'
    y_prime = _safe_normalize(v2, eps=eps)

    # z = x cross y'
    z_raw = np.cross(x, y_prime)
    z_norm = _safe_norm(z_raw, axis=-1, keepdims=False, eps=eps)  # (...,)

    # validity: both v1 and v2 non-degenerate and cross not too small
    v1_norm = _safe_norm(v1, axis=-1, keepdims=False, eps=eps)
    v2_norm = _safe_norm(v2, axis=-1, keepdims=False, eps=eps)
    valid = (v1_norm > 1e-4) & (v2_norm > 1e-4) & (z_norm > 1e-4)

    # If invalid and alt_axis is provided, try z = x cross alt_axis
    if alt_axis is not None:
        alt = alt_axis
        z_alt = np.cross(x, _safe_normalize(alt, eps=eps))
        z_alt_norm = _safe_norm(z_alt, axis=-1, keepdims=False, eps=eps)
        use_alt = (~valid) & (z_alt_norm > 1e-4)
        if np.any(use_alt):
            z_raw = np.where(use_alt[..., None], z_alt, z_raw)
            z_norm = np.where(use_alt, z_alt_norm, z_norm)
            valid = valid | use_alt

    z = _safe_normalize(z_raw, eps=eps)

    # y = z cross x (guarantee orthogonality)
    y = _safe_normalize(np.cross(z, x), eps=eps)

    return x, y, z, valid

# ------------------------------------------------------------
# Main: build coarse representation
# ------------------------------------------------------------
def build_coarse_from_mediapipe(
    pose_xyz,   # (T, 33, 3) MediaPipe Pose 3D (z assumed)
    left_hand,  # (T, 21, 3) MediaPipe Hands 3D, left
    right_hand, # (T, 21, 3) MediaPipe Hands 3D, right
    *,
    body_ids=(0, 1,2, 3, 4, 5), # nose, L/R shoulder, L/R elbow
    origin_mode="shoulder_mid",   # fixed for this project
    hand_wrist_id=0,
    hand_index_mcp_id=5,
    hand_middle_mcp_id=9,
    hand_pinky_mcp_id=17,
    use_middle_for_fallback=True,
    return_masks=True,
    add_hand_points=(4,5,8,9,12,13,16,17,20),
    is_heuristic_feature=True,
):
    """
    Returns:
      coarse: (T, 36) = [body6pts(18), Lhand(9), Rhand(9)]
              body6pts order: [p0,p11,p12,p13,p14,pmid] each 3D, all centered by mid
              hand order per hand: [wrist(3), x_dir(3), z_normal(3)] in the same centered coords
      masks (optional): dict with boolean arrays:
          - left_valid: (T,)
          - right_valid: (T,)
    """
    pose_xyz = np.asarray(pose_xyz, dtype=np.float32)
    left_hand = np.asarray(left_hand, dtype=np.float32)
    right_hand = np.asarray(right_hand, dtype=np.float32)

    assert pose_xyz.ndim == 3 and pose_xyz.shape[1] >= 13 and pose_xyz.shape[2] == 3
    assert left_hand.shape[-2:] == (21, 3)
    assert right_hand.shape[-2:] == (21, 3)
    T = pose_xyz.shape[0]
    assert left_hand.shape[0] == T and right_hand.shape[0] == T

    # ---- origin: shoulder mid (11,12 midpoint) ----
    #p11 = pose_xyz[:, 11, :]  # (T,3)
    #p12 = pose_xyz[:, 12, :]  # (T,3)
    mid = pose_xyz[:,1]   # (T,3)

    # center pose & hands
    pose_c = pose_xyz - mid[:, None, :]       # (T,33,3)
    left_c = left_hand - mid[:, None, :]      # (T,21,3)
    right_c = right_hand - mid[:, None, :]    # (T,21,3)

    # ---- body 6 points: [0,11,12,13,14,mid] ----
    # note: after centering, mid becomes (0,0,0)
    body_list = []
    #body_list.append(np.zeros((T, 3), dtype=np.float32))  # centered mid
    for idx in body_ids:
        body_list.append(pose_c[:, idx, :2])   # (T,3)
    #body_list.append(np.zeros((T, 3), dtype=np.float32))  # centered mid
    body6 = np.concatenate(body_list, axis=-1)  # (T, 18)

    # ---- hands: wrist + x_dir + z_normal ----
    def hand_feats(hand_c,valid_mask=None):
        wrist = hand_c[:, hand_wrist_id, :]          # (T,3)
        index_mcp = hand_c[:, hand_index_mcp_id, :]  # (T,3)
        pinky_mcp = hand_c[:, hand_pinky_mcp_id, :]  # (T,3)

        alt_axis = None
        if use_middle_for_fallback:
            middle_mcp = hand_c[:, hand_middle_mcp_id, :]
            alt_axis = (middle_mcp - wrist)  # (T,3)

        x, y, z, valid = _orthonormal_hand_frame(
            wrist=wrist,
            index_mcp=index_mcp,
            pinky_mcp=pinky_mcp,
            alt_axis=alt_axis,
        )
        #hand scale
        s=_safe_norm(index_mcp[:,:2] - wrist[:,:2], axis=-1, keepdims=True)  # (T,1)
        # ----- 速度計算 -----
        speed = np.zeros((T, 1), dtype=np.float32)
        if valid_mask is None:
            valid_mask = np.ones((T,), dtype=bool)
        valid_vel = valid_mask[1:] & valid_mask[:-1]
        if np.any(valid_vel):
            diff = wrist[1:,:2][valid_vel] - wrist[:-1,:2][valid_vel]
            speed[1:][valid_vel, 0] = _safe_norm(diff, axis=-1).squeeze(-1)

        # Features per frame: wrist(3), x(3), z(3),s(1),speed(1) => (T,11)
        feats = np.concatenate([wrist, x, z,s,speed], axis=-1).astype(np.float32)

        # If invalid, you can zero-out (or keep raw). Here: zero-out invalid frames.
        feats = np.where(valid[:, None], feats, 0.0)

        return feats, valid
    if is_heuristic_feature:
        left_feat, left_valid = hand_feats(left_c)#(T,11), (T,)
        right_feat, right_valid = hand_feats(right_c)#(T,11), (T,)
    else:
        #全ての値が0の特徴量のmask(T,)を作る
        left_valid = np.ones((T,), dtype=bool)
        right_valid = np.ones((T,), dtype=bool)
        #全ての値が0のフレームはFalseにする
        left_valid[np.all(left_c == 0, axis=(1,2))] = False
        right_valid[np.all(right_c == 0, axis=(1,2))] = False
        left_feat = left_c[:,:1,:].reshape(T, -1)  # (T,63)
        right_feat = right_c[:,:1,:].reshape(T, -1)  # (T,63)



    if add_hand_points is not None:
        left_points = left_c[:, add_hand_points, :].reshape(T, -1)  # (T, 3*len(add_hand_points))
        left_feat = np.concatenate([left_feat, left_points], axis=-1)
    if add_hand_points is not None:
        right_points = right_c[:, add_hand_points, :].reshape(T, -1)  # (T, 3*len(add_hand_points))
        right_feat = np.concatenate([right_feat, right_points], axis=-1)

    coarse = np.concatenate([body6, left_feat, right_feat], axis=-1)  # (T,36)
    #assert coarse.shape == (T, 36)

    if return_masks:
        masks = {
            "left_valid": left_valid.astype(bool),
            "right_valid": right_valid.astype(bool),
        }
        return coarse, masks
    return coarse

# ------------------------------------------------------------
# 既存の coarse(例: 36次元) に interaction/event を付け足す例
# ------------------------------------------------------------
def concat_coarse_with_interaction(
    coarse36,          # (T,36) 例: [body18, L(wrist,x,z)9, R(...)9]
    left_hand_c,       # (T,21,3) mid原点
    right_hand_c,      # (T,21,3)
    masks,             # dict: {"left_valid":(T,), "right_valid":(T,)}
    *,
    hold_speed_thresh=0.015,
):
    inter = compute_interaction_event_block(
        left_hand_c, right_hand_c,
        masks["left_valid"], masks["right_valid"],
        hold_speed_thresh=hold_speed_thresh
    )  # (T,5)
    coarse_out = np.concatenate([coarse36, inter], axis=-1)  # (T,41)
    return coarse_out
# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    # Dummy example shapes:
    T = 100
    pose = np.random.randn(T, 48, 3).astype(np.float32)
    lh = np.random.randn(T, 21, 3).astype(np.float32)
    rh = np.random.randn(T, 21, 3).astype(np.float32)

    coarse, masks = build_coarse_from_mediapipe(pose, lh, rh,add_hand_points=None)
    coarse_out=concat_coarse_with_interaction(coarse, lh, rh, masks)
    print(coarse.shape)  # (T,36)
    print(masks["left_valid"].mean(), masks["right_valid"].mean())