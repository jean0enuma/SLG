"""
skeleton_video.py (OpenCV版)
=============================
骨格座標 (T, 3, J) を OpenCV で描画し動画 (.mp4 / .avi) として保存する.

matplotlib版との違い:
    - cv2.line / cv2.circle による直接描画で高速 (数百フレームでも数秒)
    - 依存は opencv-python のみ (ffmpeg外部バイナリ不要)
    - x, y のみ使用 (2D描画, MediaPipe画像座標を想定. yは下向きが正)

依存: pip install opencv-python  (サーバなら opencv-python-headless)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import cv2


def _to_numpy(x) -> np.ndarray:
    """torch.Tensor / np.ndarray -> np.ndarray (T, 3, J)."""
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != 3:
        raise ValueError(f"expected (T, 3, J), got {x.shape}")
    return x


def _fit_transform(stack: np.ndarray, size: tuple[int, int], margin: float):
    """全フレームの外接箱を画面に収める スケール + オフセット を計算.
    アスペクト比は保持 (縦横で同一スケール)."""
    W, H = size
    xs, ys = stack[:, 0], stack[:, 1]
    fin = np.isfinite(xs) & np.isfinite(ys)
    x_lo, x_hi = xs[fin].min(), xs[fin].max()
    y_lo, y_hi = ys[fin].min(), ys[fin].max()
    span_x = max(x_hi - x_lo, 1e-6)
    span_y = max(y_hi - y_lo, 1e-6)
    scale = min(W * (1 - 2 * margin) / span_x,
                H * (1 - 2 * margin) / span_y)
    # 中央寄せオフセット
    off_x = (W - scale * span_x) / 2 - scale * x_lo
    off_y = (H - scale * span_y) / 2 - scale * y_lo
    return scale, off_x, off_y


def save_skeleton_video(
    x,
    save_path: str | Path,
    bones=None,
    fps: int = 15,
    size: tuple[int, int] = (640, 640),
    x_ref=None,
    joint_mask=None,
    title: str | None = None,
    margin: float = 0.08,
    flip_y: bool = False,
    color=(255, 128, 0),        # BGR: 青系
    ref_color=(160, 160, 160),  # BGR: グレー
):
    """
    骨格座標をプロットした動画を OpenCV で保存する.

    Args:
        x:          (T, 3, J)  骨格座標 (torch.Tensor / np.ndarray)
        save_path:  保存先パス (.mp4 / .avi)
        bones:      [(parent, child, ...), ...]  エッジ描画用 (Noneなら点のみ)
        fps:        フレームレート
        size:       (幅, 高さ) ピクセル
        x_ref:      (T, 3, J)  比較用の参照骨格 (グレーで重ね描き, 例: GT)
        joint_mask: (T, J) bool  Falseの関節とそれを含むボーンは非表示
        title:      左上に表示するテキスト (フレーム番号は自動付与)
        margin:     画面端の余白率
        flip_y:     Trueならy軸を反転して描画.
                    MediaPipe画像座標 (yが下向き) はそのままで正立するため
                    デフォルトFalse. 数学座標系 (yが上向き) のデータならTrue.
        color/ref_color: 描画色 (BGR)
    Returns:
        Path: 保存したファイルのパス
    """
    x = _to_numpy(x)
    T, _, J = x.shape
    ref = _to_numpy(x_ref) if x_ref is not None else None
    if joint_mask is not None and hasattr(joint_mask, "detach"):
        joint_mask = joint_mask.detach().cpu().numpy()

    edges = [(b[0], b[1]) for b in bones] if bones is not None else []
    W, H = size

    # ---- 座標 -> ピクセル変換 (全フレーム共通で固定: 画面が揺れない) ----
    stack = x if ref is None else np.concatenate([x, ref], axis=0)
    flat = stack.transpose(0, 2, 1).reshape(-1, 3)  # (T*J, 3)
    scale, off_x, off_y = _fit_transform(flat, size, margin)

    def to_px(frame_xyz):
        """(3, J) -> (J, 2) int ピクセル座標."""
        px = frame_xyz[0]*size[1]
        py = frame_xyz[1]*size[0]
        if flip_y:
            py = H - py
        return np.stack([px, py], axis=-1)

    # ---- VideoWriter -----------------------------------------------
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = save_path.suffix.lower()
    fourcc = {".mp4": "mp4v", ".avi": "XVID"}.get(suffix)
    if fourcc is None:
        raise ValueError(f"unsupported extension: {suffix} (.mp4 / .avi)")
    writer = cv2.VideoWriter(str(save_path),
                             cv2.VideoWriter_fourcc(*fourcc), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter: {save_path}")
    edges = [(b[0], b[1]) for b in bones] if bones is not None else []
    # ボーンに使われる関節の集合 (bonesなしなら全関節を点表示)
    used_joints = (sorted({j for e in edges for j in e})
                   if bones is not None else list(range(J)))
    def draw_one(canvas, pts, vis, c, thickness, radius):
        pts_i = np.round(pts).astype(int)
        for a, b in edges:
            if vis[a] and vis[b]:
                if pts_i[a][0]<=0 or pts_i[a][0]>=W or pts_i[a][1]<=0 or pts_i[a][1]>=H:
                    continue
                if pts_i[b][0]<=0 or pts_i[b][0]>=W or pts_i[b][1]<=0 or pts_i[b][1]>=H:
                    continue
                cv2.line(canvas, tuple(pts_i[a]), tuple(pts_i[b]),
                         c, thickness, cv2.LINE_AA)
        for j in used_joints:  # 変更: range(J) → used_joints
            if pts_i[j][0] <= 0 or pts_i[j][0] >= W \
                    or pts_i[j][1] <= 0 or pts_i[j][1] >= H:
                continue
            if vis[j]:
                cv2.circle(canvas, tuple(pts_i[j]), radius, c, -1,
                           cv2.LINE_AA)

    try:
        for t in range(T):
            canvas = np.full((H, W, 3), 255, dtype=np.uint8)  # 白背景

            mask_t = (joint_mask[t].astype(bool)
                      if joint_mask is not None else np.ones(J, bool))

            if ref is not None:  # 参照(GT)を先に = 下層に描く
                vis_r = np.isfinite(ref[t][0]) & np.isfinite(ref[t][1]) & mask_t
                draw_one(canvas, to_px(ref[t]), vis_r, ref_color, 2, 3)

            vis = np.isfinite(x[t][0]) & np.isfinite(x[t][1]) & mask_t
            draw_one(canvas, to_px(x[t]), vis, color, 2, 4)

            head = f"{title}  " if title else ""
            cv2.putText(canvas, f"{head}frame {t + 1}/{T}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1,
                        cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
    return save_path


# ----------------------------------------------------------------------
# 動作確認: 合成した手骨格の開閉モーションを保存
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    HAND_BONES = [
        (0, 1, 5), (1, 2, 5), (2, 3, 5), (3, 4, 5),
        (0, 5, 9), (5, 6, 9), (6, 7, 9), (7, 8, 9),
        (0, 9, 5), (9, 10, 5), (10, 11, 5), (11, 12, 5),
        (0, 13, 9), (13, 14, 9), (14, 15, 9), (15, 16, 9),
        (0, 17, 13), (17, 18, 13), (18, 19, 13), (19, 20, 13),
    ]

    # 簡易的な手モデル: MCP列を扇状に配置し, 指を周期的に屈曲させる
    T = 60
    base_dirs = np.stack([np.array([np.cos(a), -np.sin(a)]) for a in
                          np.deg2rad([140, 100, 85, 70, 55])])
    x = np.zeros((T, 3, 21))
    for t in range(T):
        bend = 0.5 * (1 - np.cos(2 * np.pi * t / T))
        for f in range(5):
            d = base_dirs[f]
            mcp = 1 + 4 * f
            chain = [0, mcp, mcp + 1, mcp + 2, mcp + 3]
            p = np.array([0.5, 0.8])
            seg_len = [0.25, 0.12, 0.08, 0.06]
            ang = 0.0
            for k, j in enumerate(chain[1:]):
                if k >= 1:
                    ang += bend * np.deg2rad(45)
                rot = np.array([[np.cos(ang), -np.sin(ang)],
                                [np.sin(ang), np.cos(ang)]])
                p = p + seg_len[k] * (rot @ d)
                x[t, 0, j], x[t, 1, j] = p
        x[t, :2, 0] = [0.5, 0.8]
    x = torch.from_numpy(x).float()

    out1 = save_skeleton_video(x, "/mnt/user-data/outputs/hand_demo_cv.mp4",
                               bones=HAND_BONES, fps=30, title="hand")

    # 比較描画 + 欠損マスクのデモ
    x_noisy = x + 0.01 * torch.randn_like(x)
    mask = torch.rand(T, 21) > 0.05
    out2 = save_skeleton_video(x_noisy,
                               "/mnt/user-data/outputs/hand_cmp_cv.mp4",
                               bones=HAND_BONES, fps=30, x_ref=x,
                               joint_mask=mask, title="recon vs GT")
    print(out1, out2, sep="\n")