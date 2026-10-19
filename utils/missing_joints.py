"""
欠損関節（全座標が厳密に 0 の関節）に対応した学習用の前処理と loss。

方針
  1. 欠損は「正規化の前」に検出する（正規化後は 0 でなくなり判別できない）
  2. 正規化: 関節10を原点、肩幅(11-12)の系列内中央値×3 で割る
     → 元実装（首原点・肩幅≈1/3・値域≈±0.5）とほぼ同じスケール
  3. 入力 trg_input: 欠損を時間方向の最近傍（直前、無ければ直後）の有効値で補完
     → 推論時には存在しない「0 = 欠損」という入力分布を学習させない
  4. 教師 trg: 補完しない。欠損要素は loss マスクで除外（有効要素のみで平均）
  5. trg_mask: 特徴次元の pad ハックではなく系列長から作る
"""
import torch
from torch import Tensor



CENTER, L_SH, R_SH = 10, 11, 12       # normalize_batch と同じ関節 index（全関節 J 基準）
def mask_to_lengths(attention_mask: Tensor, check: bool = False) -> Tensor:
    """HF形式の attention_mask (B, L)（1=有効, 0=pad）→ 有効長 (B,)。

    右詰め・左詰めのどちらでも lengths = Σ_l m[b, l] で求まる。
    check=True のとき、有効部分が連続した 1 のブロックか（穴あきでないか）を検証する。
    """
    if attention_mask.dim() != 2:
        raise ValueError(f"expected (B, L), got {tuple(attention_mask.shape)}")
    m = attention_mask.long()
    lengths = m.sum(dim=-1)
    if check:
        # 0/1 の遷移回数: 連続ブロックなら右詰めで ≤1, 左詰めで ≤1
        transitions = (m[:, 1:] != m[:, :-1]).sum(dim=-1)
        if (transitions > 1).any() or ((m != 0) & (m != 1)).any():
            raise ValueError("attention_mask is not a contiguous 0/1 mask")
    return lengths
def lengths_to_mask(
        lengths: Tensor,
        max_len: int | None = None,
        padding_side: str = "right",
        dtype: torch.dtype = torch.long,
) -> Tensor:
    """有効長 (B,) → HF形式の attention_mask (B, L)。mask_to_lengths の逆関数。

    right: m[b, l] = 1[l <  len_b]
    left : m[b, l] = 1[l >= L - len_b]
    """
    lengths = torch.as_tensor(lengths)
    L = int(lengths.max().item()) if max_len is None else max_len
    if (lengths > L).any() or (lengths < 0).any():
        raise ValueError(f"lengths must be in [0, {L}]")
    pos = torch.arange(L, device=lengths.device).unsqueeze(0)  # (1, L)
    lens = lengths.unsqueeze(1)  # (B, 1)
    if padding_side == "right":
        mask = pos < lens
    elif padding_side == "left":
        mask = pos >= (L - lens)
    else:
        raise ValueError("padding_side must be 'right' or 'left'")
    return mask.to(dtype)
def make_trg_mask(valid_len: Tensor, T: int) -> Tensor:
    valid = lengths_to_mask(valid_len.clamp(max=T), max_len=T, dtype=torch.bool)
    return valid[:, None, :, None].expand(-1, 1, -1, T)


def append_counter(x: Tensor, lengths: Tensor) -> Tensor:
    """x (B, T, D) → (B, T, D+1)。c_t = t / T_b、パディングは 0（ループなし版）。"""
    B, T, _ = x.shape
    t = torch.arange(T, device=x.device, dtype=x.dtype)[None, :]
    L = lengths.to(x.device, x.dtype)[:, None]
    c = torch.where(t < L, t / L, torch.zeros_like(t))
    return torch.cat([x, c[..., None]], dim=-1)


def joint_valid_mask(padded: Tensor, lengths: Tensor) -> Tensor:
    """padded (B, T, J, 3) → (B, T, J) bool。全座標が 0 の関節と、系列長外のフレームを無効とする。"""
    T = padded.shape[1]
    frame = lengths_to_mask(lengths, max_len=T, dtype=torch.bool)       # (B, T)
    return (padded.abs().amax(dim=-1) > 0) & frame[:, :, None]


def normalize_with_mask(padded: Tensor, valid: Tensor, div: float = 3.0, eps: float = 1e-6):
    """(B, T, J, 3) → x (B, T, 3, J)（all_inputs と同じレイアウト）。欠損要素は 0 のまま返す。

    中心: フレームごとの関節10（欠損フレームは全関節を無効化）
    スケール: 両肩が有効なフレームでの肩幅の系列内中央値（フレームごとの揺れと欠損に頑健）
    """
    x = padded.permute(0, 1, 3, 2).float()                               # (B, T, 3, J)
    center = x[..., CENTER]                                              # (B, T, 3)
    sw = (x[..., L_SH] - x[..., R_SH]).norm(dim=2)                       # (B, T)
    sw = sw.masked_fill(~(valid[..., L_SH] & valid[..., R_SH]), float("nan"))
    s = torch.nan_to_num(sw.nanmedian(dim=1).values, nan=1.0).clamp_min(eps)   # (B,)
    scale = div * s                                                      # 肩幅 → 1/div

    valid = valid & valid[..., CENTER:CENTER + 1]                        # 中心が無いフレームは使わない
    x = (x - center[..., None]) / scale[:, None, None, None]
    x = x * valid[:, :, None, :]                                         # 欠損要素は 0
    return x, valid, dict(center=center, scale=scale,center_valid=valid[..., CENTER:CENTER + 1])


def fill_missing_time(x: Tensor, valid: Tensor) -> Tensor:
    """x (B, T, C, J), valid (B, T, J)。各関節の欠損を直前の有効フレーム値で、
    先頭側の欠損は直後の有効値で埋める。系列中に一度も有効でない関節は 0。"""
    B, T, C, J = x.shape
    t = torch.arange(T, device=x.device)[None, :, None].expand(B, T, J)
    prev = torch.where(valid, t, torch.full_like(t, -1)).cummax(dim=1).values
    nxt = torch.where(valid, t, torch.full_like(t, T)).flip(1).cummin(dim=1).values.flip(1)
    src = torch.where(prev >= 0, prev, nxt)                              # (B, T, J)
    has = src < T
    idx = src.clamp(max=T - 1)[:, :, None, :].expand(B, T, C, J)
    return x.gather(1, idx) * has[:, :, None, :]


def prepare_trg(padded_cod_data: Tensor, lengths: Tensor, used_jts, device):
    """学習ループ用。戻り値:
        trg_input (B, T′, 3J′+1)  補完済み関節 + counter
        trg       (B, T′, 3J′+1)  未補完の教師（欠損要素は 0）
        loss_mask (B, T′, 3J′+1)  有効要素 = True（counter はフレーム有効性）
        trg_mask  (B, 1, T′, T′)
        norm_info                 可視化時の逆正規化用
    """
    padded = padded_cod_data.to(device)
    lengths = lengths.to(device)
    B, T = padded.shape[:2]

    valid = joint_valid_mask(padded, lengths)                            # 正規化前に検出
    x, valid, norm_info = normalize_with_mask(padded, valid)
    x, valid = x[..., used_jts], valid[..., used_jts]                    # (B, T, 3, J′), (B, T, J′)

    frame = lengths_to_mask(lengths, max_len=T, dtype=torch.bool)
    x_in = fill_missing_time(x, valid) * frame[:, :, None, None]         # 入力側のみ補完

    joints_in = x_in.reshape(B, T, -1)                                   # [x.., y.., z..]
    joints_tg = x.reshape(B, T, -1)
    elem = valid[:, :, None, :].expand(-1, -1, 3, -1).reshape(B, T, -1)  # 同じ並びのマスク

    inp = append_counter(joints_in, lengths)
    tgt = append_counter(joints_tg, lengths)
    mask = torch.cat([elem, frame[..., None]], dim=-1)

    trg_input, trg, loss_mask = inp[:, :-1], tgt[:, 1:], mask[:, 1:]
    trg_mask = make_trg_mask(lengths, T - 1)
    return trg_input, trg, loss_mask, trg_mask, norm_info


def masked_regression_loss(pred: Tensor, target: Tensor, mask: Tensor,
                           kind: str = "mse", weight: Tensor | None = None) -> Tensor:
    """有効要素のみで平均。weight (D,) で次元ごとの重み（手の強調など）も可。"""
    err = pred.float() - target.float()
    err = err.pow(2) if kind == "mse" else err.abs()
    m = mask.float() if weight is None else mask.float() * weight
    return (err * m).sum() / m.sum().clamp_min(1.0)

def _unflatten(y: Tensor) -> Tensor:
    """(B, T, 3J′[+1]) → (B, T, 3, J′)。末尾の counter 列があれば落とす。(B, T, 3, J′) はそのまま。"""
    if y.dim() == 4:
        return y
    if y.shape[-1] % 3 == 1:
        y = y[..., :-1]
    return y.reshape(y.shape[0], y.shape[1], 3, -1)
def denormalize(y: Tensor, norm_info: dict, offset: int = 0, center_mode: str = "frame",
                valid: Tensor | None = None) -> Tensor:
    """normalize_with_mask の逆変換:  x = y · scale_b + center_{b,t}

    y          : (B, T, 3J′[+1]) 平坦化（counter 付き可）または (B, T, 3, J′)
    offset     : y の時刻 0 が元系列の何フレーム目か
                 greedy 出力 hyp（先頭が y_0）→ 0,  skel_out / trg（y_1 始まり）→ 1
    center_mode: "frame"  元系列のフレームごとの中心（GT の大域移動をそのまま戻す）
                 "median" 系列内の中心の中央値で固定（GT の大域移動を持ち込まない）
    valid      : (B, T, J′) を渡すと無効要素を 0 に戻す（元データと同じ欠損表現）
    戻り値      : (B, T, 3, J′)  all_inputs[..., used_jts] と同じレイアウト・元の座標系
    """
    x = _unflatten(y).float()
    B, T = x.shape[:2]
    center, cv, scale = norm_info["center"], norm_info["center_valid"], norm_info["scale"]
    T0 = center.shape[1]

    if center_mode == "frame":
        # 中心が欠損/パディングのフレームは時間方向の最近傍の有効中心で埋める
        c = fill_missing_time(center[..., None], cv)[..., 0]  # (B, T0, 3)
        idx = (torch.arange(T, device=x.device) + offset).clamp(max=T0 - 1)  # 範囲外は末尾で延長
        c = c[:, idx]  # (B, T, 3)
    elif center_mode == "median":
        med = center.masked_fill(~cv, float("nan")).nanmedian(dim=1).values
        c = torch.nan_to_num(med, nan=0.0)[:, None, :].expand(B, T, 3)
    else:
        raise ValueError("center_mode must be 'frame' or 'median'")

    out = x * scale.to(x)[:, None, None, None] + c.to(x)[..., None]
    if valid is not None:
        out = out * valid[:, :, None, :].to(out)
    return out

# ----------------------------------------------------------------------------- 学習ループでの使い方
"""
used_jts = bones_used_joints(bones=bones, include_aux=True)

for batch_idx, batch in enumerate(dl_train):
    padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens = batch

    # --- src（従来通り）
    text_tokens = text_tokens.to(device)
    src      = text_tokens["input_ids"][:, 1:]
    src_mask = text_tokens["attention_mask"][:, 1:]
    src_length = mask_to_lengths(src_mask)
    src_mask = src_mask.unsqueeze(1).bool()

    # --- trg（欠損対応）
    trg_input, trg, loss_mask, trg_mask, norm_info = prepare_trg(
        padded_cod_data, input_length_tensor, used_jts, device)

    skel_out, _ = model(src=src, trg_input=trg_input, src_mask=src_mask,
                        src_lengths=src_length, trg_mask=trg_mask)       # まずは fp32 で
    raw_loss = masked_regression_loss(skel_out, trg, loss_mask)

    loss = raw_loss / current_accumulation_steps
    loss.backward()
    ...（以降の optimizer 更新は従来通り）
"""


# ----------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, J = 2, 6, 14
    padded = torch.randn(B, T, J, 3)
    lengths = torch.tensor([6, 4])
    padded[1, 4:] = 0                                  # パディング
    padded[0, 1:3, 13] = 0                             # 関節13がフレーム1-2で欠損
    padded[0, 0, 12] = 0                               # 関節12(肩)が先頭フレームで欠損
    padded[1, :, 7] = 0                                # 関節7が系列全体で欠損
    padded[0, 5, CENTER] = 0                           # 中心が欠損したフレーム
    used = [3, 7, 12, 13]

    ti, tg, lm, tm, info = prepare_trg(padded, lengths, used, "cpu")
    Jp = len(used)
    print("shapes", ti.shape, tg.shape, lm.shape, tm.shape)
    assert ti.shape == tg.shape == lm.shape == (B, T - 1, 3 * Jp + 1)

    # 関節13(列 index 3)の x 成分: 入力は補完済みで非ゼロ、教師側は欠損でマスク False
    col_x13 = 3
    print("sample0 j13 x input :", ti[0, :, col_x13])
    print("sample0 j13 mask    :", lm[0, :, col_x13])
    assert (ti[0, :4, col_x13] != 0).all()             # 入力 t=0..3（元フレーム1,2 は補完）
    assert torch.equal(ti[0, 1, col_x13], ti[0, 0, col_x13])   # 直前値で補完
    assert not lm[0, 0, col_x13] and not lm[0, 1, col_x13]     # 教師 t=0,1 = 元フレーム1,2 は無効
    # 系列全体で欠損した関節7(列 index 1) は入力も 0、マスクも全 False
    assert (ti[1, :, 1] == 0).all() and not lm[1, :, 1].any()
    # 中心欠損フレーム(元フレーム5 → 教師 t=4) は関節すべて無効、counter は有効
    assert not lm[0, 4, :-1].any() and lm[0, 4, -1]
    # パディング(sample1 の元フレーム4,5 → 教師 t=3,4) は counter 含めすべて無効
    assert not lm[1, 3:].any()
    # trg_mask: sample1 は有効長4 → 行 0..3 が True、行4 が False
    assert tm[1, 0, :4].all() and not tm[1, 0, 4].any()
    # loss: 有効要素のみの平均
    pred = torch.zeros_like(tg)
    l = masked_regression_loss(pred, tg, lm)
    ref = (tg[lm] ** 2).mean()
    assert torch.isclose(l, ref), (l, ref)
    print("scale", info["scale"], "loss", l.item())
    print("all tests passed")