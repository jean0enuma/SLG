# -*- coding: utf-8 -*-
"""
WER が下がらない原因の切り分け用診断スクリプト．

使い方: 学習済み(途中でも可)チェックポイントと dev ローダーを用意して
       diagnose(model, dev_loader, optimizer) を呼ぶ．
       optimizer は現在の lr を見るためだけなので None でも良い．

出力の読み方:
  [1] blank率 > 0.95            → blank崩壊．λ_R を上げる / lr を下げて再学習
  [2] del >> ins (5倍以上)      → blank崩壊 or ラベル対応の破綻
  [3] ins >> del                → 復号が暴走．正規化/特徴のスケール異常を疑う
  [4] train L_R 低 & dev WER 高 → 過学習 (話者依存特徴 or データ量不足)
  [5] lr < 1e-5                 → スケジューラ早期崩壊．patience を延ばして再学習
  [6] 特徴に NaN / std 異常     → 前処理バグ
"""
from itertools import groupby
import torch

def normalize_batch(padded_cod_data, used_3d):
    all_inputs   = padded_cod_data.permute(0, 1, 3, 2).contiguous()
    body_inputs  = padded_cod_data[:, :, :-42].permute(0, 1, 3, 2).clone()
    left_inputs  = padded_cod_data[:, :, -42:-21].permute(0, 1, 3, 2).clone()
    left_inputs[...,0]=body_inputs[...,15].clone()
    right_inputs = padded_cod_data[:, :, -21:].permute(0, 1, 3, 2).clone()
    right_inputs[...,0]=body_inputs[...,16].clone()
    norm_info = None
    if used_3d:
        center = all_inputs[:, :, :, 10].clone()
        s = torch.sqrt(
            (all_inputs[:,:,0,11]-all_inputs[:,:,0,12])**2 +
            (all_inputs[:,:,1,11]-all_inputs[:,:,1,12])**2 +
            (all_inputs[:,:,2,11]-all_inputs[:,:,2,12])**2)
        B, T = s.shape
        s = s.reshape(B, T, 1, 1)
        body_inputs = (body_inputs - center.unsqueeze(3)) / (s + 1e-8)
        all_inputs  = (all_inputs  - center.unsqueeze(3)) / (s + 1e-8)
        lc = left_inputs[:, :, :, 0].clone()
        rc = right_inputs[:, :, :, 0].clone()
        left_inputs  = (left_inputs  - lc.unsqueeze(3)) / (s/2 + 1e-8)
        right_inputs = (right_inputs - rc.unsqueeze(3)) / (s/2 + 1e-8)
        norm_info = dict(center=center, shoulder=s, left_center=lc, right_center=rc)
    return all_inputs, body_inputs, left_inputs, right_inputs, norm_info
def edit_ops(ref, hyp):
    """WER 用の編集距離と del/ins/sub の内訳 (DP でトレースバック)"""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,                              # deletion
                d[i][j - 1] + 1,                              # insertion
                d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), # sub / match
            )
    # トレースバック
    i, j, dels, ins, subs = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            dels += 1; i -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1; j -= 1
        else:
            if i > 0 and j > 0 and ref[i - 1] != hyp[j - 1]:
                subs += 1
            i -= 1; j -= 1
    return dels, ins, subs

def bones_used_joints(bones, include_aux=True):
    """bonesに登場する関節インデックスをソート済みリストで返す.

    include_aux=True : parent/child/aux すべて (座標の切り出し用)
    include_aux=False: parent/child のみ (キネマティクス構造のみ)
    """
    used = set()
    for b in bones:
        used.update(b[:3] if include_aux else b[:2])
    return sorted(used)
@torch.no_grad()
def diagnose(model, dev_loader, bones,optimizer=None, criterion=None,
             device="cuda", n_show=5, blank_idx=0):
    model.eval()
    total_dels = total_ins = total_subs = total_ref = 0
    blank_frames = valid_frames = 0
    sum_lr = sum_lt = n_batches = 0.0
    shown = 0

    feat_stats = {"nan": 0, "min": float("inf"), "max": float("-inf")}

    for batch in dev_loader:
        padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens, gloss_tokens = batch
        padded_cod_data = padded_cod_data.float().to(device)
        txt = text_tokens['input_ids']
        B, _ = txt.shape
        txt_input = txt[:, :-1].to(device)
        txt_target = txt[:, 1:].to(device)
        gls_attn_mask = gloss_tokens['attention_mask']
        gls_lengths = gls_attn_mask.sum(dim=1).long().to(device)
        gls_tokens = gloss_tokens['input_ids'].to(device)
        input_length_tensor = input_length_tensor.to(device)
        all_inputs, body_inputs, left_inputs, right_inputs, norm_info = normalize_batch(padded_cod_data,
                                                                                        True)
        new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
        new_inputs = new_inputs.reshape(new_inputs.shape[0], new_inputs.shape[1], -1)
        sgn=new_inputs
        sgn_len=input_length_tensor
        gls=gls_tokens
        gls_len=gls_lengths
        txt_in=txt_input
        txt_tgt=txt_target

        # --- [6] 特徴量の健全性 ---
        feat_stats["nan"] += torch.isnan(sgn).sum().item()
        feat_stats["min"] = min(feat_stats["min"], sgn.min().item())
        feat_stats["max"] = max(feat_stats["max"], sgn.max().item())

        # --- [1] フレーム毎の blank 率 (greedy argmax ベース) ---
        z, _ = model.encode(sgn, sgn_len)
        pred = model.gloss_output_layer(z).argmax(-1)  # (B, T)
        for b in range(pred.size(0)):
            p = pred[b, : sgn_len[b]]
            blank_frames += (p == blank_idx).sum().item()
            valid_frames += int(sgn_len[b])

        # --- [2][3] WER と del/ins/sub 内訳 ---
        hyps = model.recognize_greedy(sgn, sgn_len)
        for b in range(len(hyps)):
            ref = gls[b, : gls_len[b]].tolist()
            d_, i_, s_ = edit_ops(ref, hyps[b])
            total_dels += d_; total_ins += i_; total_subs += s_
            total_ref += len(ref)
            if shown < n_show:
                print(f"  ref: {ref}\n  hyp: {hyps[b]}\n")
                shown += 1

        # --- [4] 損失のスケール比 ---
        if criterion is not None:
            glp, wl = model(sgn, sgn_len, txt_in)
            _, lr_, lt_ = criterion(glp, sgn_len, gls, gls_len, wl, txt_tgt, sgn.size(0))
            sum_lr += lr_.item(); sum_lt += lt_.item(); n_batches += 1

    wer = 100.0 * (total_dels + total_ins + total_subs) / max(total_ref, 1)
    print("=" * 60)
    print(f"[1] blank率           : {blank_frames / max(valid_frames,1):.3f}"
          f"  (>0.95 なら blank 崩壊)")
    print(f"[2] WER               : {wer:.2f}%"
          f"  (del {100*total_dels/max(total_ref,1):.1f} /"
          f" ins {100*total_ins/max(total_ref,1):.1f} /"
          f" sub {100*total_subs/max(total_ref,1):.1f})")
    if criterion is not None:
        print(f"[4] dev L_R / L_T     : {sum_lr/n_batches:.2f} / {sum_lt/n_batches:.2f}"
              f"  (λ_R·L_R と λ_T·L_T の実効比を確認)")
    if optimizer is not None:
        print(f"[5] current lr        : {optimizer.param_groups[0]['lr']:.2e}"
              f"  (<1e-5 ならスケジューラ早期崩壊)")
    print(f"[6] features          : NaN={feat_stats['nan']}"
          f"  range=[{feat_stats['min']:.2f}, {feat_stats['max']:.2f}]")
    print("=" * 60)