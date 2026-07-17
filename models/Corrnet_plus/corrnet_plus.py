import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.Corrnet_plus.criterions import SeqKD
from models.Corrnet_plus.BiLSTM  import BiLSTMLayer
from models.Corrnet_plus.tconv import TemporalConv
from models.Corrnet_plus.resnet import resnet18 as resnet
import torchaudio.functional as AF
import torchaudio.functional as AF
from models.Corrnet_plus.decode import Decode


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes,  conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        #self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d = resnet()
        self.conv2d.fc = Identity()

        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        #self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            #inputs = x.reshape(batch * temp, channel, height, width)
            #framewise = self.masked_bn(inputs, len_x)
            #framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
            framewise = self.conv2d(x.permute(0,2,1,3,4)).view(batch, temp, -1).permute(0,2,1) # btc -> bct
        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            #"framewise_features": framewise,
            #"visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
            "loss_LiftPool_u": conv1d_outputs['loss_LiftPool_u'],
            "loss_LiftPool_p": conv1d_outputs['loss_LiftPool_p'],
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        total_loss = {}
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                total_loss['ConvCTC'] = weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
                loss += total_loss['ConvCTC']
            elif k == 'SeqCTC':
                total_loss['SeqCTC'] = weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
                loss += total_loss['SeqCTC']
            elif k == 'Dist':
                total_loss['Dist'] = weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
                loss += total_loss['Dist']
            elif k == 'Cu':
                total_loss['Cu'] = weight * ret_dict["loss_LiftPool_u"]
                loss += total_loss['Cu']
            elif k == 'Cp':
                total_loss['Cp'] = weight * ret_dict["loss_LiftPool_p"]
                loss += total_loss['Cp']
        return loss, total_loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss

    @torch.no_grad()
    def alignment(self, x, len_x, label=None, label_lgt=None,
                  mode="forced", orig_len=None, min_clip=8,
                  blank_mode="forward", overlap=True):
        """
        Args:
            mode      : "forced" / "midpoint"
            blank_mode: "forward" / "midpoint"  (mode="forced" かつ overlap=False のとき有効)
            overlap   : False — 重複なし・隙間なし分割
                        True  — 重複あり(受容野ベース)
            min_clip  : 最小クリップ長。overlap=True のときは適用しない
                        (重複ありでは区間長補正が意味を持たないため)
        """
        assert mode in ("midpoint", "forced")
        assert blank_mode in ("forward", "midpoint")

        self.eval()
        output = self.forward(x, len_x, label, label_lgt)

        log_probs = output['sequence_logits'].log_softmax(-1).permute(1, 0, 2).contiguous()
        feat_len = output['feat_len']
        rf, jump = self._get_temporal_field()
        B = log_probs.size(0)
        device = log_probs.device
        inv_gloss = dict(self.decoder.i2g_dict) if hasattr(self.decoder, 'i2g_dict') else {}

        def get_orig_T(b):
            if orig_len is not None:
                return int(orig_len[b].item() if torch.is_tensor(orig_len) else orig_len[b])
            return int(feat_len[b].item()) * jump

        if mode == "midpoint":
            return self._align_midpoint(output, feat_len, jump, rf, inv_gloss,
                                        B, get_orig_T, min_clip, overlap)
        else:
            return self._align_forced(output, log_probs, feat_len, jump, rf,
                                      label, label_lgt, inv_gloss, B, device,
                                      get_orig_T, min_clip, blank_mode, overlap)

    # ========================= 方法1: midpoint =========================

    def _align_midpoint(self, output, feat_len, jump, rf, inv_gloss,
                        B, get_orig_T, min_clip, overlap):
        emissions = output['sequence_logits'].log_softmax(-1).permute(1, 0, 2).contiguous().cpu().float()
        results = self.decoder.ctc_decoder(emissions, feat_len.cpu().to(torch.int32))

        batch_alignments = []
        for b in range(B):
            orig_T = get_orig_T(b)
            hyps = results[b]
            if not hyps or len(hyps[0].tokens) == 0:
                batch_alignments.append([])
                continue
            best = hyps[0]
            toks = best.tokens.tolist()
            steps = best.timesteps.tolist()

            if overlap:
                # 各 gloss = その timestep の受容野
                sample = []
                for tok, ts in zip(toks, steps):
                    start = ts * jump
                    end = ts * jump + rf - 1
                    sample.append({
                        "token": int(tok),
                        "gloss": inv_gloss.get(int(tok), str(int(tok))),
                        "t_start": int(max(0, start)),
                        "t_end": int(min(end, orig_T - 1)),
                    })
                batch_alignments.append(sample)
                continue

            # overlap=False: 従来の中点分割
            sample = []
            for tok, ts in zip(toks, steps):
                t_center = min(int(ts) * jump + jump // 2, orig_T - 1)
                sample.append({
                    "token": int(tok),
                    "gloss": inv_gloss.get(int(tok), str(int(tok))),
                    "t_center": int(t_center),
                })
            centers = [s["t_center"] for s in sample]
            for i, s in enumerate(sample):
                s["t_start"] = 0 if i == 0 else (centers[i - 1] + centers[i]) // 2
                s["t_end"] = orig_T - 1 if i == len(sample) - 1 \
                    else (centers[i] + centers[i + 1]) // 2 - 1
            sample[0]["t_start"] = 0
            sample[-1]["t_end"] = orig_T - 1
            for i in range(len(sample) - 1):
                sample[i + 1]["t_start"] = sample[i]["t_end"] + 1
            sample = self._clip_and_enforce_minlen(sample, orig_T, min_clip)
            batch_alignments.append(sample)
        return batch_alignments

    # ========================= 方法2: forced =========================
    def _align_forced(self, output, log_probs, feat_len, jump, rf,
                      label, label_lgt, inv_gloss, B, device,
                      get_orig_T, min_clip, blank_mode, overlap):
        """forced alignment のフレームラベルを区間化(blank_mode で blank 割当規則を選択)。"""
        import torchaudio.functional as AF

        if label is not None and label_lgt is not None:
            sample_targets = [label[b, :int(label_lgt[b].item())].long().tolist()
                              for b in range(B)]
        else:
            beam_pred = self.decoder.decode(output['sequence_logits'], feat_len,
                                            batch_first=False, probs=False)
            sample_targets = []
            for b in range(B):
                tokens = []
                for item in beam_pred[b]:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        g = item[0]
                        if hasattr(self.decoder, 'g2i_dict') and g in self.decoder.g2i_dict:
                            tokens.append(int(self.decoder.g2i_dict[g]))
                    elif isinstance(item, int):
                        tokens.append(item)
                sample_targets.append(tokens)

        batch_alignments = []
        for b in range(B):
            T_prime = int(feat_len[b].item())
            lp_b = log_probs[b, :T_prime].unsqueeze(0)
            orig_T = get_orig_T(b)
            targets = sample_targets[b]
            if len(targets) == 0:
                batch_alignments.append([])
                continue

            if T_prime < 2 * len(targets) + 1:
                frame_labels = lp_b.squeeze(0).argmax(dim=-1)
            else:
                tgt = torch.tensor(targets, dtype=torch.int32, device=device).unsqueeze(0)
                in_len = torch.tensor([T_prime], dtype=torch.int32, device=device)
                tgt_len = torch.tensor([len(targets)], dtype=torch.int32, device=device)
                paths, _ = AF.forced_align(lp_b, tgt, in_len, tgt_len, blank=0)
                frame_labels = paths.squeeze(0).long()

            sample = self._split_segments(
                frame_labels, orig_T, jump, rf,
                blank=0, inv_gloss=inv_gloss,
                overlap=overlap, blank_mode=blank_mode)
            if not overlap:
                sample = self._clip_and_enforce_minlen(sample, orig_T, min_clip)
            else:
                # 重複ありでも実フレームクリップだけは行う
                for s in sample:
                    s["t_start"] = max(0, min(s["t_start"], orig_T - 1))
                    s["t_end"] = max(0, min(s["t_end"], orig_T - 1))
            batch_alignments.append(sample)
        return batch_alignments
    # ========================= 共通ヘルパー =========================

    def _get_temporal_field(self):
        """TemporalConv の kernel_size 列から受容野(rf)と累積 stride(jump)を計算。"""
        rf, jump = 1, 1
        for ks in self.conv1d.kernel_size:
            k = int(ks[1])
            if ks[0] == 'K':  # Conv1d: kernel=k, stride=1, padding=0
                rf += (k - 1) * jump
            elif ks[0] == 'P':  # MaxPool1d: kernel=k, stride=k
                rf += (k - 1) * jump
                jump *= k
        return rf, jump

    @staticmethod
    def _fill_blanks(frame_labels, blank=0):
        """blank フレームを最近傍の非 blank ラベルに吸収(中間は前後で折半)。"""
        labels = list(frame_labels)
        n = len(labels)
        nz = [i for i, v in enumerate(labels) if v != blank]
        if not nz:
            return labels
        filled = labels[:]
        for i in range(nz[0]):
            filled[i] = labels[nz[0]]
        for i in range(nz[-1] + 1, n):
            filled[i] = labels[nz[-1]]
        for a, b in zip(nz, nz[1:]):
            if b - a > 1:
                mid = (a + b) // 2
                for i in range(a + 1, mid + 1):
                    filled[i] = labels[a]
                for i in range(mid + 1, b):
                    filled[i] = labels[b]
        return filled

    @staticmethod
    def _fill_blanks_forward(frame_labels, blank=0):
        """
        各 blank フレームを「次に出現する非 blank ラベル」で埋める。
        = ある gloss の前方の blank をすべてその gloss に含める。

        末尾側の blank(後ろに gloss が無い)は、最後の非 blank ラベルに吸収。
        """
        labels = list(frame_labels)
        n = len(labels)
        nz = [i for i, v in enumerate(labels) if v != blank]
        if not nz:
            return labels  # 全部 blank(復号失敗)
        filled = labels[:]
        last_nz = nz[-1]
        next_label = labels[last_nz]  # 後ろから走査するための初期値
        for i in range(n - 1, -1, -1):
            if i > last_nz:
                filled[i] = labels[last_nz]  # 末尾 blank → 最後の gloss
            elif labels[i] != blank:
                next_label = labels[i]  # 非 blank: 以降の blank の埋め先を更新
                filled[i] = labels[i]
            else:
                filled[i] = next_label  # blank → 次の gloss
        return filled

    def _split_segments(self, frame_labels, orig_T, jump, rf,
                        blank=0, inv_gloss=None,
                        overlap=False, blank_mode="forward"):
        """
        forced alignment のフレームラベルから元フレーム区間を作る。

        Args:
            overlap   : False — 重複なし・隙間なし(jump 単位の担当ブロック)
                        True  — 重複あり(各 gloss の受容野 rf 全体を区間とする)
            blank_mode: overlap=False のときの blank 割り当て規則
                        "forward"  — 前方 blank を後ろの gloss に吸収
                        "midpoint" — gloss 間 blank を前後で折半
                        ※ overlap=True のときは blank を埋めず単に除外するため無視
        """
        if torch.is_tensor(frame_labels):
            frame_labels = frame_labels.cpu().tolist()
        T_prime = len(frame_labels)
        inv_gloss = inv_gloss or {}

        # ---- 区間化 ----
        if overlap:
            # blank を埋めず、非 blank の連続だけを区間にする
            segs, prev, seg_start = [], None, 0
            for t in range(T_prime):
                v = frame_labels[t]
                if v != prev:
                    if prev is not None and prev != blank:
                        segs.append((prev, seg_start, t - 1))
                    seg_start, prev = t, v
            if prev is not None and prev != blank:
                segs.append((prev, seg_start, T_prime - 1))
        else:
            # blank を埋めてから連続同一ラベルを区間化
            if blank_mode == "forward":
                filled = self._fill_blanks_forward(frame_labels, blank)
            else:
                filled = self._fill_blanks(frame_labels, blank)
            if all(v == blank for v in filled):
                return []
            segs, prev, seg_start = [], None, 0
            for t in range(T_prime):
                if filled[t] != prev:
                    if prev is not None:
                        segs.append((prev, seg_start, t - 1))
                    seg_start, prev = t, filled[t]
            segs.append((prev, seg_start, T_prime - 1))

        if not segs:
            return []

        # ---- T' 区間 → 元フレーム区間 ----
        out = []
        for tok, ts, te in segs:
            if overlap:
                # 受容野全体: [ts*jump, te*jump + rf - 1]
                start = ts * jump
                end = te * jump + rf - 1
            else:
                # 担当ブロック: [ts*jump, (te+1)*jump - 1]
                start = ts * jump
                end = (te + 1) * jump - 1
            out.append({
                "token": int(tok),
                "gloss": inv_gloss.get(int(tok), str(int(tok))),
                "t_start": int(max(0, start)),
                "t_end": int(min(end, orig_T - 1)),
            })

        # ---- 端の補正 ----
        if not overlap:
            # 重複なし: 連続性を保証(隙間なし)
            out[0]["t_start"] = 0
            out[-1]["t_end"] = orig_T - 1
            for i in range(len(out) - 1):
                out[i + 1]["t_start"] = out[i]["t_end"] + 1
        # overlap=True のときは重複を許すので連続性補正はしない
        return out

    def _clip_and_enforce_minlen(self, sample, orig_T, min_clip):
        """
        動画クリップ用に区間を補正する。
          1. 実フレーム [0, orig_T-1] にクリップ
          2. 各区間に min_clip フレームの最小長を保証(隣から時間を借りる)
          3. 連続性を回復(隙間なし・重複なし)

        sample : [{"token","gloss","t_start","t_end"}, ...] 原座標・t_start昇順
        orig_T : 実動画長 L
        min_clip: 1区間の最小フレーム数(静止画回避のしきい値)
        """
        if not sample:
            return []
        n = len(sample)

        # --- 1. 実フレームにクリップ + 端を固定 ---
        for s in sample:
            s["t_start"] = max(0, min(int(s["t_start"]), orig_T - 1))
            s["t_end"] = max(0, min(int(s["t_end"]), orig_T - 1))
        sample[0]["t_start"] = 0
        sample[-1]["t_end"] = orig_T - 1

        # min_clip が動画長を超える場合は全体で頭割り
        if min_clip * n > orig_T:
            min_clip = max(1, orig_T // n)

        # --- 2. 前方からスキャンし、短い区間を次から借りて拡張 ---
        for i in range(n):
            width = sample[i]["t_end"] - sample[i]["t_start"] + 1
            if width >= min_clip:
                continue
            need = min_clip - width
            if i < n - 1:
                # 次の区間の先頭を need ぶん後ろにずらす(借りる)
                # ただし次の区間が min_clip を割らない範囲で
                nxt = sample[i + 1]
                nxt_width = nxt["t_end"] - nxt["t_start"] + 1
                give = min(need, max(0, nxt_width - min_clip))
                sample[i]["t_end"] += give
                sample[i + 1]["t_start"] = sample[i]["t_end"] + 1
            else:
                # 最後の区間: 前から借りる
                prv = sample[i - 1]
                prv_width = prv["t_end"] - prv["t_start"] + 1
                give = min(need, max(0, prv_width - min_clip))
                sample[i]["t_start"] -= give
                sample[i - 1]["t_end"] = sample[i]["t_start"] - 1

        # --- 3. 連続性の最終回復 ---
        sample[0]["t_start"] = 0
        sample[-1]["t_end"] = orig_T - 1
        for i in range(n - 1):
            if sample[i + 1]["t_start"] != sample[i]["t_end"] + 1:
                sample[i + 1]["t_start"] = sample[i]["t_end"] + 1
            # 借り過ぎで逆転したらならす
            if sample[i]["t_end"] < sample[i]["t_start"]:
                sample[i]["t_end"] = sample[i]["t_start"]
                sample[i + 1]["t_start"] = sample[i]["t_end"] + 1
        return sample