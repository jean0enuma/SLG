import os
import torch
import numpy as np
from itertools import groupby
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode,
                 blank_id=0, beam_width=10):
        self.i2g_dict   = dict((v, k) for k, v in gloss_dict.items())
        self.g2i_dict   = {v: k for k, v in self.i2g_dict.items()}
        self.gloss_dict = gloss_dict
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id    = blank_id
        self.beam_width  = beam_width

        # torchaudio decoder にはトークン名(文字列)のリストが必要。
        # 元コードと同じく Unicode の私用領域を使えば、gloss 名と衝突しない。
        BLANK = "<blank>"
        tokens = []
        for idx in range(num_classes):
            if idx == blank_id:
                tokens.append(BLANK)
            elif idx in self.i2g_dict:
                tokens.append(chr(20000 + idx))   # gloss と無関係なユニークID
            else:
                tokens.append(chr(20000 + idx))
        self.tokens = tokens

        # lexicon-free CTC beam search
        # sil_token は API 上必須だが、本タスクでは silence 概念がないので blank を再利用。
        self.ctc_decoder = ctc_decoder(
            lexicon=None,
            tokens=tokens,
            beam_size=beam_width,
            blank_token=BLANK,
            sil_token=BLANK,
            nbest=1,
        )

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        """
        torchaudio.models.decoder.ctc_decoder の入出力:
            - 入力 emissions: (B, T, C), log-probs(CPU, float32)
            - 入力 lengths : (B,), int32
            - 出力        : List[List[CTCHypothesis]]
                            hyp.tokens   : 復号後のトークンID列(blank除去・collapse済み)
                            hyp.timesteps: 各トークンが出たT'インデックス
                            hyp.score    : 累積スコア
        """
        if probs:
            emissions = torch.log(nn_output.cpu().clamp(min=1e-10))
        else:
            emissions = nn_output.log_softmax(-1).cpu()
        emissions = emissions.contiguous().float()
        lengths   = vid_lgt.cpu().to(torch.int32)

        results = self.ctc_decoder(emissions, lengths)

        ret_list = []
        for batch_idx, hyps in enumerate(results):
            if len(hyps) == 0:
                ret_list.append([])
                continue
            best = hyps[0]
            token_ids = best.tokens.tolist() if torch.is_tensor(best.tokens) \
                        else list(best.tokens)

            # 念のため blank 除去 + 連続重複の縮約(本来 decoder 側で処理済み)
            token_ids = [t for t in token_ids if t != self.blank_id]
            token_ids = [x for x, _ in groupby(token_ids)]

            ret_list.append([
                (self.i2g_dict[int(gid)], gid)
                for idx, gid in enumerate(token_ids)
                if int(gid) in self.i2g_dict
            ])
        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
        return ret_list