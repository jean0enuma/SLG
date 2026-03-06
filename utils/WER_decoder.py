import copy
import os.path

from torchaudio.models.decoder import ctc_decoder
import numpy as np
import torch
from word_process import class2gloss_sequence, wer_calculation, remove_special_token
from utils import load_phoenix_dataset


def ctcdecode_WER_WithDecoder(file_path, class2gloss, analysis=False):
    if type(file_path) == str:
        output = np.load(file_path, allow_pickle=True).item()
    elif type(file_path) != str:
        output = file_path
    vocab = [x for x in class2gloss.values()]
    if "<blank>" not in vocab:
        vocab.append("<blank>")
    if "<sep>" not in vocab:
        vocab.append("<sep>")
    if "<pad>" not in vocab:
        vocab.append("<pad>")
    # ctc_decoder=ctcdecode.CTCBeamDecoder(vocab,beam_width=10,blank_id=0,num_processes=10)
    ctc_decoder2 = ctc_decoder(lexicon=None, tokens=vocab, beam_size=5, blank_token="<blank>", sil_token="<sep>",
                               unk_word="<pad>")
    wer_dict = {}
    wer_list = []
    gloss_sentence_dict = {}
    gloss_sentence_dict_decoder = {}
    result_dict = {}
    auxiliary = False
    for k, v in output.items():
        feature = torch.tensor(v["feature"]).unsqueeze(0)
        # v["conv_feature"]が存在しない場合はスキップ

        if "conv_feature" in v.keys():
            conv_feature = torch.tensor(v["conv_feature"]).unsqueeze(0)
            conv_beam_result = ctc_decoder2(conv_feature.float())
            if len(conv_beam_result[0]) != 0:
                conv_first_result = conv_beam_result[0][0].tokens.cpu().numpy()
                conv_result = remove_special_token(conv_first_result)
                conv_hyp = class2gloss_sequence(conv_result, class2gloss, "hyp")
                auxiliary=True
            else:
                conv_first_result = []
                conv_hyp = []
                auxiliary=False
        else:
            conv_first_result = []
            conv_hyp = []
        if "decoder_feature" in v.keys():
            decoder_feature = torch.tensor(v["decoder_feature"]).unsqueeze(0)
            # decoder_result=ctc_decoder2(decoder_feature.float())
            decoder_result = torch.argmax(decoder_feature, dim=2).squeeze(0).cpu().numpy()  # best path
            if len(decoder_result) != 0:
                # decoder_result=decoder_result[0][0].tokens.cpu().numpy()
                # decoder_result = remove_blank_and_duplicate(decoder_result)
                decoder_hyp = class2gloss_sequence(decoder_result, class2gloss, "hyp")
            else:
                decoder_result = []
                decoder_hyp = []
        else:
            decoder_result = []
            decoder_hyp = []

        # beam_result, beam_scores, timesteps, out_seq_len = ctc_decoder.decode(feature)
        beam_result = ctc_decoder2(feature.float())
        # beam_result2=beam_search(v["feature"],vocab,beam_width=10,blank_idx=0)
        # beam_result = beam_result[0][0][:out_seq_len[0][0]].cpu().numpy()
        if len(beam_result[0]) == 0:
            continue
        beam_result = beam_result[0][0].tokens.cpu().numpy()
        targets = v["label"].cpu().numpy()
        first_result = remove_special_token(beam_result)
        # first_result2 = remove_blank_and_duplicate(beam_result2)

        hyp = class2gloss_sequence(first_result, class2gloss, "hyp")
        if len(hyp) == 0:
            continue

        ref = class2gloss_sequence(targets, class2gloss, "ref")
        gloss_sentence_dict[k] = {"hyp": hyp, "ref": ref, "conv_hyp": conv_hyp, "decoder_hyp": decoder_hyp}
        gloss_sentence_dict_decoder[k] = {"hyp": decoder_hyp, "ref": ref, }
        result_dict[k] = {"hyp": hyp, "ref": ref, "conv_hyp": conv_hyp, "decoder_hyp": decoder_hyp,
                          "beam_result": beam_result.tolist(), "conv_beam_result": conv_first_result.tolist(),
                          "decoder_result": decoder_result.tolist()}

    if len(gloss_sentence_dict) == 0:
        return 0,0,result_dict
    ret_encoder = wer_calculation(gloss_sentence_dict, auxiliary_pred=False)
    if len(gloss_sentence_dict_decoder) == 0:
        return np.array(ret_encoder),0,result_dict
    ret_decoder = wer_calculation(gloss_sentence_dict_decoder)
    if analysis:
        return np.array(ret_encoder), np.array(ret_decoder), result_dict
    else:
        return np.array(ret_encoder), np.array(ret_decoder)


def save_beamsearch(file_path, class2gloss):
    if type(file_path) == str:
        output = np.load(file_path, allow_pickle=True).item()
    elif type(file_path) != str:
        output = file_path
    vocab = ["_"] + [x for x in class2gloss.values()] + ["|"]
    path_dict = {}
    # ctc_decoder=ctcdecode.CTCBeamDecoder(vocab,beam_width=10,blank_id=0,num_processes=10)
    for k, v in output.items():
        feature = torch.tensor(v["feature"]).unsqueeze(0).float()
        best_path = torch.argmax(feature, dim=2).squeeze(0).cpu().numpy()
        path_dict[k] = best_path.tolist()
    return path_dict


if __name__ == "__main__":
    file_path = "/media/jean/data_storage/CSLR/keyword_models/VAC/VAC_success/dev/result_word.npy"
    dataset = load_phoenix_dataset()
    wer = ctcdecode_WER(file_path, dataset["class2gloss"])
    print("WER:", wer)
    # np.mean(wer_list)をtxtに保存
    with open(f"{os.path.dirname(file_path)}/wer.txt", "w") as f:
        f.write("平均WER:" + str(wer))
