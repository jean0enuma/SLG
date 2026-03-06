import os.path

from torchaudio.models.decoder import ctc_decoder
import numpy as np
import torch
from word_process import class2gloss_sequence,wer_calculation,remove_special_token
from utils import load_phoenix_dataset
from torchmetrics.text import WordErrorRate as WER
from ctcdecode import CTCBeamDecoder

def CTCDecodeWERWithBatch(sequence_feature, targets, class2gloss):
    vocab =[x for x in class2gloss.values()]
    #for token in special_token.values():
    #    if token not in vocab:
    #        vocab.append(token)
    #ctc_decoder=ctcdecode.CTCBeamDecoder(vocab,beam_width=10,blank_id=0,num_processes=10)
    #ctc_decoder_module=ctc_decoder(lexicon=None,tokens=vocab,beam_size=5,blank_token="<blank>",sil_token="<sep>",unk_word="<pad>")
    ctc_decoder_module=CTCBeamDecoder(vocab,beam_width=10,blank_id=0,num_processes=10)
    batch_size,seq_len,feature_dim=sequence_feature.shape
    targets = targets.cpu().numpy()
    beam_result, beam_scores, timesteps, out_seq_len = ctc_decoder_module.decode(sequence_feature.float())
    beam_result=beam_result[:,0]
    for i in range(batch_size):
        beam_result[:,out_seq_len[i,0]:]=0
    return beam_result
def CTCDecodeWERWithBatch_multilang(sequence_feature, targets, class2gloss, dataset_id):
    vocab =[x for x in class2gloss.values()]
    #for token in special_token.values():
    #    if token not in vocab:
    #        vocab.append(token)
    #ctc_decoder=ctcdecode.CTCBeamDecoder(vocab,beam_width=10,blank_id=0,num_processes=10)
    #ctc_decoder_module=ctc_decoder(lexicon=None,tokens=vocab,beam_size=5,blank_token="<blank>",sil_token="<sep>",unk_word="<pad>")
    ctc_decoder_module=CTCBeamDecoder(vocab,beam_width=10,blank_id=0,num_processes=10)
    batch_size,seq_len,feature_dim=sequence_feature.shape
    targets = targets.cpu().numpy()
    beam_result, beam_scores, timesteps, out_seq_len = ctc_decoder_module.decode(sequence_feature.float())
    beam_result=beam_result[:,0]
    for i in range(batch_size):
        beam_result[:,out_seq_len[i,0]:]=0
    return beam_result

if __name__=="__main__":
    file_path="/media/jean/data_storage/CSLR/keyword_models/VAC/VAC_success/dev/result_word.npy"
    dataset=load_phoenix_dataset()
    wer=ctcdecode_WER(file_path,dataset["class2gloss"])
    print("WER:",wer)
    #np.mean(wer_list)をtxtに保存
    with open(f"{os.path.dirname(file_path)}/wer.txt","w") as f:
        f.write("平均WER:"+str(wer))

