import os.path

from torchaudio.models.decoder import ctc_decoder
from torchaudio.models import RNNTBeamSearch
import numpy as np
import torch

from Parameter.Parameter_VAC import CTC_WEIGHTS
from word_process import class2gloss_sequence,wer_calculation
from utils import load_phoenix_dataset
from torchmetrics.text import WordErrorRate
from ctcdecode import CTCBeamDecoder


def ctcdecode_WER_multilang(file_path,class2gloss,special_token:dict,auxiliary_pred=False,dataset="phoenix"):
    if type(file_path)==str:
        output=np.load(file_path,allow_pickle=True).item()
    elif type(file_path)!=str:
        output=file_path

    WER_calculator=WordErrorRate()
    WER_calculatorConv=WordErrorRate()
    wer_dict={}
    wer_list=[]
    gloss_sentence_dict={}
    for k ,v in output.items():
        feature=torch.tensor(v["feature"]).unsqueeze(0)
        dataset_id=v["dataset_id"]
        vocab = [x for x in class2gloss[dataset_id].values()]
        ctc_decoder2 = CTCBeamDecoder(vocab, beam_width=10, blank_id=0, num_processes=10)

        #v["conv_feature"]が存在しない場合はスキップ

        if "conv_feature"  in v.keys():
            conv_feature=torch.tensor(v["conv_feature"]).unsqueeze(0)
            conv_beam_result,beam_scores,timesteps,out_seq_len=ctc_decoder2.decode(conv_feature.float())
            if len(conv_beam_result[0])!=0:
                conv_first_result = conv_beam_result[0][0][:out_seq_len[0][0]].cpu().numpy()
                conv_hyp = class2gloss_sequence(dataset,conv_first_result, class2gloss,special_token)
            else:
                conv_hyp=[]
        else:
            conv_hyp=[]



        beam_result,beam_scores,timesteps,out_seq_len=ctc_decoder2.decode(feature.float())
        if len(beam_result[0])==0:
            continue
        beam_result=beam_result[0][0][:out_seq_len[0][0]].cpu().numpy()

        hyp=class2gloss_sequence(dataset,beam_result,class2gloss,special_token)
        targets=v["label"]
        if type(targets[0])==str:
            ref=targets
        else:
            ref=class2gloss_sequence(dataset,targets.cpu().numpy(),class2gloss,special_token)
        WER_calculator.update(" ".join(hyp), " ".join(ref))
        WER_calculatorConv.update(" ".join(conv_hyp), " ".join(ref))
        if len(hyp)==0:
            continue
        gloss_sentence_dict[k]={"hyp":hyp,"ref":ref,"conv_hyp":conv_hyp}
    if len(gloss_sentence_dict)==0:
        return 1e5
    ret=wer_calculation(gloss_sentence_dict,auxiliary_pred=auxiliary_pred)
    #ret=WER_calculator.compute()*100
    retConv=WER_calculatorConv.compute()*100

    print(f"WER:{ret['WER_primary']}%")
    print(f"WERConv:{retConv.item()}%")
    return ret

def save_beamsearch(file_path,class2gloss):
    if type(file_path)==str:
        output=np.load(file_path,allow_pickle=True).item()
    elif type(file_path)!=str:
        output=file_path
    vocab = ["_"]+[x for x in class2gloss.values()]+["|"]
    path_dict={}
    #ctc_decoder=ctcdecode.CTCBeamDecoder(vocab,beam_width=10,blank_id=0,num_processes=10)
    for k ,v in output.items():
        feature=torch.tensor(v["feature"]).unsqueeze(0).float()
        best_path=torch.argmax(feature,dim=2).squeeze(0).cpu().numpy()
        path_dict[k]=best_path.tolist()
    return path_dict


if __name__=="__main__":
    file_path="/media/jean/data_storage/CSLR/keyword_models/VAC/VAC_success/dev/result_word.npy"
    dataset=load_phoenix_dataset()
    wer=ctcdecode_WER(file_path,dataset["class2gloss"])
    print("WER:",wer)
    #np.mean(wer_list)をtxtに保存
    with open(f"{os.path.dirname(file_path)}/wer.txt","w") as f:
        f.write("平均WER:"+str(wer))

