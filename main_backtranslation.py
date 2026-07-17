import os,glob,shutil,random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np

from models.evaluation.nslt_skeleton import NSLTConfig,Sign2Text,train_step,eval_step
from utils.metrics import *
from Parameter.Parameter import *
from loader.coordinate_preprocess import *
from loader.data_loader import *
from SLG_datasets.SLG_datasets_Units import SLGText2UnitsDatasets
import pandas as pd
from models.module.Hand_gcn_vae_6d import bones_used_joints
torch.autograd.set_detect_anomaly(False)
# seedを固定
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
def integrate_path(id, path_list):
    integrated_path = []
    for path in path_list:
        integrated_path.append((id, path))
    return integrated_path
def create_word_tokenizer(corpus):
    # corpusの中のすべての単語を取得
    all_words = []
    text=corpus['annotation'].values
    for sentence in text:
        words = sentence.lower().split()
        if "." in words:
            words.remove(".")
        all_words.extend(words)
    # 単語の出現回数をカウント
    word_count = {}
    for word in all_words:
        if word not in word_count:
            word_count[word] = 0
        word_count[word] += 1
    # 出現回数が多い順にソート
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    # 単語とインデックスの対応表を作成
    word2index = {word: index+3 for index, (word, count) in enumerate(sorted_word_count)}
    index2word = {index+3: word for index, (word, count) in enumerate(sorted_word_count)}
    word2index['pad_token'] = 0
    word2index['bos_token'] = 1
    word2index['eos_token'] = 2
    index2word[0] = 'pad_token'
    index2word[1] = 'bos_token'
    index2word[2] = 'eos_token'
    return word2index, index2word
def main(config, dataset,save_path,evaluation=False):
    print("---Loading datasets---")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # trainデータのパスを結合(id,pathのタプルorリスト)
    train_data_path = []
    dev_data_path = []
    test_data_path = []
    train_corpus = {}
    dev_corpus = {}
    test_corpus = {}
    train_cod_root = {}
    dev_cod_root = {}
    test_cod_root = {}
    train_face_root = {}
    dev_face_root = {}
    test_face_root = {}
    i = 0
    is_islr=False
    if dataset=="phoenixT":

        phoenixT_train_path, phoenixT_dev_path, phoenixT_test_path, phoenixTgn_train_corpus, phoenixT_dev_corpus, phoenixT_test_corpus= datasets_loader_T(
            "phoenixT")
        train_corpus[1] = phoenixTgn_train_corpus
        dev_corpus[1] = phoenixT_dev_corpus
        test_corpus[1] = phoenixT_test_corpus
        train_cod_root[1] = SKELETON_TRAIN_DATADIR_T_3D
        dev_cod_root[1] = SKELETON_DEV_DATADIR_T_3D
        test_cod_root[1] = SKELETON_TEST_DATADIR_T_3D

        train_face_root[1] = FACE_TRAIN_DATADIR_T_3D
        dev_face_root[1] = FACE_DEV_DATADIR_T_3D
        test_face_root[1] = FACE_TEST_DATADIR_T_3D
        train_data_path += integrate_path(1, phoenixT_train_path)
        dev_data_path += integrate_path(1, phoenixT_dev_path)
        test_data_path += integrate_path(1, phoenixT_test_path)
        tokenizer,decoder=create_word_tokenizer(pd.concat([train_corpus[1],dev_corpus[1],test_corpus[1]],axis=0))
        i += 1
    elif dataset=="CSL-Daily":
        csl_daily_train_path, csl_daily_dev_path, csl_daily_test_path, csl_daily_train_corpus, csl_daily_dev_corpus, csl_daily_test_corpus = datasets_loader_T(
            "CSL-Daily")

        train_corpus[1] = csl_daily_train_corpus
        dev_corpus[1] = csl_daily_dev_corpus
        test_corpus[1] = csl_daily_test_corpus
        if config['dataset_parameters']['is_processed']:
            train_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_PROCESSED
            dev_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_PROCESSED
            test_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_PROCESSED

            train_face_root[1] = FACE_CSL_DAILY_DATADIR_PROCESSED
            dev_face_root[1] = FACE_CSL_DAILY_DATADIR_PROCESSED
            test_face_root[1] = FACE_CSL_DAILY_DATADIR_PROCESSED
            is_3d = True
        else:
            train_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_3D
            dev_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_3D
            test_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_3D

            train_face_root[1] = FACE_CSL_DAILY_DATADIR_3D
            dev_face_root[1] = FACE_CSL_DAILY_DATADIR_3D
            test_face_root[1] = FACE_CSL_DAILY_DATADIR_3D
            is_3d = True
        train_data_path += integrate_path(1, csl_daily_train_path)
        dev_data_path += integrate_path(1, csl_daily_dev_path)
        test_data_path += integrate_path(1, csl_daily_test_path)
        tokenizer,decoder=create_word_tokenizer(pd.concat([train_corpus[1],dev_corpus[1],test_corpus[1]],axis=0))

        i += 1
    elif dataset=="how2sign":
        how2sign_train_path, how2sign_dev_path, how2sign_test_path, how2sign_train_corpus, how2sign_dev_corpus, how2sign_test_corpus = datasets_loader_T(
            "how2sign")
        train_corpus[2] = how2sign_train_corpus
        dev_corpus[2] = how2sign_dev_corpus
        test_corpus[2] = how2sign_test_corpus
        if config['dataset_parameters']['is_processed']:
            train_cod_root[2] = SKELETON_HOW2SIGN_TRAIN_DATADIR_PROCESSED
            dev_cod_root[2] = SKELETON_HOW2SIGN_DEV_DATADIR_PROCESSED
            test_cod_root[2] = SKELETON_HOW2SIGN_TEST_DATADIR_PROCESSED

            train_face_root[2] = FACE_HOW2SIGN_TRAIN_DATADIR_PROCESSED
            dev_face_root[2] = FACE_HOW2SIGN_DEV_DATADIR_PROCESSED
            test_face_root[2] = FACE_HOW2SIGN_TEST_DATADIR_PROCESSED
            is_3d = True
        else:
            train_cod_root[2] = SKELETON_HOW2SIGN_TRAIN_DATADIR_3D
            dev_cod_root[2] = SKELETON_HOW2SIGN_DEV_DATADIR_3D
            test_cod_root[2] = SKELETON_HOW2SIGN_TEST_DATADIR_3D

            train_face_root[2] = FACE_HOW2SIGN_TRAIN_DATADIR_3D
            dev_face_root[2] = FACE_HOW2SIGN_DEV_DATADIR_3D
            test_face_root[2] = FACE_HOW2SIGN_TEST_DATADIR_3D
            is_3d = True

        train_data_path += integrate_path(2, how2sign_train_path)
        dev_data_path += integrate_path(2, how2sign_dev_path)
        test_data_path += integrate_path(2, how2sign_test_path)
        tokenizer,decoder=create_word_tokenizer(pd.concat([train_corpus[2],dev_corpus[2],test_corpus[2]],axis=0))

        i += 1.
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    print("Datasets loaded.")
    print("保存場所:", save_path)
    print("Is GPU available?:", torch.cuda.is_available())
    print(f"vocab size: {len(tokenizer)}")
    device = config["device"] if torch.cuda.is_available() else "cpu"
    # deviceからgpuの名前を取得して表示
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(int(device[-1]))
        print("GPU name:", gpu_name)
    print("---Loading tokenizer---")
    #tokenizer=create_word_tokenizer(train_corpus)
    print("---Creating datasets---")
    ds_train = SLGText2UnitsDatasets(train_data_path, train_cod_root, train_face_root, is_3d=True,
                                     is_processed=False, is_sg_filter=False,
                                     is_coarse=False,
                                     trainable=True, tokenizer=tokenizer,texts_corpus=train_corpus,is_islr=is_islr,
                                     scale_ratio=(0.8,1.2),gloss2class=None,is_delete_cod=False,is_norm=False)
    ds_dev = SLGText2UnitsDatasets(dev_data_path, dev_cod_root, dev_face_root, trainable=False, is_3d=True,
                                   is_processed=False, is_sg_filter=False,
                                   is_coarse=False,tokenizer=tokenizer,texts_corpus=dev_corpus,is_islr=is_islr,
                                   scale_ratio=(0.8,1.2),gloss2class=None,is_delete_cod=False,is_norm=False)
    ds_test = SLGText2UnitsDatasets(test_data_path, test_cod_root, test_face_root, trainable=False, is_3d=True,
                                    is_processed=False, is_sg_filter=False,
                                    is_coarse=False,tokenizer=tokenizer,texts_corpus=test_corpus,is_islr=is_islr,
                                    scale_ratio=(0.8,1.2),gloss2class=None,is_delete_cod=False,is_norm=False)
    dl_train=torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=4, collate_fn=ds_train.collate_fn,drop_last=True)
    dl_dev=torch.utils.data.DataLoader(ds_dev, batch_size=32, shuffle=False, num_workers=4, collate_fn=ds_dev.collate_fn)
    if evaluation==False:
        dl_test=torch.utils.data.DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=4, collate_fn=ds_test.collate_fn)
    else:
        dl_test=torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=ds_test.collate_fn)
    model_config=NSLTConfig(
        input_dim= 144,  # 骨格座標次元 (例: 50 keypoints × (x,y,z))
        embed_dim= 512,
        hidden_dim= 1000,  # 論文: 1000 hidden units
        num_layers= 4,  # 論文: 4 stacked residual layers
        dropout= 0.2,  # 論文: drop probability 0.2
        attention="luong",  # 論文: Luong が最良 (Table 3/5)
        vocab_size= len(tokenizer),  # PHOENIX14T ドイツ語語彙 2887 + 特殊トークン
        label_smoothing= 0.0
    )
    model=Sign2Text(model_config)
    model=model.to(device)
    bones=ALL_BONES

    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
    epochs=50
    min_dev_loss=float("inf")
    if evaluation==False:
        for epoch in range(epochs):
            model.train()
            train_avg_loss=[]
            dev_avg_loss=[]
            test_avg_loss=[]
            for batch in dl_train:
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens = batch
                padded_cod_data = padded_cod_data.float().to(device)
                text_tokens = text_tokens['input_ids'].to(device)
                input_length_tensor = input_length_tensor.to(device)
                all_inputs, body_inputs, left_inputs, right_inputs, norm_info = normalize_batch(padded_cod_data,
                                                                                                True)
                new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                new_inputs=new_inputs.reshape(new_inputs.shape[0],new_inputs.shape[1],-1)
                loss=train_step(model,optimizer,(new_inputs,input_length_tensor,text_tokens))
                train_avg_loss.append(loss)
            print("Epoch {}: Train loss: {:.4f}".format(epoch, np.mean(train_avg_loss)))
            model.eval()
            with torch.no_grad():
                for batch in dl_dev:
                    padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens = batch
                    padded_cod_data = padded_cod_data.float().to(device)
                    text_tokens = text_tokens['input_ids'].to(device)
                    input_length_tensor = input_length_tensor.to(device)
                    all_inputs, body_inputs, left_inputs, right_inputs, norm_info = normalize_batch(padded_cod_data,
                                                                                                    True)
                    new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                    new_inputs = new_inputs.reshape(new_inputs.shape[0], new_inputs.shape[1], -1)
                    loss=eval_step(model,(new_inputs,input_length_tensor,text_tokens))
                    dev_avg_loss.append(loss)
                print("Epoch {}: Dev loss: {:.4f}".format(epoch, np.mean(dev_avg_loss)))
                for batch in dl_test:
                    padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens = batch
                    padded_cod_data = padded_cod_data.float().to(device)
                    text_tokens = text_tokens['input_ids'].to(device)
                    input_length_tensor = input_length_tensor.to(device)
                    all_inputs, body_inputs, left_inputs, right_inputs, norm_info = normalize_batch(padded_cod_data,
                                                                                                    True)
                    new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                    new_inputs = new_inputs.reshape(new_inputs.shape[0], new_inputs.shape[1], -1)
                    loss=eval_step(model,(new_inputs,input_length_tensor,text_tokens))
                    test_avg_loss.append(loss)
                print("Epoch {}: Test loss: {:.4f}".format(epoch, np.mean(test_avg_loss)))
            scheduler.step(np.mean(dev_avg_loss))
            if np.mean(dev_avg_loss) < min_dev_loss:
                min_dev_loss = np.mean(dev_avg_loss)
                torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
                print("Best model saved.")
    else:
        model.load_state_dict(torch.load(os.path.join(save_path, "best_model.pth")))
        model.eval()
        ref_list=[]
        pred_list=[]
        with torch.no_grad():
            for batch in dl_test:
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens = batch
                padded_cod_data = padded_cod_data.float().to(device)
                text_tokens = text_tokens['input_ids'].to(device)
                input_length_tensor = input_length_tensor.to(device)
                all_inputs, body_inputs, left_inputs, right_inputs, norm_info = normalize_batch(padded_cod_data,
                                                                                                True)
                new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                new_inputs = new_inputs.reshape(new_inputs.shape[0], new_inputs.shape[1], -1)
                output=model.beam_search(new_inputs,input_length_tensor,beam_width=5)
                pred_sentence = " ".join([decoder[token] for token in output if token not in [0,1,2]])
                ref_tokens=text_tokens[0].tolist()
                ref_sentence=" ".join([decoder[token] for token in ref_tokens if token not in [0,1,2]])
                pred_list.append(pred_sentence)
                ref_list.append(ref_sentence)
        bleu_score=bleu(pred_list,ref_list)
        rouge_score=rouge(pred_list,ref_list)
        print("BLEU score:", bleu_score)
        print("ROUGE score:", rouge_score)





if __name__=="__main__":
    config = {
        "device": "cuda:0",
        "dataset_parameters": {
            "is_processed": False
        }
    }
    dataset = "phoenixT"
    save_path = "./results_nslt"
    main(config, dataset, save_path, evaluation=True)