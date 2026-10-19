import os,glob,shutil,random

from torchvision.transforms.v2.functional import jpeg_image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
import yaml

from models.evaluation.sign_language_transformer import *
from utils.metrics import *
from utils.diagnose import *
from Parameter.Parameter import *
from loader.coordinate_preprocess import *
from loader.data_loader import *
from SLG_datasets.SLTdatasets import SLTDatasets
import pandas as pd
from models.module.Hand_gcn_vae_6d import bones_used_joints, hand_joints_to_6d,reconstruct_joints_from_6d
from models.module.transformer6d_vae import HandTransformerVAE

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
def normalize_batch(padded_cod_data, used_3d,is_openpose=False):
    all_inputs   = padded_cod_data.permute(0, 1, 3, 2).contiguous()
    body_inputs  = padded_cod_data[:, :, :-42].permute(0, 1, 3, 2).clone()
    left_inputs  = padded_cod_data[:, :, -42:-21].permute(0, 1, 3, 2).clone()
    right_inputs = padded_cod_data[:, :, -21:].permute(0, 1, 3, 2).clone()

    if not is_openpose:
        left_inputs[...,0]=body_inputs[...,15].clone()
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
def create_word_tokenizer(corpus,key='annotation',is_lower=True):
    # corpusの中のすべての単語を取得
    all_words = []
    text=corpus[key].values
    for sentence in text:
        if is_lower:
            words = sentence.lower().split()
        else:
            words = sentence.split()
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
    word2index['<pad>'] = 0
    word2index['<s>'] = 1
    word2index['</s>'] = 2
    index2word[0] = '<pad>'
    index2word[1] = '<s>'
    index2word[2] = '</s>'
    return word2index, index2word
def rotation_6d_to_matrix(d6):
    # d6: (..., 6) → (..., 3, 3)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)  # 行として積む場合。列基底なら dim=-1
def main(config, dataset, save_path, evaluation=False, is_openpose=False, eval_data=None, vae_weights=None, use_3d=False, only_gloss=False,lr=1e-3):
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

        phoenixT_train_path, phoenixT_dev_path, phoenixT_test_path, phoenixTgn_train_corpus, phoenixT_dev_corpus, phoenixT_test_corpus= datasets_loader_gloss_trans(
            "phoenixT")
        gloss2class=np.load(f"./Parameter/gloss_dict_T.npy", allow_pickle=True).item()
        print("min:", min(gloss2class.values()), " 0を使うgloss:", [k for k, v in gloss2class.items() if v == 0])

        gloss2class['blank']=0
        class2gloss={v:k for k,v in gloss2class.items()}
        train_corpus[1] = phoenixTgn_train_corpus
        dev_corpus[1] = phoenixT_dev_corpus
        test_corpus[1] = phoenixT_test_corpus
        train_cod_root[1] = SKELETON_TRAIN_DATADIR_T_3D if not is_openpose else SKELETON_TRAIN_DATADIR_T_OPENPOSE_PROCESSED
        dev_cod_root[1] = SKELETON_DEV_DATADIR_T_3D if not is_openpose else SKELETON_DEV_DATADIR_T_OPENPOSE_PROCESSED
        test_cod_root[1] = SKELETON_TEST_DATADIR_T_3D if not is_openpose else SKELETON_TEST_DATADIR_T_OPENPOSE_PROCESSED

        train_face_root[1] = FACE_TRAIN_DATADIR_T_3D if not is_openpose else FACE_TRAIN_DATADIR_T_OPENPOSE_PROCESSED
        dev_face_root[1] = FACE_DEV_DATADIR_T_3D if not is_openpose else FACE_DEV_DATADIR_T_OPENPOSE_PROCESSED
        test_face_root[1] = FACE_TEST_DATADIR_T_3D if not is_openpose else FACE_TEST_DATADIR_T_OPENPOSE_PROCESSED
        train_data_path += integrate_path(1, phoenixT_train_path) if   not is_openpose else integrate_path(1, phoenixT_train_path)
        dev_data_path += integrate_path(1, phoenixT_dev_path) if not is_openpose else integrate_path(1, phoenixT_dev_path)
        test_data_path += integrate_path(1, phoenixT_test_path) if not is_openpose else integrate_path(1, phoenixT_test_path)
        tokenizer,decoder=create_word_tokenizer(pd.concat([train_corpus[1],dev_corpus[1],test_corpus[1]],axis=0),key="translation")
        i += 1
    elif dataset=="CSL-Daily":
        csl_daily_train_path, csl_daily_dev_path, csl_daily_test_path, csl_daily_train_corpus, csl_daily_dev_corpus, csl_daily_test_corpus = datasets_loader_T(
            "CSL-Daily")
        gloss2class=np.load(f"./Parameter/gloss_dict_CSL-Daily.npy", allow_pickle=True).item()
        class2gloss={v:k for k,v in gloss2class.items()}
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
    ds_train =SLTDatasets(train_data_path, train_cod_root, train_face_root, is_3d=True,
                                     is_processed=False, is_sg_filter=False,
                                     trainable=True, tokenizer=tokenizer,texts_corpus=train_corpus,
                                     scale_ratio=(1.0,1.0),gloss2class=gloss2class,is_delete_cod=False,is_norm=False,is_openpose=is_openpose)
    ds_dev = SLTDatasets(dev_data_path, dev_cod_root, dev_face_root, trainable=False, is_3d=True,
                                   is_processed=False, is_sg_filter=False,
                                   tokenizer=tokenizer,texts_corpus=dev_corpus,
                                   scale_ratio=(1.0,1.0),gloss2class=gloss2class,is_delete_cod=False,is_norm=False,is_openpose=is_openpose)
    ds_test = SLTDatasets(test_data_path, test_cod_root, test_face_root, trainable=False, is_3d=True,
                                    is_processed=False, is_sg_filter=False,
                                    tokenizer=tokenizer,texts_corpus=test_corpus,
                                    scale_ratio=(1.0,1.0),gloss2class=gloss2class,is_delete_cod=False,is_norm=False,is_openpose=is_openpose)
    dl_train=torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=4, collate_fn=ds_train.collate_fn,drop_last=True)
    dl_dev=torch.utils.data.DataLoader(ds_dev, batch_size=32, shuffle=False, num_workers=4, collate_fn=ds_dev.collate_fn)
    if evaluation==False:
        dl_test=torch.utils.data.DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=4, collate_fn=ds_test.collate_fn)
    else:
        dl_test=torch.utils.data.DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=4, collate_fn=ds_test.collate_fn)
    trans_vocab_size=len(tokenizer)
    gloss_vocab_size=len(gloss2class)
    print("translation vocab size:", trans_vocab_size)
    print("gloss vocab size:", gloss_vocab_size)
    model=SignLanguageTransformer(input_size=144 if use_3d else 282,
                                  gloss_vocab_size=gloss_vocab_size,
                                  text_vocab_size=trans_vocab_size,
                                  d_model=256,
                                  d_ff=2048,num_layers=3,n_heads=8)
    #model=STSignLanguageTransformer(input_size=6,
    #                                gloss_vocab_size=gloss_vocab_size,
    #                                text_vocab_size=trans_vocab_size, )

    model=model.to(device)
    bones=ALL_BONES if not is_openpose else ALL_BONES_OPENPOSE

    optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.998),weight_decay=1e-3)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=7)
    #scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.2)
    #scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
    if only_gloss:
        criterion=JointLoss(txt_pad_idx=tokenizer['<pad>'],blank_idx=gloss2class['blank'],lambda_r=1.0,lambda_t=0.0)
    else:
        criterion=JointLoss(txt_pad_idx=tokenizer['<pad>'],blank_idx=gloss2class['blank'],lambda_r=5.0,lambda_t=1.0)
    epochs=5000
    last_best_lr=-1
    batch = next(iter(dl_train))
    pd_, _, lens, *_ = batch
    inv_l = inv_r = tot = 0
    for b in range(len(lens)):
        L = lens[b]
        inv_l += (pd_[b, :L, -42:-21].abs().sum(dim=(1, 2)) == 0).sum().item()
        inv_r += (pd_[b, :L, -21:].abs().sum(dim=(1, 2)) == 0).sum().item()
        tot += int(L)
    print(f"左手欠損率: {inv_l / tot:.1%}  右手欠損率: {inv_r / tot:.1%}")
    min_dev_loss=float("inf")
    max_dev_metrics=1000.0 if only_gloss else 0.0
    #overfit_test(dl_train,device,normalize_batch,bones,gloss_vocab_size, trans_vocab_size, tokenizer, gloss2class)
    if evaluation==False:
        for epoch in range(epochs):
            model.train()
            train_avg_loss=[]
            dev_avg_loss=[]
            test_avg_loss=[]

            for batch in dl_train:
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens ,gloss_tokens= batch

                padded_cod_data = padded_cod_data.float().to(device)
                txt = text_tokens['input_ids']
                B,_=txt.shape
                txt_input = txt[:,:-1].to(device)
                txt_target = txt[:,1:].to(device)
                gls_attn_mask=gloss_tokens['attention_mask']
                gls_lengths=gls_attn_mask.sum(dim=1).long().to(device)
                gls_tokens=gloss_tokens['input_ids'].to(device)
                input_length_tensor = input_length_tensor.to(device)
                all_inputs,body_inputs,left_inputs,right_inputs,norm_info=normalize_batch(padded_cod_data,False)

                #new_inputs=all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                if not use_3d:
                    new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                else:
                    new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]

                new_inputs=new_inputs.reshape(new_inputs.shape[0],new_inputs.shape[1],-1)
                # エポックごとに train の数バッチで測る
                z, _ = model.encode(new_inputs, input_length_tensor)
                probs = model.gloss_output_layer(z).softmax(-1)
                train_blank = (probs.argmax(-1) == 0).float().mean()
                max_nonblank = probs[..., 1:].max().item()  # 非blank確率の最大値
                print(f"train blank率 {train_blank:.3f} | 非blank最大確率 {max_nonblank:.3f}")
                gloss_log_probs, word_logits = model(new_inputs, input_length_tensor, txt_input)
                total, lr_, lt_ = criterion(gloss_log_probs, input_length_tensor, gls_tokens, gls_lengths,
                                            word_logits, txt_target, batch_size=B)
                if torch.isnan(total) or torch.isinf(total):
                    print("Warning: Loss is inf or nan. Skipping this batch.")
                    continue
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
                loss=total.item()
                train_avg_loss.append(loss)
            print("Epoch {}: Train loss: {:.4f}".format(epoch, np.mean(train_avg_loss)))
            model.eval()
            ref_list = []
            ref_gloss_list = []
            pred_list = []
            pred_gloss_list = []
            with torch.no_grad():
                blanks = valid = 0

                for batch in dl_dev:
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
                                                                                                    False,is_openpose=is_openpose)
                    #new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                    if not use_3d:
                        new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                    else:
                        new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                    new_inputs = new_inputs.reshape(new_inputs.shape[0], new_inputs.shape[1], -1)
                    z, _ = model.encode(new_inputs, input_length_tensor)
                    pred = model.gloss_output_layer(z).argmax(-1)
                    for b in range(len(input_length_tensor)):
                        blanks += (pred[b, :input_length_tensor[b]] == 0).sum().item()
                        valid += int(input_length_tensor[b])
                    print(f"Epoch {epoch}: dev blank率 {blanks / valid:.3f}")
                    gloss_log_probs, word_logits = model(new_inputs, input_length_tensor, txt_input)
                    total, lr_, lt_ = criterion(gloss_log_probs, input_length_tensor, gls_tokens, gls_lengths,
                                                word_logits, txt_target, batch_size=B)
                    loss = total.item()
                    # translate_word=model.translate_beam(new_inputs, input_length_tensor, beam_size=5)
                    translate_word = model.translate_greedy(new_inputs, input_length_tensor)

                    #gloss_word=model.recognize_beam(new_inputs, input_length_tensor, beam_size=5,prune_topk=len(class2gloss))
                    gloss_word = model.recognize_greedy(new_inputs, input_length_tensor)

                    for b in range(len(input_length_tensor)):
                        pred_sentences=" ".join([decoder[token] for token in translate_word[b] if token not in [0, 1, 2]])
                        ref_sentences=" ".join([decoder[token] for token in txt[b].tolist() if token not in [0, 1, 2]])
                        pred_glosses=" ".join([class2gloss[token] for token in gloss_word[b] if token not in [0]])
                        ref_glosses=" ".join([class2gloss[token] for token in gls_tokens[b].tolist() if token not in [0]])
                        pred_glosses=clean_phoenix_2014_trans(pred_glosses)
                        ref_glosses=clean_phoenix_2014_trans(ref_glosses)
                        pred_list.append(pred_sentences)
                        ref_list.append(ref_sentences)
                        pred_gloss_list.append(pred_glosses)
                        ref_gloss_list.append(ref_glosses)


                    dev_avg_loss.append(loss)
                bleu_score = bleu(pred_list, ref_list)
                rouge_score = rouge(pred_list, ref_list)
                wer_score = wer(pred_gloss_list, ref_gloss_list)
                metrics_score=wer_score if only_gloss else bleu_score['bleu1']
                print(f"Epoch {epoch}: Dev loss: {np.mean(dev_avg_loss):.4f}, BLEU: {bleu_score}, ROUGE: {rouge_score:.4f}, WER: {wer_score:.4f}")
                print("Metrics score for model selection: {:.4f}".format(metrics_score))
                ref_list = []
                ref_gloss_list = []
                pred_list = []
                pred_gloss_list = []
                for batch in dl_test:
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
                                                                                                    False,is_openpose=is_openpose)
                    #new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]

                    if not use_3d:
                        new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                    else:
                        new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                    new_inputs = new_inputs.reshape(new_inputs.shape[0], new_inputs.shape[1], -1)
                    gloss_log_probs, word_logits = model(new_inputs, input_length_tensor, txt_input)
                    total, lr_, lt_ = criterion(gloss_log_probs, input_length_tensor, gls_tokens, gls_lengths,
                                                word_logits, txt_target, batch_size=B)
                    loss = total.item()
                    test_avg_loss.append(loss)
                    # translate_word=model.translate_beam(new_inputs, input_length_tensor, beam_size=5)
                    translate_word = model.translate_greedy(new_inputs, input_length_tensor)

                    #gloss_word=model.recognize_beam(new_inputs, input_length_tensor, beam_size=5,prune_topk=len(class2gloss))
                    gloss_word = model.recognize_greedy(new_inputs, input_length_tensor)

                    for b in range(len(input_length_tensor)):
                        pred_sentences = " ".join(
                            [decoder[token] for token in translate_word[b] if token not in [0, 1, 2]])
                        ref_sentences = " ".join(
                            [decoder[token] for token in txt[b].tolist() if token not in [0, 1, 2]])
                        pred_glosses = " ".join([class2gloss[token] for token in gloss_word[b] if token not in [0]])
                        ref_glosses = " ".join(
                            [class2gloss[token] for token in gls_tokens[b].tolist() if token not in [0]])
                        pred_glosses = clean_phoenix_2014_trans(pred_glosses)
                        ref_glosses = clean_phoenix_2014_trans(ref_glosses)
                        pred_list.append(pred_sentences)
                        ref_list.append(ref_sentences)
                        pred_gloss_list.append(pred_glosses)
                        ref_gloss_list.append(ref_glosses)

                bleu_score = bleu(pred_list, ref_list)
                rouge_score = rouge(pred_list, ref_list)
                wer_score = wer(pred_gloss_list, ref_gloss_list)
                print(
                    f"Epoch {epoch}: Test loss: {np.mean(test_avg_loss):.4f}, BLEU1: {bleu_score['bleu1']:.4f},BLEU2: {bleu_score['bleu2']:.4f},BLEU3: {bleu_score['bleu3']:.4f},BLEU4: {bleu_score['bleu4']:.4f}, ROUGE: {rouge_score:.4f}, WER: {wer_score:.4f}")
                prev_lr=scheduler.optimizer.param_groups[0]['lr']
                scheduler.step(np.mean(dev_avg_loss))
                now_lr=scheduler.optimizer.param_groups[0]['lr']
            for param_group in optimizer.param_groups:
                current_lr=param_group['lr']
            if only_gloss:
                if metrics_score < max_dev_metrics:
                    max_dev_metrics=metrics_score
                    print("Best model improved. Current best metrics: {:.4f}".format(max_dev_metrics))

                    print("New best model found. Saving model...")
                    torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
                    last_best_lr=current_lr
            else:
                if metrics_score>max_dev_metrics:
                    max_dev_metrics = metrics_score
                    print("Best model improved. Current best WER: {:.4f}".format(max_dev_metrics))
                    print("New best model found. Saving model...")
                    torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
                    last_best_lr = current_lr
            #if prev_lr!=now_lr:
            #    print(f"Learning rate changed from {prev_lr} to {now_lr}.")
            #    if last_best_lr!=current_lr:
            #        break
            if current_lr<1e-7:
                break
    else:
        if os.path.exists(f"{vae_weights}/config_vae.yaml"):
            with open(f"{vae_weights}/config_vae.yaml", "r") as f:
                vae_config = yaml.safe_load(f)
        vae_config=vae_config['vae']['model']
        vae = HandTransformerVAE(in_channels=6, bones=ALL_BONES, n_stages=vae_config['n_stages'],
                                 blocks_per_stage=vae_config['blocks_per_stage'],
                                 dropout=vae_config['dropout'], d_model=vae_config['d_model'],
                                 latent_dim=vae_config['latent_dim'], is_temporal=vae_config['is_temporal'])
        vae.load_state_dict(torch.load(f"{vae_weights}/best_model_all.pth", map_location=device))
        vae=vae.to(device)
        model.load_state_dict(torch.load(os.path.join(save_path, "best_model.pth")))
        model.eval()
        ref_list=[]
        pred_list=[]
        pred_gloss_list=[]
        ref_gloss_list=[]
        #diagnose(model, dl_test, bones,device=device, n_show=5, blank_idx=gloss2class['blank'])
        if eval_data!=None:
            dev_eval_data=eval_data['dev']
            test_eval_data=eval_data['test']
            test_cod_eval_root={1:eval_data['test']}
            dev_cod_eval_root={1:eval_data['dev']}

            dl_dev = torch.utils.data.DataLoader(ds_dev, batch_size=32, shuffle=False, num_workers=4,
                                                 collate_fn=ds_dev.collate_fn)
            ds_dev_eval=SLTDatasets(dev_data_path, dev_cod_eval_root, dev_face_root, trainable=False, is_3d=True,
                                   is_processed=False, is_sg_filter=False,is_generated=True,
                                   tokenizer=tokenizer,texts_corpus=dev_corpus,
                                   scale_ratio=(1.0,1.0),gloss2class=gloss2class,is_delete_cod=False,is_norm=False,is_openpose=True)
            dl_dev_eval=torch.utils.data.DataLoader(ds_dev_eval, batch_size=32, shuffle=False, num_workers=4, collate_fn=ds_dev.collate_fn)
            ds_test_eval= SLTDatasets(test_data_path, test_cod_eval_root, test_face_root, trainable=False, is_3d=True,
                                    is_processed=False, is_sg_filter=False,is_generated=True,
                                    tokenizer=tokenizer,texts_corpus=test_corpus,
                                    scale_ratio=(1.0,1.0),gloss2class=gloss2class,is_delete_cod=False,is_norm=False,is_openpose=True)
            dl_test_eval=torch.utils.data.DataLoader(ds_test_eval, batch_size=32, shuffle=False, num_workers=4, collate_fn=ds_test_eval.collate_fn)
            dev_save_path=f"{os.path.dirname(dev_eval_data)}"
            test_save_path=f"{os.path.dirname(test_eval_data)}"

        else:
            dev_save_path=test_save_path=save_path
            dl_test_eval=None
            dl_dev_eval=None
        gt_dict_dev = {}
        gt_dict_test={}
        gen_dict_dev = {}
        gen_dict_test={}
        if use_3d:
            sfix = "_3d"
        else:
            sfix = "_6d"
        if only_gloss:
            sfix += "_onlygloss"
        with torch.no_grad():
            for batch in dl_dev:
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
                all_inputs,body_inputs,left_inputs,right_inputs,norm_info=normalize_batch(padded_cod_data,False,is_openpose=is_openpose)

                #new_inputs=all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                if not use_3d:
                    new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                else:
                    new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]

                new_inputs = new_inputs.reshape(new_inputs.shape[0], new_inputs.shape[1], -1)
                #new_inputs += torch.randn_like(new_inputs)
                gloss_log_probs, word_logits = model(new_inputs, input_length_tensor, txt_input)


                translate_word=model.translate_beam(new_inputs, input_length_tensor, beam_size=5)
                #translate_word=model.translate_greedy(new_inputs, input_length_tensor)

                gloss_word=model.recognize_beam(new_inputs, input_length_tensor, beam_size=5,prune_topk=len(class2gloss))
                #gloss_word=model.recognize_greedy(new_inputs, input_length_tensor)
                for b in range(len(input_length_tensor)):
                    gt_dict_dev[data_path[b]] = new_inputs[b]

                    pred_sentences = " ".join(
                        [decoder[token] for token in translate_word[b] if token not in [0, 1, 2]])
                    ref_sentences = " ".join(
                        [decoder[token] for token in txt[b].tolist() if token not in [0, 1, 2]])
                    pred_glosses = " ".join([class2gloss[token] for token in gloss_word[b] if token not in [0]])
                    ref_glosses = " ".join(
                        [class2gloss[token] for token in gls_tokens[b].tolist() if token not in [0]])
                    pred_glosses = clean_phoenix_2014_trans(pred_glosses)
                    ref_glosses = clean_phoenix_2014_trans(ref_glosses)
                    pred_list.append(pred_sentences)
                    ref_list.append(ref_sentences)
                    pred_gloss_list.append(pred_glosses)
                    ref_gloss_list.append(ref_glosses)
                    print("Predicted sentence:", pred_sentences)
                    print("Reference sentence:", ref_sentences)
                    print("Predicted gloss:", pred_glosses)
                    print("Reference gloss:", ref_glosses)

        bleu_score=bleu(pred_list,ref_list)
        rouge_score=rouge(pred_list,ref_list)
        chrf_score = chrf(pred_list, ref_list)
        wer_score=wer(pred_gloss_list,ref_gloss_list)
        print("BLEU score:", bleu_score)
        print("ROUGE score:", rouge_score)
        print("CHRF score:", chrf_score)
        print("WER score:", wer_score)
        with open(os.path.join(dev_save_path, f"evaluation_dev_results_gt{sfix}.txt"), "w") as f:
            f.write(f"\nBLEU score: {bleu_score}\n")
            f.write(f"ROUGE score: {rouge_score}\n")
            f.write(f"CHRF score: {chrf_score}\n")
            f.write(f"WER score: {wer_score}\n")
        ref_list=[]
        pred_list=[]
        pred_gloss_list=[]
        ref_gloss_list=[]
        with torch.no_grad():
            for batch in dl_test:
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
                all_inputs,body_inputs,left_inputs,right_inputs,norm_info=normalize_batch(padded_cod_data,False,is_openpose=is_openpose)

                #new_inputs=all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]

                if not use_3d:
                    new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                else:
                    new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]

                new_inputs = new_inputs.reshape(new_inputs.shape[0], new_inputs.shape[1], -1)
                #new_inputs += torch.randn_like(new_inputs)
                gloss_log_probs, word_logits = model(new_inputs, input_length_tensor, txt_input)

                #translate_word=model.translate_beam(new_inputs, input_length_tensor, beam_size=5)
                translate_word=model.translate_greedy(new_inputs, input_length_tensor)
                gloss_word=model.recognize_beam(new_inputs, input_length_tensor, beam_size=5,prune_topk=len(class2gloss))
                #gloss_word=model.recognize_greedy(new_inputs, input_length_tensor)
                for b in range(len(input_length_tensor)):
                    gt_dict_test[data_path[b]] = new_inputs[b]

                    pred_sentences = " ".join(
                        [decoder[token] for token in translate_word[b] if token not in [0, 1, 2]])
                    ref_sentences = " ".join(
                        [decoder[token] for token in txt[b].tolist() if token not in [0, 1, 2]])
                    pred_glosses = " ".join([class2gloss[token] for token in gloss_word[b] if token not in [0]])
                    ref_glosses = " ".join(
                        [class2gloss[token] for token in gls_tokens[b].tolist() if token not in [0]])
                    pred_glosses = clean_phoenix_2014_trans(pred_glosses)
                    ref_glosses = clean_phoenix_2014_trans(ref_glosses)
                    pred_list.append(pred_sentences)
                    ref_list.append(ref_sentences)
                    pred_gloss_list.append(pred_glosses)
                    ref_gloss_list.append(ref_glosses)
                    print("Predicted sentence:", pred_sentences)
                    print("Reference sentence:", ref_sentences)
                    print("Predicted gloss:", pred_glosses)
                    print("Reference gloss:", ref_glosses)
        bleu_score=bleu(pred_list,ref_list)
        rouge_score=rouge(pred_list,ref_list)
        chrf_score = chrf(pred_list, ref_list)
        wer_score=wer(pred_gloss_list,ref_gloss_list)
        print("BLEU score:", bleu_score)
        print("ROUGE score:", rouge_score)
        print("CHRF score:", chrf_score)
        print("WER score:", wer_score)
        with open(os.path.join(test_save_path, f"evaluation_test_results_gt{sfix}.txt"), "w") as f:
            f.write(f"\nBLEU score: {bleu_score}\n")
            f.write(f"ROUGE score: {rouge_score}\n")
            f.write(f"CHRF score: {chrf_score}\n")
            f.write(f"WER score: {wer_score}\n")
        if eval_data!=None:
            all_dtw_dev_scores=[]
            all_dtw_test_scores=[]
            jpe_dev_list=[]
            jpe_test_list=[]
            ref_list = []
            pred_list = []
            pred_gloss_list = []
            ref_gloss_list = []
            with torch.no_grad():
                for batch in dl_dev_eval:
                    padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens, gloss_tokens = batch
                    # 生成CSVは前処理(nan補間/0埋め)を通らないため，GT側と同様に欠損(NaN)を0にする
                    padded_cod_data = torch.nan_to_num(padded_cod_data.float(), nan=0.0).to(device)
                    txt = text_tokens['input_ids']
                    B, _ = txt.shape
                    txt_input = txt[:, :-1].to(device)
                    txt_target = txt[:, 1:].to(device)
                    gls_attn_mask = gloss_tokens['attention_mask']
                    gls_lengths = gls_attn_mask.sum(dim=1).long().to(device)
                    gls_tokens = gloss_tokens['input_ids'].to(device)
                    input_length_tensor = input_length_tensor.to(device)
                    all_inputs, body_inputs, left_inputs, right_inputs, norm_info = normalize_batch(padded_cod_data,
                                                                                                    False,
                                                                                                    is_openpose=is_openpose)

                    #new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                    used_jts = bones_used_joints(bones=bones, include_aux=True)
                    o2c = {o: i for i, o in enumerate(used_jts)}
                    #mapped_bones = [(o2c[p], o2c[c], o2c[a]) for p, c, a in bones]
                    mapped_bones=bones
                    if not use_3d:

                        new_inputs = hand_joints_to_6d(all_inputs, bones=mapped_bones).float()
                    else:
                        new_inputs = all_inputs[:, :, :, bones_used_joints(bones=mapped_bones, include_aux=True)]


                    new_inputs = new_inputs.reshape(new_inputs.shape[0], new_inputs.shape[1], -1)
                    #new_inputs += torch.randn_like(new_inputs)
                    gloss_log_probs, word_logits = model(new_inputs, input_length_tensor, txt_input)

                    #translate_word=model.translate_beam(new_inputs, input_length_tensor, beam_size=5)
                    translate_word = model.translate_greedy(new_inputs, input_length_tensor)

                    gloss_word=model.recognize_beam(new_inputs, input_length_tensor, beam_size=5,prune_topk=len(class2gloss))
                    #gloss_word=model.recognize_greedy(new_inputs, input_length_tensor)
                    for b in range(len(input_length_tensor)):
                        gen_dict_dev[data_path[b]] = new_inputs[b,:input_length_tensor[b].int()]
                        ref_inputs = gt_dict_dev[data_path[b]]

                        dtw_ref = ref_inputs.unsqueeze(0).reshape(1, ref_inputs.shape[0], -1)[:, :input_length_tensor[b].int(), :]
                        dtw_hyp = new_inputs[b,:input_length_tensor[b].int()]
                        dtw_hyp=dtw_hyp.unsqueeze(0).reshape(1, dtw_hyp.shape[0], -1)
                        assert dtw_ref.shape[-1] == dtw_hyp.shape[
                            -1], f"DTW計算のために、参照と生成の次元が一致する必要があります。参照の次元: {dtw_ref.shape[-1]}, 生成の次元: {dtw_hyp.shape[-1]}"
                        dtw_dev = calculate_dtw_nocounter(dtw_ref, dtw_hyp)
                        jpe_dev=mpjpe(dtw_hyp, dtw_ref)
                        all_dtw_dev_scores.extend(dtw_dev)
                        jpe_dev_list.append(jpe_dev)
                        pred_sentences = " ".join(
                            [decoder[token] for token in translate_word[b] if token not in [0, 1, 2]])
                        ref_sentences = " ".join(
                            [decoder[token] for token in txt[b].tolist() if token not in [0, 1, 2]])
                        pred_glosses = " ".join([class2gloss[token] for token in gloss_word[b] if token not in [0]])
                        ref_glosses = " ".join(
                            [class2gloss[token] for token in gls_tokens[b].tolist() if token not in [0]])
                        pred_glosses = clean_phoenix_2014_trans(pred_glosses)
                        ref_glosses = clean_phoenix_2014_trans(ref_glosses)
                        pred_list.append(pred_sentences)
                        ref_list.append(ref_sentences)
                        pred_gloss_list.append(pred_glosses)
                        ref_gloss_list.append(ref_glosses)
                        print("Predicted sentence:", pred_sentences)
                        print("Reference sentence:", ref_sentences)
                        print("Predicted gloss:", pred_glosses)
                        print("Reference gloss:", ref_glosses)

            bleu_score=bleu(pred_list,ref_list)
            rouge_score=rouge(pred_list,ref_list)
            chrf_score=chrf(pred_list,ref_list)
            wer_score=wer(pred_gloss_list,ref_gloss_list)
            jpe_score=np.mean(jpe_dev_list)
            print("BLEU score:", bleu_score)
            print("ROUGE score:", rouge_score)
            print("CHRF score:", chrf_score)
            print("WER score:", wer_score)
            print("MPJPE score:", jpe_score)
            with open(os.path.join(dev_save_path, f"evaluation_dev_results{sfix}.txt"), "w") as f:
                """
                f.write("Predicted sentences:\n")
                for pred in pred_list:
                    f.write(pred + "\n")
                f.write("\nReference sentences:\n")
                for ref in ref_list:
                    f.write(ref + "\n")
                f.write("\nPredicted glosses:\n")
                for pred_gloss in pred_gloss_list:
                    f.write(pred_gloss + "\n")
                f.write("\nReference glosses:\n")
                for ref_gloss in ref_gloss_list:
                    f.write(ref_gloss + "\n")
                """
                f.write(f"\nBLEU score: {bleu_score}\n")
                f.write(f"ROUGE score: {rouge_score}\n")
                f.write(f"CHRF score: {chrf_score}\n")
                f.write(f"WER score: {wer_score}\n")
                f.write(f"MPJPE score: {jpe_score}\n")

            ref_list=[]
            pred_list=[]
            pred_gloss_list=[]
            ref_gloss_list=[]
            with torch.no_grad():
                for batch in dl_test_eval:
                    padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens, gloss_tokens = batch
                    # 生成CSVは前処理(nan補間/0埋め)を通らないため，GT側と同様に欠損(NaN)を0にする
                    padded_cod_data = torch.nan_to_num(padded_cod_data.float(), nan=0.0).to(device)
                    txt = text_tokens['input_ids']
                    B, _ = txt.shape
                    txt_input = txt[:, :-1].to(device)
                    txt_target = txt[:, 1:].to(device)
                    gls_attn_mask = gloss_tokens['attention_mask']
                    gls_lengths = gls_attn_mask.sum(dim=1).long().to(device)
                    gls_tokens = gloss_tokens['input_ids'].to(device)
                    input_length_tensor = input_length_tensor.to(device)
                    all_inputs, body_inputs, left_inputs, right_inputs, norm_info = normalize_batch(padded_cod_data,
                                                                                                    False,
                                                                                                    is_openpose=is_openpose)

                    #new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                    used_jts = bones_used_joints(bones=bones, include_aux=True)
                    o2c = {o: i for i, o in enumerate(used_jts)}
                    #mapped_bones = [(o2c[p], o2c[c], o2c[a]) for p, c, a in bones]
                    mapped_bones=bones
                    if not use_3d:

                        new_inputs = hand_joints_to_6d(all_inputs, bones=mapped_bones).float()
                    else:
                        new_inputs = all_inputs[:, :, :, bones_used_joints(bones=mapped_bones, include_aux=True)]

                    new_inputs = new_inputs.reshape(new_inputs.shape[0], new_inputs.shape[1], -1)
                    #new_inputs += torch.randn_like(new_inputs)
                    gloss_log_probs, word_logits = model(new_inputs, input_length_tensor, txt_input)

                    #translate_word=model.translate_beam(new_inputs, input_length_tensor, beam_size=5)
                    gloss_word=model.recognize_beam(new_inputs, input_length_tensor, beam_size=5,prune_topk=len(class2gloss))
                    translate_word = model.translate_greedy(new_inputs, input_length_tensor)

                    #gloss_word=model.recognize_greedy(new_inputs, input_length_tensor)
                    for b in range(len(input_length_tensor)):
                        gen_dict_test[data_path[b]] = new_inputs[b:,:input_length_tensor[b].int()]
                        ref_inputs = gt_dict_test[data_path[b]]
                        dtw_ref = ref_inputs.reshape(1, ref_inputs.shape[0], -1)[:, :input_length_tensor[b].int(), :]
                        dtw_hyp = new_inputs[b,:input_length_tensor[b].int()]
                        dtw_hyp=dtw_hyp.reshape(1, dtw_hyp.shape[0], -1)
                        assert dtw_ref.shape[-1] == dtw_hyp.shape[
                            -1], f"DTW計算のために、参照と生成の次元が一致する必要があります。参照の次元: {dtw_ref.shape[-1]}, 生成の次元: {dtw_hyp.shape[-1]}"
                        dtw_test = calculate_dtw_nocounter(dtw_ref, dtw_hyp)
                        jpe_test=mpjpe(dtw_hyp,dtw_ref)
                        all_dtw_test_scores.extend(dtw_test)
                        jpe_test_list.append(jpe_test)
                        pred_sentences = " ".join(
                            [decoder[token] for token in translate_word[b] if token not in [0, 1, 2]])
                        ref_sentences = " ".join(
                            [decoder[token] for token in txt[b].tolist() if token not in [0, 1, 2]])
                        pred_glosses = " ".join([class2gloss[token] for token in gloss_word[b] if token not in [0]])
                        ref_glosses = " ".join(
                            [class2gloss[token] for token in gls_tokens[b].tolist() if token not in [0]])
                        pred_glosses = clean_phoenix_2014_trans(pred_glosses)
                        ref_glosses = clean_phoenix_2014_trans(ref_glosses)
                        pred_list.append(pred_sentences)
                        ref_list.append(ref_sentences)
                        pred_gloss_list.append(pred_glosses)
                        ref_gloss_list.append(ref_glosses)
                        print("Predicted sentence:", pred_sentences)
                        print("Reference sentence:", ref_sentences)
                        print("Predicted gloss:", pred_glosses)
                        print("Reference gloss:", ref_glosses)

            bleu_score=bleu(pred_list,ref_list)
            chrf_score=chrf(pred_list,ref_list)
            rouge_score=rouge(pred_list,ref_list)
            wer_score=wer(pred_gloss_list,ref_gloss_list)
            jpe_score=np.mean(jpe_test_list)
            print("BLEU score:", bleu_score)
            print("ROUGE score:", rouge_score)
            print("CHRF score:", chrf_score)
            print("WER score:", wer_score)
            print("MPJPE score:", jpe_score)
            with open(os.path.join(test_save_path, f"evaluation_test_results{sfix}.txt"), "w") as f:
                """
                f.write("Predicted sentences:\n")
                for pred in pred_list:
                    f.write(pred + "\n")
                f.write("\nReference sentences:\n")
                for ref in ref_list:
                    f.write(ref + "\n")
                f.write("\nPredicted glosses:\n")
                for pred_gloss in pred_gloss_list:
                    f.write(pred_gloss + "\n")
                f.write("\nReference glosses:\n")
                for ref_gloss in ref_gloss_list:
                    f.write(ref_gloss + "\n")
                """
                f.write(f"\nBLEU score: {bleu_score}\n")
                f.write(f"ROUGE score: {rouge_score}\n")
                f.write(f"CHRF score: {chrf_score}\n")
                f.write(f"WER score: {wer_score}\n")
                f.write(f"MPJPE score: {jpe_score}\n")
            def sort_dict_by_path(gt_dict,gen_dict):
                gt_list=[]
                gen_list=[]

                for path in sorted(gt_dict.keys()):
                    gt_list.append(gt_dict[path])
                    gen_list.append(gen_dict[path])
                return gt_list,gen_list



            gt_dev_list,gen_dev_list=sort_dict_by_path(gt_dict_dev,gen_dict_dev)
            gt_test_list,gen_test_list=sort_dict_by_path(gt_dict_test,gen_dict_test)
            print(f"Dev DTW: {np.mean(all_dtw_dev_scores):.4f}")
            print(f"Test DTW: {np.mean(all_dtw_test_scores):.4f}")
            #参考:GT同士のDTW
            print("GT座標範囲:", gt_test_list[0].min().item(), gt_test_list[0].max().item(),
                  "平均絶対値:", gt_test_list[0].abs().mean().item())
            print("生成座標範囲:", gen_test_list[0].min().item(), gen_test_list[0].max().item(),
                  "平均絶対値:", gen_test_list[0].abs().mean().item())
            ### MPJAE
            """
            mpjae_dev=mpjae(gen_dev_list,gt_dev_list,mode="coord",rot6d_fn=rotation_6d_to_matrix,bones=bones)
            mpjae_test=mpjae(gen_test_list,gt_test_list,mode="coord",rot6d_fn=rotation_6d_to_matrix,bones=bones)
            print(f"Dev MPJAE: {mpjae_dev:.4f}")
            print(f"Test MPJAE: {mpjae_test:.4f}")
            mpjpe_dev=mpjpe(gen_dev_list,gt_dev_list)
            mpjpe_test=mpjpe(gen_test_list,gt_test_list)
            print(f"Dev MPJPE: {mpjpe_dev:.4f}")
            print(f"Test MPJPE: {mpjpe_test:.4f}")
            """
            with open(os.path.join(dev_save_path, f"evaluation_dev_results_gen{sfix}.txt"), "w") as f:
                f.write(f"\nDev DTW: {np.mean(all_dtw_dev_scores):.4f}\n")
                f.write(f"Dev MPJAE: {np.mean(jpe_dev_list):.4f}\n")
                #f.write(f"Dev MPJPE: {mpjpe_dev:.4f}\n")
            with open(os.path.join(test_save_path, f"evaluation_test_results_gen{sfix}.txt"), "w") as f:
                f.write(f"\nTest DTW: {np.mean(all_dtw_test_scores):.4f}\n")
                f.write(f"Test MPJAE: {np.mean(jpe_test_list):.4f}\n")
                #f.write(f"Test MPJPE: {mpjpe_test:.4f}\n")



if __name__=="__main__":
    config = {
        "device": "cuda:0",
        "dataset_parameters": {
            "is_processed": False
        }
    }



    dataset = "phoenixT"
    save_path = "/media/caffe/data_storage/CSLR/keyword_models/FlowMatching/results_sgnt_6d_onlygloss2"
    eval_data={"dev":"/media/caffe/data_storage/CSLR/keyword_models/FlowMatching/results_flow_cross_slerp_stride4_latent16_ffnmoe_boundary0.3/visualize_dev/csv",
                "test":"/media/caffe/data_storage/CSLR/keyword_models/FlowMatching/results_flow_cross_slerp_stride4_latent16_ffnmoe_boundary0.3/visualize_test/csv"}
    vae_weights="/media/caffe/data_storage/CSLR/keyword_models/FlowMatching/results_slerp_inside_latent8_stride4"
    #main(config, dataset, save_path, evaluation=False,is_openpose=False,use_3d=False,only_gloss=False,lr=1e-3)
    main(config, dataset, save_path, evaluation=True, is_openpose=False, eval_data=eval_data, vae_weights=vae_weights, use_3d=False,
         only_gloss=False)
    #save_path = "/media/caffe/data_storage/CSLR/keyword_models/FlowMatching/results_sgnt_3d_onlygloss"

    #main(config, dataset, save_path, evaluation=False,is_openpose=False,use_3d=True,only_gloss=False)
    #main(config, dataset, save_path, evaluation=True,is_openpose=False,eval_data=eval_data,vae_weights=vae_weights,use_3d=True,only_gloss=True)
    #save_path="/media/caffe/data_storage/CSLR/keyword_models/FlowMatching/results_sgnt_6d"
    #main(config, dataset, save_path, evaluation=False,is_openpose=False,use_3d=False,only_gloss=False)
    #main(config, dataset, save_path, evaluation=True,is_openpose=False,eval_data=eval_data,vae_weights=vae_weights,use_3d=False,only_gloss=False)
    #save_path="/media/caffe/data_storage/CSLR/keyword_models/FlowMatching/results_sgnt_3d"
    #main(config, dataset, save_path, evaluation=False,is_openpose=False,use_3d=True,only_gloss=False)
    #main(config, dataset, save_path, evaluation=True,is_openpose=False,eval_data=eval_data,vae_weights=vae_weights,use_3d=True,only_gloss=False)