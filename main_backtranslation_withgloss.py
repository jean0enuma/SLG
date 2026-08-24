import os,glob,shutil,random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
import yaml

from models.evaluation.sign_language_transformer import SignLanguageTransformer,JointLoss
from utils.metrics import *
from utils.diagnose import *
from Parameter.Parameter import *
from loader.coordinate_preprocess import *
from loader.data_loader import *
from SLG_datasets.SLTdatasets import SLTDatasets
import pandas as pd
from models.module.Hand_gcn_vae_6d import bones_used_joints, hand_joints_to_6d
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
def create_word_tokenizer(corpus,key='annotation'):
    # corpusの中のすべての単語を取得
    all_words = []
    text=corpus[key].values
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
def main(config, dataset,save_path,evaluation=False,is_openpose=False,eval_data=None,vae_weights=None):
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
        dl_test=torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=ds_test.collate_fn)
    trans_vocab_size=len(tokenizer)
    gloss_vocab_size=len(gloss2class)
    print("translation vocab size:", trans_vocab_size)
    print("gloss vocab size:", gloss_vocab_size)
    model=SignLanguageTransformer(input_size=282,
                                  gloss_vocab_size=gloss_vocab_size,
                                  text_vocab_size=trans_vocab_size,)


    model=model.to(device)
    bones=ALL_BONES if not is_openpose else ALL_BONES_OPENPOSE

    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(0.9,0.998),weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)
    #scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.2)
    criterion=JointLoss(txt_pad_idx=tokenizer['pad_token'],blank_idx=gloss2class['blank'],lambda_r=1.0,lambda_t=1.0)
    epochs=80
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
                new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()

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

                    new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
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

                    dev_avg_loss.append(loss)
                print("Epoch {}: Dev loss: {:.4f}".format(epoch, np.mean(dev_avg_loss)))
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

                    new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                    new_inputs = new_inputs.reshape(new_inputs.shape[0], new_inputs.shape[1], -1)
                    gloss_log_probs, word_logits = model(new_inputs, input_length_tensor, txt_input)
                    total, lr_, lt_ = criterion(gloss_log_probs, input_length_tensor, gls_tokens, gls_lengths,
                                                word_logits, txt_target, batch_size=B)
                    loss = total.item()
                    test_avg_loss.append(loss)
                print("Epoch {}: Test loss: {:.4f}".format(epoch, np.mean(test_avg_loss)))
            scheduler.step(np.mean(dev_avg_loss))
            if np.mean(dev_avg_loss) < min_dev_loss:
                min_dev_loss = np.mean(dev_avg_loss)
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
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

            dl_dev = torch.utils.data.DataLoader(ds_dev, batch_size=1, shuffle=False, num_workers=4,
                                                 collate_fn=ds_dev.collate_fn)
            ds_dev_eval=SLTDatasets(dev_data_path, dev_cod_eval_root, dev_face_root, trainable=False, is_3d=True,
                                   is_processed=False, is_sg_filter=False,
                                   tokenizer=tokenizer,texts_corpus=dev_corpus,
                                   scale_ratio=(1.0,1.0),gloss2class=gloss2class,is_delete_cod=False,is_norm=False,is_openpose=True)
            dl_dev_eval=torch.utils.data.DataLoader(ds_dev_eval, batch_size=1, shuffle=False, num_workers=4, collate_fn=ds_dev.collate_fn)
            ds_test_eval= SLTDatasets(test_data_path, test_cod_eval_root, test_face_root, trainable=False, is_3d=True,
                                    is_processed=False, is_sg_filter=False,
                                    tokenizer=tokenizer,texts_corpus=test_corpus,
                                    scale_ratio=(1.0,1.0),gloss2class=gloss2class,is_delete_cod=False,is_norm=False,is_openpose=True)
            dl_test_eval=torch.utils.data.DataLoader(ds_test_eval, batch_size=1, shuffle=False, num_workers=4, collate_fn=ds_test.collate_fn)
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
                all_inputs,body_inputs,left_inputs,right_inputs,norm_info=normalize_batch(padded_cod_data,False)

                #new_inputs=all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]

                new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                gt_dict_dev[data_path[0]]= new_inputs[0]
                new_inputs = new_inputs.reshape(new_inputs.shape[0], new_inputs.shape[1], -1)
                #new_inputs += torch.randn_like(new_inputs)
                gloss_log_probs, word_logits = model(new_inputs, input_length_tensor, txt_input)


                #translate_word=model.translate_beam(new_inputs, input_length_tensor, beam_size=5)
                translate_word=model.translate_greedy(new_inputs, input_length_tensor)

                gloss_word=model.recognize_beam(new_inputs, input_length_tensor, beam_size=5,prune_topk=len(class2gloss))
                #gloss_word=model.recognize_greedy(new_inputs, input_length_tensor)
                pred_sentence = " ".join([decoder[token] for token in translate_word[0] if token not in [0,1,2]])
                ref_tokens=txt[0].tolist()
                ref_sentence=" ".join([decoder[token] for token in ref_tokens if token not in [0,1,2]])

                pred_gloss=" ".join([class2gloss[token] for token in gloss_word[0] if token not in [0]])
                ref_gloss=" ".join([class2gloss[token] for token in gls_tokens[0].tolist() if token not in [0]])
                print("Predicted sentence:", pred_sentence)
                print("Reference sentence:", ref_sentence)
                print("Predicted gloss:", pred_gloss)
                print("Reference gloss:", ref_gloss)
                pred_list.append(pred_sentence)
                ref_list.append(ref_sentence)
                pred_gloss_list.append(pred_gloss)
                ref_gloss_list.append(ref_gloss)

        bleu_score=bleu(pred_list,ref_list)
        rouge_score=rouge(pred_list,ref_list)
        wer_score=wer(pred_gloss_list,ref_gloss_list)
        print("BLEU score:", bleu_score)
        print("ROUGE score:", rouge_score)
        print("WER score:", wer_score)
        with open(os.path.join(dev_save_path, "evaluation_dev_results_gt.txt"), "w") as f:
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
                all_inputs,body_inputs,left_inputs,right_inputs,norm_info=normalize_batch(padded_cod_data,False)

                #new_inputs=all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]

                new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                gt_dict_test[data_path[0]] = new_inputs[0]
                new_inputs = new_inputs.reshape(new_inputs.shape[0], new_inputs.shape[1], -1)
                #new_inputs += torch.randn_like(new_inputs)
                gloss_log_probs, word_logits = model(new_inputs, input_length_tensor, txt_input)

                #translate_word=model.translate_beam(new_inputs, input_length_tensor, beam_size=5)
                translate_word=model.translate_greedy(new_inputs, input_length_tensor)
                gloss_word=model.recognize_beam(new_inputs, input_length_tensor, beam_size=5,prune_topk=len(class2gloss))
                #gloss_word=model.recognize_greedy(new_inputs, input_length_tensor)
                pred_sentence = " ".join([decoder[token] for token in translate_word[0] if token not in [0,1,2]])
                ref_tokens=txt[0].tolist()
                ref_sentence=" ".join([decoder[token] for token in ref_tokens if token not in [0,1,2]])

                pred_gloss=" ".join([class2gloss[token] for token in gloss_word[0] if token not in [0]])
                ref_gloss=" ".join([class2gloss[token] for token in gls_tokens[0].tolist() if token not in [0]])
                #print("Predicted sentence:", pred_sentence)
                #print("Reference sentence:", ref_sentence)
                #print("Predicted gloss:", pred_gloss)
                #print("Reference gloss:", ref_gloss)
                pred_list.append(pred_sentence)
                ref_list.append(ref_sentence)
                pred_gloss_list.append(pred_gloss)
                ref_gloss_list.append(ref_gloss)

        bleu_score=bleu(pred_list,ref_list)
        rouge_score=rouge(pred_list,ref_list)
        wer_score=wer(pred_gloss_list,ref_gloss_list)
        print("BLEU score:", bleu_score)
        print("ROUGE score:", rouge_score)
        print("WER score:", wer_score)
        with open(os.path.join(test_save_path, "evaluation_test_results_gt.txt"), "w") as f:
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
            f.write(f"WER score: {wer_score}\n")
        if eval_data!=None:
            with torch.no_grad():
                for batch in dl_dev_eval:
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
                    all_inputs,body_inputs,left_inputs,right_inputs,norm_info=normalize_batch(padded_cod_data,False)

                    #new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]

                    new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                    gen_dict_dev[data_path[0]] = new_inputs[0]
                    new_inputs = new_inputs.reshape(new_inputs.shape[0], new_inputs.shape[1], -1)
                    #new_inputs += torch.randn_like(new_inputs)
                    gloss_log_probs, word_logits = model(new_inputs, input_length_tensor, txt_input)

                    #translate_word=model.translate_beam(new_inputs, input_length_tensor, beam_size=5)
                    translate_word = model.translate_greedy(new_inputs, input_length_tensor)

                    gloss_word=model.recognize_beam(new_inputs, input_length_tensor, beam_size=5,prune_topk=len(class2gloss))
                    #gloss_word=model.recognize_greedy(new_inputs, input_length_tensor)
                    pred_sentence = " ".join([decoder[token] for token in translate_word[0] if token not in [0,1,2]])
                    ref_tokens=txt[0].tolist()
                    ref_sentence=" ".join([decoder[token] for token in ref_tokens if token not in [0,1,2]])

                    pred_gloss=" ".join([class2gloss[token] for token in gloss_word[0] if token not in [0]])
                    ref_gloss=" ".join([class2gloss[token] for token in gls_tokens[0].tolist() if token not in [0]])
                    print("Predicted sentence:", pred_sentence)
                    print("Reference sentence:", ref_sentence)
                    print("Predicted gloss:", pred_gloss)
                    print("Reference gloss:", ref_gloss)
                    pred_list.append(pred_sentence)
                    ref_list.append(ref_sentence)
                    pred_gloss_list.append(pred_gloss)
                    ref_gloss_list.append(ref_gloss)

            bleu_score=bleu(pred_list,ref_list)
            rouge_score=rouge(pred_list,ref_list)
            wer_score=wer(pred_gloss_list,ref_gloss_list)
            print("BLEU score:", bleu_score)
            print("ROUGE score:", rouge_score)
            print("WER score:", wer_score)
            with open(os.path.join(dev_save_path, "evaluation_dev_results.txt"), "w") as f:
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
                f.write(f"WER score: {wer_score}\n")
            ref_list=[]
            pred_list=[]
            pred_gloss_list=[]
            ref_gloss_list=[]
            with torch.no_grad():
                for batch in dl_test_eval:
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
                    all_inputs,body_inputs,left_inputs,right_inputs,norm_info=normalize_batch(padded_cod_data,False)

                    #new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]

                    new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                    gen_dict_test[data_path[0]] = new_inputs[0]
                    new_inputs = new_inputs.reshape(new_inputs.shape[0], new_inputs.shape[1], -1)
                    #new_inputs += torch.randn_like(new_inputs)
                    gloss_log_probs, word_logits = model(new_inputs, input_length_tensor, txt_input)

                    #translate_word=model.translate_beam(new_inputs, input_length_tensor, beam_size=5)
                    gloss_word=model.recognize_beam(new_inputs, input_length_tensor, beam_size=5,prune_topk=len(class2gloss))
                    translate_word = model.translate_greedy(new_inputs, input_length_tensor)

                    #gloss_word=model.recognize_greedy(new_inputs, input_length_tensor)
                    pred_sentence = " ".join([decoder[token] for token in translate_word[0] if token not in [0,1,2]])
                    ref_tokens=txt[0].tolist()
                    ref_sentence=" ".join([decoder[token] for token in ref_tokens if token not in [0,1,2]])

                    pred_gloss=" ".join([class2gloss[token] for token in gloss_word[0] if token not in [0]])
                    ref_gloss=" ".join([class2gloss[token] for token in gls_tokens[0].tolist() if token not in [0]])
                    print("Predicted sentence:", pred_sentence)
                    print("Reference sentence:", ref_sentence)
                    print("Predicted gloss:", pred_gloss)
                    print("Reference gloss:", ref_gloss)
                    pred_list.append(pred_sentence)
                    ref_list.append(ref_sentence)
                    pred_gloss_list.append(pred_gloss)
                    ref_gloss_list.append(ref_gloss)

            bleu_score=bleu(pred_list,ref_list)
            rouge_score=rouge(pred_list,ref_list)
            wer_score=wer(pred_gloss_list,ref_gloss_list)
            print("BLEU score:", bleu_score)
            print("ROUGE score:", rouge_score)
            print("WER score:", wer_score)
            with open(os.path.join(test_save_path, "evaluation_test_results.txt"), "w") as f:
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
                f.write(f"WER score: {wer_score}\n")
            def sort_dict_by_path(gt_dict,gen_dict):
                gt_mu_list=[]
                gt_logvar_list=[]
                gen_mu_list=[]
                gen_logvar_list=[]
                for path in sorted(gt_dict.keys()):
                    gt_mu_list.append(gt_dict[path]['mu'])
                    gen_mu_list.append(gen_dict[path]['mu'])
                    gt_logvar_list.append(gt_dict[path]['logvar'])
                    gen_logvar_list.append(gen_dict[path]['logvar'])
                return gt_mu_list,gen_mu_list,gt_logvar_list,gen_logvar_list
            """
            gt_mu_list_dev,gen_mu_list_dev,gt_logvar_list_dev,gen_logvar_list_dev=sort_dict_by_path(gt_dict_dev,gen_dict_dev)
            gt_mu_list_test,gen_mu_list_test,gt_logvar_list_test,gen_logvar_list_test=sort_dict_by_path(gt_dict_test,gen_dict_test)
            fid_lower=fid_noextract(gt_mu_list_test[:200],gt_logvar_list_test[:200],gt_mu_list_test[200:400],gt_logvar_list_test[200:400])
            fid_dev=fid_noextract(gen_mu_list_dev,gen_logvar_list_dev,gt_mu_list_dev,gt_logvar_list_dev)
            fid_test=fid_noextract(gen_mu_list_test,gen_logvar_list_test,gt_mu_list_test,gt_logvar_list_test)
            print("FID score lower:", fid_lower)
            print("FID score dev:", fid_dev)
            print("FID score test:", fid_test)
            with open(os.path.join(dev_save_path, "evaluation_dev_results.txt"), "a") as f:
                f.write(f"\nFID score: {fid_dev}\n")
            with open(os.path.join(test_save_path, "evaluation_test_results.txt"), "a") as f:
                f.write(f"\nFID score: {fid_test}\n")
            """



if __name__=="__main__":
    config = {
        "device": "cuda:0",
        "dataset_parameters": {
            "is_processed": False
        }
    }
    dataset = "phoenixT"
    save_path = "/media/caffe/data_storage/CSLR/keyword_models/FlowMatching/results_sgnt_3d"
    eval_data={"dev":"/media/caffe/data_storage/CSLR/keyword_models/FlowMatching/results_flow_cross_slerp_stride4__attn_raw/visualize_dev/csv",
                "test":"/media/caffe/data_storage/CSLR/keyword_models/FlowMatching/results_flow_cross_slerp_stride4__attn_raw/visualize_test/csv"}
    vae_weights="/media/caffe/data_storage/CSLR/keyword_models/FlowMatching/results_slerp_inside_latent8_stride2"
    #main(config, dataset, save_path, evaluation=False,is_openpose=False)
    main(config, dataset, save_path, evaluation=True,is_openpose=False,eval_data=eval_data,vae_weights=vae_weights)