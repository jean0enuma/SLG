import os,glob,yaml
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from Parameter.Parameter import *
from loader.data_loader import *
from loader.coordinate_preprocess import *
from SLG_datasets.SLG_datasets_Units import SLGText2UnitsDatasets


from models.module.Hand_gcn_vae_6d import *
from loader.skeleton_video import save_skeleton_video
from models.module.STGCNHand import make_mask

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
def compute_bone_lengths(x, bones, bone_mask=None, eps=1e-6):
    """
    x:         (B, T, 3, J)
    bone_mask: (B, T, Nb) bool  有効ボーン (make_bone_maskの出力)
               Noneなら「長さ > eps」で自動判定
    return:    (B, Nb)  有効フレームのみの中央値ボーン長
    """
    pos = x.movedim(-2, -1)                                   # (B, T, J, 3)
    idx = torch.as_tensor([b[:2] for b in bones], device=x.device)
    vec = pos[..., idx[:, 1], :] - pos[..., idx[:, 0], :]
    lens = vec.norm(dim=-1)                                   # (B, T, Nb)

    valid = bone_mask if bone_mask is not None else lens > eps
    lens = lens.masked_fill(~valid, float("nan"))
    L = lens.nanmedian(dim=1).values                          # (B, Nb)

    # 全フレーム無効のボーンへのフォールバック(バッチ内中央値で補完)
    all_nan = torch.isnan(L)
    if all_nan.any():
        batch_med = L.nanmedian(dim=0).values                 # (Nb,)
        L = torch.where(all_nan, batch_med.expand_as(L), L)
        if torch.isnan(L).any():
            raise ValueError("some bones have no valid frame in entire batch")
    return L
def main(config, dataset,save_path,parts="hands",visualize=False,used_3d=False):
    print("---Loading datasets---")
    is_openpose=config['is_openpose']
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
        if is_islr==True:
            phoenixT_train_path, phoenixT_dev_path, phoenixT_test_path, phoenixT_gloss2class, phoenixT_class2gloss, phoenixT_video2gloss = islr_datasets_loader(
                "phoenixT")
            train_corpus[3] = phoenixT_video2gloss
            dev_corpus[3] = phoenixT_video2gloss
            test_corpus[3] = phoenixT_video2gloss
            #gloss2class=np.load(f"./Parameter/gloss_dict_T.npy",allow_pickle=True).item()
            train_cod_root[3] = f"{WORDS_DATADIR_T_SKELETON}/train"
            dev_cod_root[3] = f"{WORDS_DATADIR_T_SKELETON}/dev"
            test_cod_root[3] = f"{WORDS_DATADIR_T_SKELETON}/test"

            train_face_root[3] = f"{WORDS_DATADIR_T_SKELETON_FACE}/train"
            dev_face_root[3] = f"{WORDS_DATADIR_T_SKELETON_FACE}/dev"
            test_face_root[3] = f"{WORDS_DATADIR_T_SKELETON_FACE}/test"
            is_3d = True

            train_data_path += integrate_path(3, phoenixT_train_path)
            dev_data_path += integrate_path(3, phoenixT_dev_path)
            test_data_path += integrate_path(3, phoenixT_test_path)
            gloss2class=phoenixT_gloss2class
            class2gloss=phoenixT_class2gloss
            train_data_path += integrate_path(3, phoenixT_train_path)
            dev_data_path += integrate_path(3, phoenixT_dev_path)
            test_data_path += integrate_path(3, phoenixT_test_path)
            i += 1
        else:
            phoenixT_train_path, phoenixT_dev_path, phoenixT_test_path, phoenixTgn_train_corpus, phoenixT_dev_corpus, phoenixT_test_corpus= datasets_loader_T(
                "phoenixT")
            train_corpus[1] = phoenixTgn_train_corpus
            dev_corpus[1] = phoenixT_dev_corpus
            test_corpus[1] = phoenixT_test_corpus
            if is_openpose:
                train_cod_root[1] = SKELETON_TRAIN_DATADIR_T_OPENPOSE_PROCESSED
                dev_cod_root[1] = SKELETON_DEV_DATADIR_T_OPENPOSE_PROCESSED
                test_cod_root[1] = SKELETON_TEST_DATADIR_T_OPENPOSE_PROCESSED

                train_face_root[1] = FACE_TRAIN_DATADIR_T_OPENPOSE_PROCESSED
                dev_face_root[1] = FACE_DEV_DATADIR_T_OPENPOSE_PROCESSED
                test_face_root[1] = FACE_TEST_DATADIR_T_OPENPOSE_PROCESSED
            else:
                train_cod_root[1] = SKELETON_TRAIN_DATADIR_T_3D
                dev_cod_root[1] = SKELETON_DEV_DATADIR_T_3D
                test_cod_root[1] = SKELETON_TEST_DATADIR_T_3D

                train_face_root[1] = FACE_TRAIN_DATADIR_T_3D
                dev_face_root[1] = FACE_DEV_DATADIR_T_3D
                test_face_root[1] = FACE_TEST_DATADIR_T_3D
            train_data_path += integrate_path(1, phoenixT_train_path)
            dev_data_path += integrate_path(1, phoenixT_dev_path)
            test_data_path += integrate_path(1, phoenixT_test_path)
            is_3d = True
    if dataset=="CSL-Daily":
        csl_daily_train_path, csl_daily_dev_path, csl_daily_test_path, csl_daily_train_corpus, csl_daily_dev_corpus, csl_daily_test_corpus = datasets_loader_T(
            "CSL-Daily")

        train_corpus[1] = csl_daily_train_corpus
        dev_corpus[1] = csl_daily_dev_corpus
        test_corpus[1] = csl_daily_test_corpus
        if config['is_processed']:
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
        i += 1
    if dataset=="how2sign":
        how2sign_train_path, how2sign_dev_path, how2sign_test_path, how2sign_train_corpus, how2sign_dev_corpus, how2sign_test_corpus = datasets_loader_T(
            "how2sign")
        train_corpus[2] = how2sign_train_corpus
        dev_corpus[2] = how2sign_dev_corpus
        test_corpus[2] = how2sign_test_corpus
        if config['is_processed']:
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
        i += 1
    if dataset=="phoenix":
        phoenix_train_path, phoenix_dev_path, phoenix_test_path, phoenix_gloss2class, phoenix_class2gloss, phoenix_video2gloss =islr_datasets_loader(
            "phoenix")
        train_corpus[3] = phoenix_video2gloss
        dev_corpus[3] = phoenix_video2gloss
        test_corpus[3] = phoenix_video2gloss
        num_class=len(phoenix_gloss2class)
        train_cod_root[3] = f"{WORDS_DATADIR_SKELETON}/train"
        dev_cod_root[3] = f"{WORDS_DATADIR_SKELETON}/dev"
        test_cod_root[3] = f"{WORDS_DATADIR_SKELETON}/test"

        train_face_root[3] = f"{WORDS_DATADIR_SKELETON_FACE}/train"
        dev_face_root[3] = f"{WORDS_DATADIR_SKELETON_FACE}/dev"
        test_face_root[3] = f"{WORDS_DATADIR_SKELETON_FACE}/test"
        is_3d = True

        train_data_path += integrate_path(3, phoenix_train_path)
        dev_data_path += integrate_path(3, phoenix_dev_path)
        test_data_path += integrate_path(3, phoenix_test_path)
        is_islr=True
        i += 1
    if dataset=="AUTSL":
        autsl_train_path, autsl_dev_path, autsl_test_path,autsl_gloss2class, autsl_class2gloss,autsl_video2gloss  = islr_datasets_loader(
            "AUTSL")
        train_corpus[5] = autsl_video2gloss
        dev_corpus[5] =autsl_video2gloss
        test_corpus[5] = autsl_video2gloss
        num_class = len(autsl_gloss2class)
        if config['dataset_parameters']['is_processed']:
           raise NotImplementedError("Processed data for AUTSL is not available yet.")
        else:
            train_cod_root[5] = SKELETON_AUTSL_TRAIN_DATADIR_3D
            dev_cod_root[5] = SKELETON_AUTSL_DEV_DATADIR_3D
            test_cod_root[5] = SKELETON_AUTSL_TEST_DATADIR_3D

            train_face_root[5] = FACE_AUTSL_TRAIN_DATADIR_3D
            dev_face_root[5] = FACE_AUTSL_DEV_DATADIR_3D
            test_face_root[5] = FACE_AUTSL_TEST_DATADIR_3D
            is_3d=True
        train_data_path += integrate_path(5, autsl_train_path)
        dev_data_path += integrate_path(5, autsl_dev_path)
        test_data_path += integrate_path(5, autsl_test_path)
        is_islr=True
        i += 1
    print("Datasets loaded.")
    print("保存場所:", save_path)
    print("Is GPU available?:", torch.cuda.is_available())

    device = config["device"] if torch.cuda.is_available() else "cpu"
    # deviceからgpuの名前を取得して表示
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(int(device[-1]))
        print("GPU name:", gpu_name)
    print("---Loading tokenizer---")
    #tokenizer=gloss2class
    print("---Creating datasets---")
    ds_train = SLGText2UnitsDatasets(train_data_path, train_cod_root, train_face_root, is_3d=True,
                                     is_processed=False, is_sg_filter=False,
                                     is_coarse=False,
                                     trainable=True, tokenizer=None,texts_corpus=train_corpus,is_islr=is_islr,
                                     scale_ratio=(0.8,1.2),gloss2class=None,is_delete_cod=False,is_norm=False,is_openpose=is_openpose)
    ds_dev = SLGText2UnitsDatasets(dev_data_path, dev_cod_root, dev_face_root, trainable=False, is_3d=True,
                                   is_processed=False, is_sg_filter=False,
                                   is_coarse=False,tokenizer=None,texts_corpus=dev_corpus,is_islr=is_islr,
                                   scale_ratio=(0.8,1.2),gloss2class=None,is_delete_cod=False,is_norm=False,is_openpose=is_openpose)
    ds_test = SLGText2UnitsDatasets(test_data_path, test_cod_root, test_face_root, trainable=False, is_3d=True,
                                    is_processed=False, is_sg_filter=False,
                                    is_coarse=False,tokenizer=None,texts_corpus=test_corpus,is_islr=is_islr,
                                    scale_ratio=(0.8,1.2),gloss2class=None,is_delete_cod=False,is_norm=False,is_openpose=is_openpose)
    dl_train=torch.utils.data.DataLoader(ds_train, batch_size=2, shuffle=True, num_workers=4, collate_fn=ds_train.collate_fn)
    dl_dev=torch.utils.data.DataLoader(ds_dev, batch_size=2, shuffle=False, num_workers=4, collate_fn=ds_dev.collate_fn)
    dl_test=torch.utils.data.DataLoader(ds_test, batch_size=2, shuffle=False, num_workers=4, collate_fn=ds_test.collate_fn)
    print("Datasets created.")
    if parts=="hands":
        bones=HAND_BONES
    elif parts=="all":
        if is_openpose:
            bones=ALL_BONES_OPENPOSE
        else:
            bones=ALL_BONES
    else:
        if is_openpose:
            bones=BODY_BONES_OPENPOSE
        else:
            bones=BODY_BONES
    if used_3d:
        in_channels=3
    else:
        in_channels=6
    #model=HandGCNVAE(in_channels=in_channels,bones=bones,enc_strides=(1,2,2),dec_strides=(2,2))
    vae_config=config["vae"]
    model_config=vae_config['model']
    model=HandTransformerVAE(in_channels=in_channels,bones=bones,n_stages=model_config['n_stages'],blocks_per_stage=model_config['blocks_per_stage'],
                             dropout=model_config['dropout'],d_model=model_config['d_model'],latent_dim=model_config['latent_dim'])
    with open(f"{save_path}/config_vae.yaml", "w") as f:
        yaml.dump(config, f)
    model=model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs = vae_config['lr_parameters']['epoch']
    # scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = vae_loss if used_3d==False else vae_loss_cod

    if visualize!=True:
        min_loss=float('inf')
        scaler=torch.amp.GradScaler('cuda')
        beta_max, warmup = 1e-4, 5  # epochs
        for epoch in range(epochs):
            model.train()
            beta = beta_max #* min(1.0, (epoch + 1) / warmup)
            total_loss=0
            dev_total_loss=0
            test_total_loss=0
            for batch in dl_train:
                optimizer.zero_grad()
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path = batch
                padded_cod_data=padded_cod_data.float().to(device)

                all_inputs,body_inputs,left_inputs,right_inputs,norm_info=normalize_batch(padded_cod_data,used_3d,is_openpose=is_openpose)

                with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                    if parts=="hands":
                        new_inputs=torch.cat([left_inputs,right_inputs],dim=0)
                        if used_3d:
                            new_inputs=new_inputs[:,:,:,bones_used_joints(bones=bones, include_aux=True)]


                            outputs, mu, logvar = model(new_inputs)
                            left_mask = make_mask(input_length_tensor.to(device), left_inputs)#(B,T,J)
                            left_mask = left_mask[...,bones_used_joints(bones=bones, include_aux=True)]
                            right_mask = make_mask(input_length_tensor.to(device), right_inputs)
                            right_mask = right_mask[...,bones_used_joints(bones=bones, include_aux=True)]
                            new_mask = torch.cat([left_mask, right_mask], dim=0)
                        else:
                            new_inputs=hand_joints_to_6d(new_inputs,bones=bones).float()
                            outputs, mu, logvar = model(new_inputs)
                            left_mask = make_mask(input_length_tensor.to(device), left_inputs)#(B,T,J)
                            left_mask = make_bone_mask(left_mask)
                            right_mask = make_mask(input_length_tensor.to(device), right_inputs)
                            right_mask = make_bone_mask(right_mask)
                            new_mask = torch.cat([left_mask, right_mask], dim=0)
                        len_mask=torch.cat([left_mask, right_mask], dim=0)
                        target_cod=torch.cat([left_inputs,right_inputs],dim=0)
                    elif parts=="all":
                        #joint8,29を削除
                        if used_3d:
                            new_inputs=all_inputs[:,:,:,bones_used_joints(bones=bones, include_aux=True)]
                            outputs, mu, logvar = model(new_inputs)
                            all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                            new_mask = all_mask[:,:,bones_used_joints(bones=bones, include_aux=True)]
                        else:
                            new_inputs=hand_joints_to_6d(all_inputs,bones=bones).float()
                            outputs, mu, logvar = model(new_inputs)
                            all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                            new_mask = make_bone_mask(all_mask,bones=bones)
                        len_mask=all_mask
                        target_cod=all_inputs

                    else:
                        if used_3d:
                            new_inputs=body_inputs[:,:,:,bones_used_joints(bones=bones, include_aux=True)]
                            outputs, mu, logvar = model(new_inputs)
                            body_mask = make_mask(input_length_tensor.to(device), body_inputs)
                            new_mask = body_mask[:,:,bones_used_joints(bones=bones, include_aux=True)]
                        else:
                            new_inputs=hand_joints_to_6d(body_inputs,bones=bones)
                            outputs, mu, logvar = model(new_inputs)
                            body_mask = make_mask(input_length_tensor.to(device), body_inputs)
                            new_mask = make_bone_mask(body_mask,bones=bones)
                        len_mask=body_mask
                        target_cod=body_inputs
                    if used_3d:
                        loss = \
                        loss_fn(outputs, new_inputs, mu, logvar, mask=new_mask)[0]
                    else:
                        loss,logs=loss_fn(outputs,new_inputs,mu,logvar,beta=beta,free_bits=0.05,mask=new_mask,target_cod=target_cod,bones=bones,j_mask=len_mask)
                        kl_per_dim = logs['kl'] / (mu.numel() / mu.shape[0])
                        if kl_per_dim<0.01:
                            print("Warning: KL per dim is too small, stopping training.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss+=loss.item()
            #scheduler.step()
            avg_loss=total_loss/len(ds_train)
            for batch in dl_dev:
                model.eval()
                with torch.no_grad():
                    padded_cod_data, padded_mask, input_length_tensor, id_list, data_path = batch
                    padded_cod_data = padded_cod_data.float().to(device)
                    all_inputs, body_inputs, left_inputs, right_inputs, norm_info = normalize_batch(padded_cod_data,
                                                                                                    used_3d,is_openpose=is_openpose)


                    if parts == "hands":
                        new_inputs = torch.cat([left_inputs, right_inputs], dim=0)
                        if used_3d:
                            new_inputs = new_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                            outputs, mu, logvar = model(new_inputs)
                            left_mask = make_mask(input_length_tensor.to(device), left_inputs)  # (B,T,J)
                            left_mask = left_mask[..., bones_used_joints(bones=bones, include_aux=True)]
                            right_mask = make_mask(input_length_tensor.to(device), right_inputs)
                            right_mask = right_mask[..., bones_used_joints(bones=bones, include_aux=True)]
                            new_mask = torch.cat([left_mask, right_mask], dim=0)
                        else:
                            new_inputs = hand_joints_to_6d(new_inputs, bones=bones).float()
                            outputs, mu, logvar = model(new_inputs)
                            left_mask = make_mask(input_length_tensor.to(device), left_inputs)  # (B,T,J)
                            left_mask = make_bone_mask(left_mask)
                            right_mask = make_mask(input_length_tensor.to(device), right_inputs)
                            right_mask = make_bone_mask(right_mask)
                            new_mask = torch.cat([left_mask, right_mask], dim=0)
                        len_mask = torch.cat([left_mask, right_mask], dim=0)
                        target_cod = torch.cat([left_inputs, right_inputs], dim=0)
                    elif parts == "all":
                        # joint8,29を削除
                        if used_3d:
                            new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                            outputs, mu, logvar = model(new_inputs)
                            all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                            new_mask = all_mask[:, :, bones_used_joints(bones=bones, include_aux=True)]
                        else:
                            new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                            outputs, mu, logvar = model(new_inputs)
                            all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                            new_mask = make_bone_mask(all_mask, bones=bones)
                        len_mask = all_mask
                        target_cod = all_inputs

                    else:
                        if used_3d:
                            new_inputs = body_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                            outputs, mu, logvar = model(new_inputs)
                            body_mask = make_mask(input_length_tensor.to(device), body_inputs)
                            new_mask = body_mask[:, :, bones_used_joints(bones=bones, include_aux=True)]
                        else:
                            new_inputs = hand_joints_to_6d(body_inputs, bones=bones)
                            outputs, mu, logvar = model(new_inputs)
                            body_mask = make_mask(input_length_tensor.to(device), body_inputs)
                            new_mask = make_bone_mask(body_mask, bones=bones)
                        len_mask = body_mask
                        target_cod = body_inputs

                    if used_3d:
                        loss = \
                            loss_fn(outputs, new_inputs, mu, logvar, mask=new_mask)[0]
                    else:
                        loss = \
                        loss_fn(outputs, new_inputs, mu, logvar, mask=new_mask, target_cod=target_cod, bones=bones,
                                j_mask=len_mask)[0]
                    dev_total_loss+=loss.item()
            dev_avg_loss=dev_total_loss/len(ds_dev)
            for batch in dl_test:
                model.eval()
                with torch.no_grad():
                    padded_cod_data, padded_mask, input_length_tensor, id_list, data_path = batch
                    padded_cod_data = padded_cod_data.float().to(device)
                    all_inputs, body_inputs, left_inputs, right_inputs, norm_info = normalize_batch(padded_cod_data,
                                                                                                    used_3d,is_openpose=is_openpose)

                    if parts == "hands":
                        new_inputs = torch.cat([left_inputs, right_inputs], dim=0)
                        if used_3d:
                            new_inputs = new_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                            outputs, mu, logvar = model(new_inputs)
                            left_mask = make_mask(input_length_tensor.to(device), left_inputs)  # (B,T,J)
                            left_mask = left_mask[..., bones_used_joints(bones=bones, include_aux=True)]
                            right_mask = make_mask(input_length_tensor.to(device), right_inputs)
                            right_mask = right_mask[..., bones_used_joints(bones=bones, include_aux=True)]
                            new_mask = torch.cat([left_mask, right_mask], dim=0)
                        else:
                            new_inputs = hand_joints_to_6d(new_inputs, bones=bones).float()
                            outputs, mu, logvar = model(new_inputs)
                            left_mask = make_mask(input_length_tensor.to(device), left_inputs)  # (B,T,J)
                            left_mask = make_bone_mask(left_mask)
                            right_mask = make_mask(input_length_tensor.to(device), right_inputs)
                            right_mask = make_bone_mask(right_mask)
                            new_mask = torch.cat([left_mask, right_mask], dim=0)
                        len_mask = torch.cat([left_mask, right_mask], dim=0)
                        target_cod = torch.cat([left_inputs, right_inputs], dim=0)
                    elif parts == "all":
                        # joint8,29を削除
                        if used_3d:
                            new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                            outputs, mu, logvar = model(new_inputs)
                            all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                            new_mask = all_mask[:, :, bones_used_joints(bones=bones, include_aux=True)]
                        else:
                            new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                            outputs, mu, logvar = model(new_inputs)
                            all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                            new_mask = make_bone_mask(all_mask, bones=bones)
                        len_mask = all_mask
                        target_cod = all_inputs

                    else:
                        if used_3d:
                            new_inputs = body_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                            outputs, mu, logvar = model(new_inputs)
                            body_mask = make_mask(input_length_tensor.to(device), body_inputs)
                            new_mask = body_mask[:, :, bones_used_joints(bones=bones, include_aux=True)]
                        else:
                            new_inputs = hand_joints_to_6d(body_inputs, bones=bones)
                            outputs, mu, logvar = model(new_inputs)
                            body_mask = make_mask(input_length_tensor.to(device), body_inputs)
                            new_mask = make_bone_mask(body_mask, bones=bones)
                        len_mask = body_mask
                        target_cod = body_inputs

                    if used_3d:
                        loss = \
                            loss_fn(outputs, new_inputs, mu, logvar, mask=new_mask)[0]
                    else:
                        loss = \
                        loss_fn(outputs, new_inputs, mu, logvar, mask=new_mask, target_cod=target_cod, bones=bones,
                                j_mask=len_mask)[0]
                    test_total_loss+=loss.item()
            test_avg_loss=test_total_loss/len(ds_test)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}"
                  f", Dev Loss: {dev_avg_loss:.4f}, Test Loss: {test_avg_loss:.4f}")
            if dev_avg_loss<min_loss:
                min_loss=dev_avg_loss
                torch.save(model.state_dict(), os.path.join(save_path, f"best_model_{parts}.pth"))
                print(f"Best model saved at epoch {epoch+1} with loss {min_loss:.4f}")
    else:
        model.load_state_dict(torch.load(f"{save_path}/best_model_{parts}.pth", map_location=device))
        os.makedirs(f"{save_path}/visualize/{parts}", exist_ok=True)
        for batch in dl_test:
            model.eval()
            with torch.no_grad():
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path = batch
                padded_cod_data = padded_cod_data.float().to(device)
                all_inputs, body_inputs, left_inputs, right_inputs, norm_info = normalize_batch(padded_cod_data,
                                                                                                used_3d,is_openpose=is_openpose)
                if norm_info!=None:
                    center_data, shoulder_length, left_center_data, right_center_data = norm_info["center"], norm_info["shoulder"], norm_info["left_center"], norm_info["right_center"]
                else:
                    center_data, shoulder_length, left_center_data, right_center_data = None, None, None, None

                if parts == "hands":
                    new_inputs = torch.cat([left_inputs, right_inputs], dim=0)
                    if used_3d:
                        new_inputs = new_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                        outputs, mu, logvar = model(new_inputs)
                        left_pred_coordinates = outputs[:left_inputs.shape[0]]
                        right_pred_coordinates = outputs[left_inputs.shape[0]:]
                        left_pred_coordinates*=shoulder_length/2
                        left_pred_coordinates+=left_center_data.unsqueeze(3)
                        right_pred_coordinates*=shoulder_length/2
                        right_pred_coordinates+=right_center_data.unsqueeze(3)
                        left_inputs*=shoulder_length/2
                        left_inputs+=center_data.unsqueeze(3)
                        right_inputs*=shoulder_length/2
                        right_inputs+=center_data.unsqueeze(3)
                    else:
                        new_inputs = hand_joints_to_6d(new_inputs, bones=HAND_BONES).float()
                        outputs, mu, logvar = model(new_inputs)
                        left_mask = make_mask(input_length_tensor.to(device), left_inputs)  # (B,T,J)
                        left_mask = make_bone_mask(left_mask).cpu()
                        right_mask = make_mask(input_length_tensor.to(device), right_inputs)
                        right_mask = make_bone_mask(right_mask).cpu()
                        left_inputs = left_inputs.detach().cpu()
                        right_inputs = right_inputs.detach().cpu()
                        outputs = outputs.detach().cpu()

                        left_pred_coordinates = reconstruct_joints_from_6d(left_inputs, outputs[:left_inputs.shape[0]],
                                                                           bones=bones, bone_mask=left_mask)
                        right_pred_coordinates = reconstruct_joints_from_6d(right_inputs, outputs[left_inputs.shape[0]:],
                                                                            bones=bones, bone_mask=right_mask)

                    # pred_coordinatesとnew_inputsをプロットして動画に保存する
                    for i in range(left_pred_coordinates.shape[0]):
                        pred_video_path = f"{save_path}/visualize/{parts}/{os.path.basename(data_path[i])}_left.mp4"
                        save_skeleton_video(left_pred_coordinates[i][:input_length_tensor[i]].cpu().numpy(),
                                            pred_video_path, bones=bones,
                                            x_ref=left_inputs[i][:input_length_tensor[i]].cpu().numpy())
                    for i in range(right_pred_coordinates.shape[0]):
                        pred_video_path = f"{save_path}/visualize/{parts}/{os.path.basename(data_path[i])}_right.mp4"
                        save_skeleton_video(right_pred_coordinates[i][:input_length_tensor[i]].cpu().numpy(),
                                            pred_video_path, bones=bones,
                                            x_ref=right_inputs[i][:input_length_tensor[i]].cpu().numpy())


                elif parts == "all":
                    if used_3d:
                        new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                        outputs, mu, logvar = model(new_inputs)
                        pred_coordinates = outputs
                        pred_coordinates=pred_coordinates*shoulder_length/2
                        pred_coordinates+=center_data.unsqueeze(3)
                        all_inputs*=shoulder_length
                        all_inputs+=center_data.unsqueeze(3)
                    else:
                        new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                        outputs, mu, logvar = model(new_inputs)
                        all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                        new_mask = make_bone_mask(all_mask, bones=bones).cpu()
                        #print("出力の時間分散:", outputs.std(dim=1).mean().item())
                        #print("入力の時間分散:", new_inputs.std(dim=1).mean().item())
                        all_inputs = all_inputs.detach().cpu()
                        outputs = outputs.detach().cpu()
                        pred_coordinates = reconstruct_joints_from_6d(all_inputs, outputs, bones=bones, bone_mask=new_mask)
                    # pred_coordinatesとnew_inputsをプロットして動画に保存する
                    for i in range(pred_coordinates.shape[0]):
                        pred_video_path = f"{save_path}/visualize/{parts}/{os.path.basename(data_path[i])}.mp4"
                        t = 10  # 有効そうなフレーム
                        f = all_inputs[i][t].numpy()  # (3, J)
                        canvas = np.full((640, 640, 3), 255, np.uint8)
                        for j in range(f.shape[1]):
                            px, py = int(f[0, j] * 640), int(f[1, j] * 640)
                            cv2.circle(canvas, (px, py), 4, (0, 0, 255), -1)
                            cv2.putText(canvas, str(j), (px + 5, py - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        cv2.imwrite("joints_labeled.png", canvas)
                        save_skeleton_video(pred_coordinates[i][:input_length_tensor[i]].cpu().numpy(), pred_video_path,
                                            bones=bones,
                                            x_ref=all_inputs[i][:input_length_tensor[i]].cpu().numpy())
                else:
                    if used_3d:
                        new_inputs = body_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                        outputs, mu, logvar = model(new_inputs)
                        pred_coordinates = outputs
                        pred_coordinates=pred_coordinates*shoulder_length/2
                        pred_coordinates+=center_data.unsqueeze(3)
                        x_rt = new_inputs
                        x_rt=x_rt*shoulder_length
                        x_rt+=center_data.unsqueeze(3)
                    else:
                        new_inputs = hand_joints_to_6d(body_inputs, bones=bones)
                        outputs, mu, logvar = model(new_inputs)
                        body_inputs = body_inputs.detach().cpu()
                        outputs = outputs.detach().cpu()
                        body_mask = make_mask(input_length_tensor, body_inputs)
                        new_mask = make_bone_mask(body_mask, bones=bones)
                        new_inputs = new_inputs.detach().cpu()
                        x_rt = reconstruct_joints_from_6d(body_inputs, new_inputs, bones=bones, bone_mask=new_mask)
                        pred_coordinates = reconstruct_joints_from_6d(body_inputs, outputs, bones=bones, bone_mask=new_mask)
                    # pred_coordinatesとnew_inputsをプロットして動画に保存する
                    for i in range(pred_coordinates.shape[0]):
                        pred_video_path = f"{save_path}/visualize/{parts}/{os.path.basename(data_path[i])}.mp4"
                        gt_video_path = f"{save_path}/visualize/{parts}/{os.path.basename(data_path[i])}_gt.mp4"
                        # save_skeleton_video(x_rt[i].cpu().numpy(), gt_video_path, bones=bones)
                        save_skeleton_video(pred_coordinates[i][:input_length_tensor[i]].cpu().numpy(), pred_video_path,
                                            bones=BODY_BONES2 if used_3d else bones,
                                            x_ref=x_rt[i][:input_length_tensor[i]].cpu().numpy())


if __name__ == "__main__":
    with open(f"/home/caffe/work/SLG/Parameter/config_flowmatch.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset = "phoenixT"  # or "CSL-Daily", "how2sign", "phoenix", "AUTSL"
    save_path = "/media/caffe/data_storage/CSLR/keyword_models/FlowMatching/results_openpose"
    #main(config, dataset, save_path, parts="body",visualize=False,used_3d=False)
    #main(config, dataset, save_path, parts="body",visualize=True,used_3d=False)

    #main(config, dataset, save_path, parts="hands",visualize=False,used_3d=False)
    #main(config, dataset, save_path, parts="hands",visualize=True,used_3d=False)

    main(config, dataset, save_path, parts="all",visualize=False,used_3d=False)
    main(config, dataset, save_path,parts="all",visualize=True,used_3d=False)

    #save_path = "./results_3d"
    #main(config, dataset, save_path, parts="body",visualize=False,used_3d=True)
    #main(config, dataset, save_path, parts="body",visualize=True,used_3d=True)

    #main(config, dataset, save_path, parts="hands", visualize=False, used_3d=True)
    #main(config, dataset, save_path, parts="hands", visualize=True, used_3d=True)

    #main(config, dataset, save_path, parts="all", visualize=False, used_3d=True)
    #main(config, dataset, save_path, parts="all", visualize=True, used_3d=True)



