import os,glob

import torch.cuda

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from Parameter.Parameter import *
from loader.data_loader import *
from loader.coordinate_preprocess import *
from SLG_datasets.SLG_datasets_Units import SLGText2UnitsDatasets
from models.module.Hand_gcn_vae_6d import *
from models.ema import EMA
from loader.skeleton_video import save_skeleton_video
from models.module.STGCNHand import make_mask
from models.module.transformer6d_vae import HandTransformerVAE
from models. module.Latent_flow_moe import SignLatentFlowMoE
from transformers import AutoTokenizer,AutoModel
import shutil
import json,yaml
import wandb
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
@torch.no_grad()
def sample_with_diagnostics(model, n, T, steps=50, cond=None, cond_mask=None,
                            guidance_scale=1.0, lengths=None, device=None):
    """sampleと同じ軌道を辿りつつ、各stepの統計を記録して返す."""
    device = device or next(model.parameters()).device
    use_cfg = (cond is not None and guidance_scale != 1.0)
    if lengths is not None:
        lengths = torch.as_tensor(lengths, device=device)
        T = int(lengths.max().item())
    T_lat = max(1, T // model.vae.t_stride)
    valid = model._length_mask(lengths, T_lat, device)
    time_pad = ~valid if valid is not None else None

    z = torch.randn(n, T_lat, model.num_tokens, model.latent_dim, device=device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    log = {"t": [], "z_norm": [], "v_norm": [], "dz_norm": [], "is_high": []}
    for i in range(steps):
        t_now = ts[i].expand(n)
        if use_cfg:
            v_c = model.velocity(z, t_now, context=cond, context_mask=cond_mask,
                                 time_pad=time_pad)
            v_u = model.velocity(z, t_now, context=None, time_pad=time_pad)
            v = v_u + guidance_scale * (v_c - v_u)
        else:
            v = model.velocity(z, t_now, context=cond, context_mask=cond_mask,
                               time_pad=time_pad)
        dz = (ts[i] - ts[i + 1]) * v
        # 有効フレームのみで統計を取る (パディングはゼロで薄まるため)
        m = valid[:, :, None, None].float() if valid is not None else \
            torch.ones_like(z[..., :1])
        nel = m.sum() * z.shape[2] * z.shape[3] / m.numel() * m.numel()
        def rms(x):  # マスク付きRMSノルム
            return ((x * x * m).sum() / (m.sum() * x.shape[2] * x.shape[3])).sqrt().item()
        log["t"].append(ts[i].item())
        log["z_norm"].append(rms(z))
        log["v_norm"].append(rms(v))
        log["dz_norm"].append(rms(dz))
        log["is_high"].append(bool(model.router(t_now)[0]))
        z = z - dz
    return z, log
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
def visualize_skeletons(predicted_d6,gt_cods,data_path, lengths, save_dir, bones, mask=None,):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    pred_coordinates = reconstruct_joints_from_6d(gt_cods,predicted_d6, bones=bones, bone_mask=mask,is_L_median=True)
    pred_coordinates,_=mask_zero_6d_joints(pred_coordinates,predicted_d6,bones)
    #pred_coordinates=pred_coordinates[:,:,:,bones_used_joints(bones)]
    #gt_cods=gt_cods[:,:,:,bones_used_joints(bones)]
    for i in range(pred_coordinates.shape[0]):
        pred_video_path = f"{save_dir}/{os.path.basename(data_path[i])}.mp4"
        save_skeleton_video(pred_coordinates[i][:lengths[i]].cpu().numpy(), pred_video_path,
                            bones=bones,
                            x_ref=gt_cods[i][:lengths[i]].cpu().numpy())


def main(config, dataset,save_path,parts="hands",visualize=False,used_3d=False,vae_weights=None,weights=None):
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
    is_gloss=config['is_gloss']
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
                "phoenixT",gloss=is_gloss)
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
        if config['is_processed']:
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
    flow_config=config['flow_matching']

    device = config["device"] if torch.cuda.is_available() else "cpu"
    # deviceからgpuの名前を取得して表示
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(int(device[-1]))
        print("GPU name:", gpu_name)
    print("---Loading tokenizer---")
    tokenizer= AutoTokenizer.from_pretrained("google-bert/bert-base-german-dbmdz-uncased")
    print("---Creating datasets---")
    ds_train = SLGText2UnitsDatasets(train_data_path, train_cod_root, train_face_root, is_3d=True,
                                     is_processed=False, is_sg_filter=False,
                                     is_coarse=False,
                                     trainable=True, tokenizer=tokenizer,texts_corpus=train_corpus,is_islr=is_islr,
                                     scale_ratio=(1.0,1.0),gloss2class=None,is_delete_cod=False,is_norm=False)
    ds_dev = SLGText2UnitsDatasets(dev_data_path, dev_cod_root, dev_face_root, trainable=False, is_3d=True,
                                   is_processed=False, is_sg_filter=False,
                                   is_coarse=False,tokenizer=tokenizer,texts_corpus=dev_corpus,is_islr=is_islr,
                                   scale_ratio=(1.0,1.0),gloss2class=None,is_delete_cod=False,is_norm=False)
    ds_test = SLGText2UnitsDatasets(test_data_path, test_cod_root, test_face_root, trainable=False, is_3d=True,
                                    is_processed=False, is_sg_filter=False,
                                    is_coarse=False,tokenizer=tokenizer,texts_corpus=test_corpus,is_islr=is_islr,
                                    scale_ratio=(1.0,1.0),gloss2class=None,is_delete_cod=False,is_norm=False)
    dl_train=torch.utils.data.DataLoader(ds_train, batch_size=flow_config['lr_parameters']['batch_size'], shuffle=True, num_workers=4, collate_fn=ds_train.collate_fn,drop_last=True)
    dl_dev=torch.utils.data.DataLoader(ds_dev, batch_size=flow_config['lr_parameters']['batch_size'], shuffle=True, num_workers=4, collate_fn=ds_dev.collate_fn)
    dl_test=torch.utils.data.DataLoader(ds_test, batch_size=flow_config['lr_parameters']['batch_size'], shuffle=False, num_workers=4, collate_fn=ds_test.collate_fn)
    print("Datasets created.")
    bones=ALL_BONES
    hand_bones=HAND_BONES
    body_bnones=BODY_BONES

    #model=HandGCNVAE(in_channels=in_channels,bones=bones,enc_strides=(1,2,2),dec_strides=(2,2))
    vae_config=config['vae']['model']
    vae=HandTransformerVAE(in_channels=6,bones=bones,n_stages=vae_config['n_stages'],blocks_per_stage=vae_config['blocks_per_stage'],
                            dropout=vae_config['dropout'],d_model=vae_config['d_model'],latent_dim=vae_config['latent_dim'],is_temporal=vae_config['is_temporal'])
    hand_vae=HandTransformerVAE(in_channels=6,bones=hand_bones,n_stages=vae_config['n_stages'],blocks_per_stage=vae_config['blocks_per_stage'],
                            dropout=vae_config['dropout'],d_model=vae_config['d_model'],latent_dim=vae_config['latent_dim'],is_temporal=vae_config['is_temporal'])
    body_vae=HandTransformerVAE(in_channels=6,bones=body_bnones,n_stages=vae_config['n_stages'],blocks_per_stage=vae_config['blocks_per_stage'],
                            dropout=vae_config['dropout'],d_model=vae_config['d_model'],latent_dim=vae_config['latent_dim'],is_temporal=vae_config['is_temporal'])
    model_config=config['flow_matching']['model']
    model=SignLatentFlowMoE(vae,hand_vae=None,body_vae=None,
                            boundary=model_config['boundary'],
                              d_model=model_config['d_model'], n_heads=model_config['n_heads'],
                              depth_high=model_config['depth_high'], depth_low=model_config['depth_low'],
                              cond_dim=model_config['cond_dim'],cross_attn=model_config['cross_attn'],
                            is_rope=model_config['is_rope'],sliding_attn=model_config['sliding_attn'],
                            window=model_config['window'],local_heads=model_config['local_heads'],
                            part_heads=model_config['part_heads'] ,
                            tau=model_config['tau'],alpha_max=model_config['alpha_max']) # テキスト埋め込み次元 (例: T5系)

    with open(f"{save_path}/flow_match.yaml", "w") as f:
        yaml.dump(config, f)

    text_encoder=AutoModel.from_pretrained("google-bert/bert-base-german-dbmdz-uncased")
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    model=model.to(device)
    if vae_weights is not None:
        all_vae_weights=f"{vae_weights}/best_model_all.pth"
        hand_vae_weights=f"{vae_weights}/best_model_hands.pth"
        body_vae_weights=f"{vae_weights}/best_model_body.pth"
        model.vae.load_state_dict(torch.load(all_vae_weights, map_location=device))
        if hasattr(model, 'hand_vae'):
            model.hand_vae.load_state_dict(torch.load(hand_vae_weights, map_location=device))
        if hasattr(model, 'body_vae'):
            model.body_vae.load_state_dict(torch.load(body_vae_weights, map_location=device))
    text_encoder=text_encoder.to(device)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (p.ndim < 2 or "token_emb" in n or "token_bias" in n
                or "null_cond" in n or ".mod" in n or "moe_ffn" in n
            or "part_bias" in n):
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=2e-4, betas=(0.9, 0.95))
    epochs = flow_config['lr_parameters']['epoch']
    warmup, total = epochs * len(dl_train)*0.1//(flow_config['lr_parameters']['batch_size']*flow_config['lr_parameters']['accumulation_steps']), epochs * len(dl_train)//(flow_config['lr_parameters']['batch_size']*flow_config['lr_parameters']['accumulation_steps'])

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    #ema_model=torch.optim.swa_utils.AveragedModel(model,
    #                                              torch.optim.swa_utils.get_ema_multi_avg_fn(0.9), use_buffers=True)
    if visualize!=True:
        min_loss=float('inf')
        scaler=torch.amp.GradScaler('cuda')
        beta_max, warmup = 1e-4, 5  # epochs
        # --- 学習開始前 (model.to(device) の後, ループの前) ---
        if weights==None and vae_weights!=None:
            model.eval()
            # float64 で総和・二乗和を蓄積し、全データの std を正確に求める
            total_sum = torch.zeros((), device=device, dtype=torch.float64)
            total_sq_sum = torch.zeros((), device=device, dtype=torch.float64)
            total_sum_hands= torch.zeros((), device=device, dtype=torch.float64)
            total_sum_body= torch.zeros((), device=device, dtype=torch.float64)
            total_sq_sum_hands= torch.zeros((), device=device, dtype=torch.float64)
            total_sq_sum_body= torch.zeros((), device=device, dtype=torch.float64)
            total_count = 0
            total_count_hands=0
            total_count_body=0
            with torch.no_grad():
                for batch in dl_train:
                    padded_cod_data, _, input_length_tensor, _, _, _ = batch
                    padded_cod_data = padded_cod_data.float().to(device)
                    all_inputs, *_ = normalize_batch(padded_cod_data, used_3d)
                    new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                    input_length_tensor = input_length_tensor.to(device)
                    # VAE の潜在平均 mu: (B, T_lat, V, latent_dim)
                    mu, _ = model.vae.encode(new_inputs, input_length_tensor)
                    if hasattr(model, 'hand_vae') and hasattr(model, 'body_vae'):
                        mu_l,_=model.hand_vae.encode(new_inputs[:,:,:,-40:-20],input_length_tensor)
                        mu_r,_=model.hand_vae.encode(new_inputs[:,:,:,-20:],input_length_tensor)
                        mu_b,_=model.body_vae.encode(new_inputs[:,:,:,:-40],input_length_tensor)
                        mu_hands=torch.cat([mu_l,mu_r],dim=0)

                    # 元コードと同じく、パディング領域は統計計算から除外
                    valid = model._length_mask(
                        input_length_tensor,
                        T_lat=mu.shape[1],
                        device=mu.device,
                    )

                    if valid is not None:
                        m = mu[valid]  # (有効時刻数, V, latent_dim)
                        if hasattr(model, 'hand_vae') and hasattr(model, 'body_vae'):
                            valid_hand=torch.cat([valid,valid],dim=0)
                            m_hands=mu_hands[valid_hand]
                            m_body=mu_b[valid]
                            m_hands=m_hands.to(torch.float64)
                            m_body=m_body.to(torch.float64)
                    else:
                        m = mu

                    # 念のため float64 にしてから蓄積
                    m = m.to(torch.float64)

                    total_sum += m.sum()
                    total_sq_sum += (m * m).sum()
                    total_count += m.numel()
                    if hasattr(model, 'hand_vae') and hasattr(model, 'body_vae'):
                        total_sum_hands+=m_hands.sum()
                        total_sq_sum_hands+=(m_hands*m_hands).sum()
                        total_count_hands+=m_hands.numel()
                        total_sum_body+=m_body.sum()
                        total_sq_sum_body+=(m_body*m_body).sum()
                        total_count_body+=m_body.numel()


                if total_count < 2:
                    raise RuntimeError(
                        "潜在統計を計算するための有効データが不足しています。"
                    )

                    # 元の m.std() と整合する不偏標準偏差（correction=1）
                mean = total_sum / total_count
                var = (
                              total_sq_sum - total_count * mean.square()
                      ) / (total_count - 1)

                std = var.clamp_min(1e-8).sqrt().clamp_min(1e-4)

                # register_buffer を維持するため、代入ではなく copy_ を使う
                model.z_mean.copy_(
                    mean.to(device=model.z_mean.device, dtype=model.z_mean.dtype).reshape(1)
                )
                model.z_std.copy_(
                    std.to(device=model.z_std.device, dtype=model.z_std.dtype).reshape(1)
                )
                if hasattr(model, 'hand_vae') and hasattr(model, 'body_vae'):
                    mean_hands = total_sum_hands / total_count_hands
                    var_hands = (total_sq_sum_hands - total_count_hands * mean_hands.square()) / (total_count_hands - 1)
                    std_hands = var_hands.clamp_min(1e-8).sqrt().clamp_min(1e-4)
                    mean_body = total_sum_body / total_count_body
                    var_body = (total_sq_sum_body - total_count_body * mean_body.square()) / (total_count_body - 1)
                    std_body = var_body.clamp_min(1e-8).sqrt().clamp_min(1e-4)
                    model.z_mean_hand.copy_(
                        mean_hands.to(device=model.z_mean_hand.device, dtype=model.z_mean_hand.dtype).reshape(1)
                    )
                    model.z_std_hand.copy_(
                        std_hands.to(device=model.z_std_hand.device, dtype=model.z_std_hand.dtype).reshape(1)
                    )
                    model.z_mean_body.copy_(
                        mean_body.to(device=model.z_mean_body.device, dtype=model.z_mean_body.dtype).reshape(1)
                    )
                    model.z_std_body.copy_(
                        std_body.to(device=model.z_std_body.device, dtype=model.z_std_body.dtype).reshape(1)
                    )
            print("z_std after stats:", model.z_std.item())  # ≈0.246 になるはず
            if hasattr(model, 'hand_vae') and hasattr(model, 'body_vae'):
                print("z_std_hands after stats:", model.z_std_hand.item())  # ≈0.246 になるはず
                print("z_std_body after stats:", model.z_std_body.item())  # ≈0.246 になるはず

        model.train()
        if vae_weights!=None:
            model.vae.eval()
        step=0
        ema=EMA(model, decay=0.9999, warmup_steps=10)
        wandb.init(project="SLG_flow", entity="tkeda-jean-tokyo-city-university")
        accumulation_steps=flow_config['lr_parameters']['accumulation_steps']
        num_batches=len(dl_train)
        if weights!=None:
            checkpoint=torch.load(weights,map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            step=checkpoint['step']
            init_epoch=checkpoint['epoch']
            ema.load_state_dict(checkpoint['ema'])
            # cross-attn の gate (g_x) が開いているか

        else:
            init_epoch=0
        for epoch in range(init_epoch,epochs):
            torch.cuda.empty_cache()
            model.train()
            beta = beta_max #* min(1.0, (epoch + 1) / warmup)
            total_loss=0
            dev_total_loss=0
            test_total_loss=0
            for batch_idx,batch in enumerate(dl_train):
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path,text_tokens = batch
                padded_cod_data=padded_cod_data.float().to(device)
                padded_mask=padded_mask.float().to(device)
                text_tokens=text_tokens.to(device)
                input_length_tensor=input_length_tensor.to(device)

                all_inputs,body_inputs,left_inputs,right_inputs,norm_info=normalize_batch(padded_cod_data,used_3d)
                # 最後の区間がaccumulation_steps未満の場合にも対応
                window_start = (
                                       batch_idx // accumulation_steps
                               ) * accumulation_steps

                current_accumulation_steps = min(
                    accumulation_steps,
                    num_batches - window_start,
                )
                with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                    if used_3d:
                        new_inputs=all_inputs[:,:,:,bones_used_joints(bones=bones, include_aux=True)]
                        all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                    else:
                        new_inputs=hand_joints_to_6d(all_inputs,bones=bones).float()
                        #all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                        all_mask=torch.ones_like(new_inputs, device=device).mean(dim=-2)#(B, T, V)
                        all_mask[:, :, -40:-20] = padded_mask[:, :, 0:1]
                        all_mask[:, :, -20:] = padded_mask[:, :, 1:2]
                    #new_inputs=new_inputs*all_mask.unsqueeze(-2)
                    text_emb=text_encoder(**text_tokens,output_hidden_states=True).hidden_states[8]
                    context_mask=text_tokens['attention_mask'].to(device)
                    """
                    z0 = model._encode(new_inputs, lengths=input_length_tensor)
                    valid = model._length_mask(input_length_tensor, z0.shape[1], z0.device)
                    z0v = z0[valid] if valid is not None else z0
                    print("z0 std (valid):", z0v.std().item())  # ≈1.0 が正常
                    print("z_mean/z_std buffers:", model.z_mean.item(), model.z_std.item())
                    """
                    raw_loss, logs = model.training_loss(
                        new_inputs,cond_mask=context_mask, lengths=input_length_tensor, cond=text_emb, cond_drop_prob=0.1,
                        part_weight={"body": 1.0, "left": 2.0, "right": 2.0},all_mask=all_mask)
                    loss =raw_loss / current_accumulation_steps

                scaler.scale(loss).backward()
                is_update_step = (batch_idx + 1) % accumulation_steps == 0
                if is_update_step:
                    # Gradient Clipping前にscaleを解除
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=flow_config['lr_parameters']['grad_clip_norm'],
                    )

                    # optimizer.step()の代わり
                    scaler.step(optimizer)

                    # 次回用にscale値を更新
                    scaler.update()
                    ema.update(model)
                    # 蓄積した勾配をリセット
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    step += 1
                else:
                    if (batch_idx + 1) == num_batches:
                        # 最後のステップで更新が残っている場合は更新せず，勾配をリセットする
                        optimizer.zero_grad(set_to_none=True)
                total_loss+=raw_loss.item()*len(padded_cod_data)
            #scheduler.step()
            avg_loss=total_loss/len(ds_train)
            for batch in dl_dev:
                model.eval()
                with torch.no_grad():
                    padded_cod_data, padded_mask, input_length_tensor, id_list, data_path ,text_tokens= batch
                    padded_cod_data = padded_cod_data.float().to(device)
                    padded_mask = padded_mask.float().to(device)
                    text_tokens = text_tokens.to(device)
                    input_length_tensor = input_length_tensor.to(device)
                    all_inputs, body_inputs, left_inputs, right_inputs, norm_info = normalize_batch(padded_cod_data,
                                                                                                    used_3d)

                    if used_3d:
                        new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                        all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                    else:
                        new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                        #all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                        all_mask = torch.ones_like(new_inputs, device=device).mean(dim=-2)
                        all_mask[:, :, -40:-20] = padded_mask[:, :, 0:1]
                        all_mask[:, :, -20:] = padded_mask[:, :, 1:2]
                    #new_inputs = new_inputs * all_mask.unsqueeze(-2)
                    text_emb = text_encoder(**text_tokens,output_hidden_states=True).hidden_states[8]
                    context_mask=text_tokens['attention_mask'].to(device)

                    loss, logs = model.training_loss(
                        new_inputs, lengths=input_length_tensor, cond=text_emb, cond_drop_prob=0.1,
                        part_weight={"body": 1.0, "left": 2.0, "right": 2.0},cond_mask=context_mask,all_mask=all_mask)
                    dev_total_loss+=loss.item()*len(padded_cod_data)
            dev_avg_loss=dev_total_loss/len(ds_dev)
            for batch in dl_test:
                model.eval()
                with torch.no_grad():
                    padded_cod_data, padded_mask, input_length_tensor, id_list, data_path,text_tokens = batch
                    padded_cod_data = padded_cod_data.float().to(device)
                    #padded_mask = padded_mask.float().to(device)
                    text_tokens = text_tokens.to(device)
                    input_length_tensor = input_length_tensor.to(device)
                    all_inputs, body_inputs, left_inputs, right_inputs, norm_info = normalize_batch(padded_cod_data,
                                                                                                    used_3d)
                    if used_3d:
                        new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                        all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                    else:
                        new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                        #all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                        all_mask = torch.ones_like(new_inputs, device=device).mean(dim=-2)
                        all_mask[:, :, -40:-20] = padded_mask[:, :, 0:1]
                        all_mask[:, :, -20:] = padded_mask[:, :, 1:2]
                    new_inputs = new_inputs * all_mask.unsqueeze(-2)

                    text_emb = text_encoder(**text_tokens,output_hidden_states=True).hidden_states[8]
                    context_mask=text_tokens['attention_mask'].to(device)

                    loss, logs = model.training_loss(
                        new_inputs, lengths=input_length_tensor, cond=text_emb, cond_drop_prob=0.1,
                        part_weight={"body": 1.0, "left": 2.0, "right": 2.0},cond_mask=context_mask,all_mask=all_mask)
                    test_total_loss+=loss.item()*len(padded_cod_data)
            test_avg_loss=test_total_loss/len(ds_test)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}"
                  f", Dev Loss: {dev_avg_loss:.4f}, Test Loss: {test_avg_loss:.4f}")
            torch.save({"model":model.state_dict(),"ema":ema.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch+1,},os.path.join(save_path,f"checkpoint.pth"))
            if dev_avg_loss<min_loss:
                min_loss=dev_avg_loss
                print(f"Best loss updated: {min_loss:.4f}.")
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "epoch": epoch + 1, },
                           os.path.join(save_path, f"best_checkpoint.pth"))

            log_dict = {"train_loss": avg_loss, "dev_loss": dev_avg_loss, "test_loss": test_avg_loss}
            wandb.log(log_dict)
            if step>5000:
                i=0
                model.eval()
                for batch in dl_dev:
                    if i >= 1:
                        break
                    with torch.no_grad():
                        padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens = batch
                        padded_cod_data = padded_cod_data.float().to(device)
                        text_tokens = text_tokens.to(device)
                        input_length_tensor = input_length_tensor.to(device)
                        all_inputs, body_inputs, left_inputs, right_inputs, norm_info = normalize_batch(padded_cod_data,
                                                                                                        used_3d)
                        if used_3d:
                            new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                            all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                        else:
                            new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                            all_mask = make_mask(input_length_tensor.to(device), all_inputs)

                        text_emb = text_encoder(**text_tokens,output_hidden_states=True).hidden_states[8]
                        context_mask = text_tokens['attention_mask'].to(device)

                        idx=random.randint(0,len(padded_cod_data)-1)
                        pred_cods=ema.ema_model.sample(1,input_length_tensor[idx], lengths=input_length_tensor[idx:idx+1], cond=text_emb[idx:idx+1],cond_mask=context_mask[idx:idx+1],decode="parts",guidance_scale=5.0)


                        pred_cods=torch.cat([pred_cods['body'][0],pred_cods['left'][0],pred_cods['right'][0]],dim=-1).unsqueeze(0)

                        visualize_skeletons(pred_cods[:input_length_tensor[idx]],all_inputs[idx:idx+1,:input_length_tensor[idx]],data_path[idx:idx+1],input_length_tensor[idx:idx+1],
                                             f"{save_path}/visualize/",bones=bones,mask=all_mask[idx:idx+1])
                        i+=1

                step=0
        wandb.alert(
            title="Finish",
            text='無事学習が終了しました。'
        )

    else:

        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=ds_test.collate_fn)
        dl_dev= torch.utils.data.DataLoader(ds_dev, batch_size=1, shuffle=False, num_workers=4, collate_fn=ds_dev.collate_fn)
        ema=EMA(model, decay=0.9999, warmup_steps=10)

        """
        # ---- boundary連続性の診断 (最初のdevバッチで1回だけ) ----
        diag_batch = next(iter(dl_dev))
        padded_cod_data, _, input_length_tensor, _, _, text_tokens = diag_batch
        padded_cod_data = padded_cod_data.float().to(device)
        text_tokens = text_tokens.to(device)
        input_length_tensor = input_length_tensor.to(device)
        with torch.no_grad():
            text_emb = text_encoder(**text_tokens).last_hidden_state
        context_mask = text_tokens['attention_mask'].to(device)
        torch.manual_seed(42)  # 再現可能な診断のため固定
        sample_with_diagnostics(
            ema.ema_model, n=padded_cod_data.shape[0],
            T=int(input_length_tensor.max()),
            steps=50, cond=text_emb, cond_mask=context_mask,
            guidance_scale=3.0, lengths=input_length_tensor,
            save_plot=f"{save_path}/sampling_diagnostics.png")
        """
        ema.load_state_dict(torch.load(f"{save_path}/checkpoint.pth", map_location=device)['ema'])
        for i, blk in enumerate(ema.ema_model.expert_high.blocks):
            w = blk.mod[1].weight  # (12*D, D)
            D = w.shape[1]
            g_x = w[8 * D:9 * D]  # 12チャンク中 9番目 = cross-attn gate
            g_s = w[2 * D:3 * D]  # 3番目 = spatial gate (比較用)
            print(f"block{i}: cross gate={g_x.norm():.4f}  spatial gate={g_s.norm():.4f}")
        for i, blk in enumerate(ema.ema_model.expert_high.blocks):
            n_moe=[]
            for p in ("b", "l", "r"):
                n = blk.moe_ffn.__getattr__(f"ffn_{p}")[-1].weight.norm().item()
                n_moe.append(n)
                print(f"  {p}: {n:.2f}")
            n_moe=sum(n_moe)/len(n_moe)
            n_base = blk.ffn[-1].weight.norm().item()
            print(f"block{i}: |W2_moe|={n_moe:.4f}  |W2_base|={n_base:.4f}  ratio={n_moe / n_base:.3f}")
        ema.ema_model.eval()
        print("ema_model training:", ema.ema_model.training)
        if os.path.exists(f"{save_path}/visualize_dev/"):
            shutil.rmtree(f"{save_path}/visualize_dev/")
        if os.path.exists(f"{save_path}/visualize_test/"):
            shutil.rmtree(f"{save_path}/visualize_test/")
        os.makedirs(f"{save_path}/visualize_dev/", exist_ok=True)
        os.makedirs(f"{save_path}/visualize_test/", exist_ok=True)

        for batch in dl_dev:
            model.eval()
            with torch.no_grad():
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens = batch
                padded_cod_data = padded_cod_data.float().to(device)
                text_tokens = text_tokens.to(device)
                input_length_tensor = input_length_tensor.to(device)
                all_inputs, body_inputs, left_inputs, right_inputs, norm_info = normalize_batch(padded_cod_data,
                                                                                                used_3d)
                if used_3d:
                    new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                    all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                else:
                    new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                    all_mask = make_mask(input_length_tensor.to(device), all_inputs)

                text_emb = text_encoder(**text_tokens,output_hidden_states=True).hidden_states[8]
                context_mask = text_tokens['attention_mask'].to(device)
                torch.manual_seed(0)
                pred_cods = ema.ema_model.sample(1, input_length_tensor, lengths=input_length_tensor,
                                                 cond=text_emb, cond_mask=context_mask,guidance_scale=7.0,
                                                 decode="parts",steps=100)
                torch.manual_seed(0)
                g1 = ema.ema_model.sample(1, input_length_tensor, lengths=input_length_tensor,
                                                 cond=text_emb, cond_mask=context_mask,guidance_scale=1.0,
                                                 decode="parts",steps=100)
                torch.manual_seed(1)
                g1b = ema.ema_model.sample(1, input_length_tensor, lengths=input_length_tensor,
                                                 cond=text_emb, cond_mask=context_mask,guidance_scale=1.0,
                                                 decode="parts",steps=100)
                g1 = torch.cat([g1['body'][0], g1['left'][0], g1['right'][0]],
                                      dim=-1).unsqueeze(0)
                g1b = torch.cat([g1b['body'][0], g1b['left'][0], g1b['right'][0]],
                                      dim=-1).unsqueeze(0)
                floor = (g1 - g1b).abs().mean().item()  # 同条件でもこれだけは変わる
                scale = g1.abs().mean().item()

                torch.manual_seed(0)
                text_emb = text_encoder(**text_tokens,output_hidden_states=True).hidden_states[-1]
                g2 = ema.ema_model.sample(1, input_length_tensor, lengths=input_length_tensor,
                                                 cond=text_emb, cond_mask=context_mask,guidance_scale=1.0,
                                                 decode="parts",steps=100)
                g2 = torch.cat([g2['body'][0], g2['left'][0], g2['right'][0]],
                                      dim=-1).unsqueeze(0)
                sensitivity=(g1 - g2).abs().mean().item()
                print(f"条件感度 {sensitivity:.3f} / 同条件ばらつき {floor:.3f} / 生成スケール {scale:.3f}")
                print(f"比: 感度/ばらつき = {sensitivity / floor:.2f}")
                print("条件感度:", (g1 - g2).abs().mean().item())

                pred_cods = torch.cat([pred_cods['body'][0], pred_cods['left'][0], pred_cods['right'][0]],
                                      dim=-1).unsqueeze(0)
                B, T, C, V = pred_cods.shape
                pred_cods_to3d=reconstruct_joints_from_6d(all_inputs,pred_cods,bones=bones,bone_mask=all_mask,is_L_median=True)
                pred_cods_to3d,_=mask_zero_6d_joints(pred_cods_to3d,pred_cods,bones)
                V3 = pred_cods_to3d.shape[-1]

                for i in range(pred_cods.shape[0]):
                    pred_video_path = f"{save_path}/visualize_dev/video/{os.path.basename(data_path[i])}.mp4"
                    pred_csv_path = f"{save_path}/visualize_dev/csv/{os.path.basename(data_path[i])}.csv"
                    os.makedirs(os.path.dirname(pred_video_path), exist_ok=True)
                    os.makedirs(os.path.dirname(pred_csv_path), exist_ok=True)

                    #pred_cods = np.where(pred_cods == 0, np.nan, pred_cods)
                    #pred_cods = apply_savgol_filter(pred_cods, window_size=7, poly_order=2)
                    #pred_cods = np.where(np.isnan(pred_cods), 0, pred_cods)
                    np.savetxt(pred_csv_path, pred_cods_to3d[i][:input_length_tensor[i]].reshape(-1,3*V3).cpu().numpy(), delimiter=",")
                    # pred_coordinates=pred_coordinates[:,:,:,bones_used_joints(bones)]
                    # gt_cods=gt_cods[:,:,:,bones_used_joints(bones)]
                    save_skeleton_video(pred_cods_to3d[i][:input_length_tensor[i]].cpu().numpy(), pred_video_path,
                                        bones=bones,
                                        x_ref=all_inputs[i][:input_length_tensor[i]].cpu().numpy())
        for batch in dl_test:
            model.eval()
            with torch.no_grad():
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens = batch
                padded_cod_data = padded_cod_data.float().to(device)
                text_tokens = text_tokens.to(device)
                input_length_tensor = input_length_tensor.to(device)
                all_inputs, body_inputs, left_inputs, right_inputs, norm_info = normalize_batch(padded_cod_data,
                                                                                                used_3d)
                if used_3d:
                    new_inputs = all_inputs[:, :, :, bones_used_joints(bones=bones, include_aux=True)]
                    all_mask = make_mask(input_length_tensor.to(device), all_inputs)
                else:
                    new_inputs = hand_joints_to_6d(all_inputs, bones=bones).float()
                    all_mask = make_mask(input_length_tensor.to(device), all_inputs)

                text_emb = text_encoder(**text_tokens,output_hidden_states=True).hidden_states[8]
                context_mask = text_tokens['attention_mask'].to(device)
                pred_cods = ema.ema_model.sample(1, input_length_tensor, lengths=input_length_tensor,
                                                 cond=text_emb, cond_mask=context_mask,guidance_scale=7.0,
                                                 decode="parts")


                pred_cods = torch.cat([pred_cods['body'][0], pred_cods['left'][0], pred_cods['right'][0]],
                                      dim=-1).unsqueeze(0)
                # pred_codsのC方向ですべてが0の場合0に，それ以外を2にするマスクを作成
                mask = pred_cods.ne(0).any(dim=2).to(pred_cods.dtype)  # (B, T, V)
                B, T, C, V = pred_cods.shape
                pred_cods_to3d=reconstruct_joints_from_6d(all_inputs,pred_cods,bones=bones,bone_mask=mask)
                pred_cods_to3d,_=mask_zero_6d_joints(pred_cods_to3d,pred_cods,bones)

                V3 = pred_cods_to3d.shape[-1]

                for i in range(pred_cods.shape[0]):
                    pred_video_path = f"{save_path}/visualize_test/video/{os.path.basename(data_path[i])}.mp4"
                    pred_csv_path = f"{save_path}/visualize_test/csv/{os.path.basename(data_path[i])}.csv"
                    os.makedirs(os.path.dirname(pred_video_path), exist_ok=True)
                    os.makedirs(os.path.dirname(pred_csv_path), exist_ok=True)

                    #pred_cods = np.where(pred_cods == 0, np.nan, pred_cods)
                    #pred_cods = apply_savgol_filter(pred_cods, window_size=7, poly_order=2)
                    #pred_cods = np.where(np.isnan(pred_cods), 0, pred_cods)
                    np.savetxt(pred_csv_path, pred_cods_to3d[i][:input_length_tensor[i]].reshape(-1,3*V3).cpu().numpy(), delimiter=",")
                    # pred_coordinates=pred_coordinates[:,:,:,bones_used_joints(bones)]
                    # gt_cods=gt_cods[:,:,:,bones_used_joints(bones)]
                    save_skeleton_video(pred_cods_to3d[i][:input_length_tensor[i]].cpu().numpy(), pred_video_path,
                                        bones=bones,
                                        x_ref=all_inputs[i][:input_length_tensor[i]].cpu().numpy())


if __name__ == "__main__":
    dataset = "phoenixT"  # or "CSL-Daily", "how2sign", "phoenix", "AUTSL"
    vae_weights="/media/caffe/data_storage/CSLR/keyword_models/FlowMatching/results_slerp_inside_latent8_stride4"
    with open(f"/home/caffe/work/SLG/Parameter/config_flowmatch.yaml", "r") as f:
        config = yaml.safe_load(f)
    if os.path.exists(f"{vae_weights}/config_vae.yaml"):
        with open(f"{vae_weights}/config_vae.yaml", "r") as f:
            vae_config = yaml.safe_load(f)
        config["vae"] = vae_config["vae"]
    save_path = ("/media/caffe/data_storage/CSLR/keyword_models/FlowMatching/results_flow_cross_slerp_stride4_attn")
    if os.path.exists(f"{save_path}/flow_match.yaml"):
        with open(f"{save_path}/flow_match.yaml", "r") as f:
            config = yaml.safe_load(f)
    dataset=config['dataset']
    main(config, dataset, save_path, parts="all", visualize=False, used_3d=False, vae_weights=vae_weights)
    main(config, dataset, save_path, parts="all", visualize=True, used_3d=False, vae_weights=vae_weights)


