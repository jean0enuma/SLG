import warnings

import torch.utils.data

from Parameter.Parameter_learning_env import *
from Parameter.file_control import *

warnings.simplefilter('ignore')
import torch.optim as optim
from models.scheduler.scheduler import CosineAnnealingLR
from torchvision.transforms.v2 import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ColorJitter, \
    RandAugment, GaussianBlur

import time
from models.text2pose import Text2Pose
from models.module.VQ_VAE import VQVAE1D,VQLossWeights
from models.module.VQ_VAE_Transformer import VQVAETransformer1D,VQVAETransformer1DSeparated
from SLG_datasets.SLG_datasets_Units import SLGText2UnitsDatasets
from loader import *
from Parameter.Parameter import *
from trainer.VQVAE_trainer import VQVAETrainer
from trainer.VQVAE_separate_trainer import VQVAESeparateTrainer
import csv, json
import wandb
import copy
import cv2

cv2.setNumThreads(0)

import numpy as np
import subprocess
import yaml
import faulthandler

faulthandler.enable()

np.random.seed(0)
from utils import *

torch.autograd.set_detect_anomaly(False)
# seedを固定
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"


def integrate_path(id, path_list):
    integrated_path = []
    for path in path_list:
        integrated_path.append((id, path))
    return integrated_path


def main(config, mode, checkpoint):
    save_path = config["save_path"]
    print("保存場所:", save_path)
    print("Is GPU available?:", torch.cuda.is_available())

    device = config["device"] if torch.cuda.is_available() else "cpu"
    print("---Loading datasets---")

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
    if config['dataset_parameters']['use_phoenixT']:
        phoenixT_train_path, phoenixT_dev_path, phoenixT_test_path, phoenixT_train_corpus, phoenixT_dev_corpus, phoenixT_test_corpus = datasets_loader_T(
            "phoenixT")
        train_corpus[0] = phoenixT_train_corpus
        dev_corpus[0] = phoenixT_dev_corpus
        test_corpus[0] = phoenixT_test_corpus

        train_cod_root[0] = SKELETON_TRAIN_DATADIR_T_PROCESSED
        dev_cod_root[0] = SKELETON_DEV_DATADIR_T_PROCESSED
        test_cod_root[0] = SKELETON_TEST_DATADIR_T_PROCESSED

        train_face_root[0] = FACE_TRAIN_DATADIR_T_PROCESSED
        dev_face_root[0] = FACE_DEV_DATADIR_T_PROCESSED
        test_face_root[0] = FACE_TEST_DATADIR_T_PROCESSED

        train_data_path += integrate_path(0, phoenixT_train_path)
        dev_data_path += integrate_path(0, phoenixT_dev_path)
        test_data_path += integrate_path(0, phoenixT_test_path)
        i += 1
    if config['dataset_parameters']['use_csl-daily']:
        csl_daily_train_path, csl_daily_dev_path, csl_daily_test_path, csl_daily_train_corpus, csl_daily_dev_corpus, csl_daily_test_corpus = datasets_loader_T(
            "CSL-Daily")

        train_corpus[1] = csl_daily_train_corpus
        dev_corpus[1] = csl_daily_dev_corpus
        test_corpus[1] = csl_daily_test_corpus

        train_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_PROCESSED
        dev_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_PROCESSED
        test_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_PROCESSED

        train_face_root[1] = FACE_CSL_DAILY_DATADIR_PROCESSED
        dev_face_root[1] = FACE_CSL_DAILY_DATADIR_PROCESSED
        test_face_root[1] = FACE_CSL_DAILY_DATADIR_PROCESSED
        train_data_path += integrate_path(1, csl_daily_train_path)
        dev_data_path += integrate_path(1, csl_daily_dev_path)
        test_data_path += integrate_path(1, csl_daily_test_path)
        i += 1
    if config['dataset_parameters']['use_how2sign']:
        how2sign_train_path, how2sign_dev_path, how2sign_test_path, how2sign_train_corpus, how2sign_dev_corpus, how2sign_test_corpus = datasets_loader_T(
            "how2sign")
        train_corpus[2] = how2sign_train_corpus
        dev_corpus[2] = how2sign_dev_corpus
        test_corpus[2] = how2sign_test_corpus

        train_cod_root[2] = SKELETON_HOW2SIGN_TRAIN_DATADIR_PROCESSED
        dev_cod_root[2] = SKELETON_HOW2SIGN_DEV_DATADIR_PROCESSED
        test_cod_root[2] = SKELETON_HOW2SIGN_TEST_DATADIR_PROCESSED

        train_face_root[2] = FACE_HOW2SIGN_TRAIN_DATADIR_PROCESSED
        dev_face_root[2] = FACE_HOW2SIGN_DEV_DATADIR_PROCESSED
        test_face_root[2] = FACE_HOW2SIGN_TEST_DATADIR_PROCESSED

        train_data_path += integrate_path(2, how2sign_train_path)
        dev_data_path += integrate_path(2, how2sign_dev_path)
        test_data_path += integrate_path(2, how2sign_test_path)
        i += 1
    if config['dataset_parameters']['use_phoenix']:
        phoenix_train_path, phoenix_dev_path, phoenix_test_path, phoenix_train_corpus, phoenix_dev_corpus, phoenix_test_corpus = datasets_loader_T(
            "phoenix")
        train_corpus[3] = phoenix_train_corpus
        dev_corpus[3] = phoenix_dev_corpus
        test_corpus[3] = phoenix_test_corpus

        train_cod_root[3] = SKELETON_TRAIN_DATADIR_PROCESSED
        dev_cod_root[3] = SKELETON_DEV_DATADIR_PROCESSED
        test_cod_root[3] = SKELETON_TEST_DATADIR_PROCESSED

        train_face_root[3] = FACE_TRAIN_DATADIR_PROCESSED
        dev_face_root[3] = FACE_DEV_DATADIR_PROCESSED
        test_face_root[3] = FACE_TEST_DATADIR_PROCESSED

        train_data_path += integrate_path(3, phoenix_train_path)
        dev_data_path += integrate_path(3, phoenix_dev_path)
        test_data_path += integrate_path(3, phoenix_test_path)
        i += 1
    if i == 0:
        raise ValueError("At least one dataset must be selected.")
    config['model']['decoder_num_lang'] = i + 1
    for id, path in train_data_path:
        try:
            print(path)
        except UnicodeEncodeError:
            print("UnicodeEncodeError! Removing...")
            train_data_path.remove((id, path))
            print(f"Removed this path from train_path")
    print("Datasets loaded.")
    print("---Loading tokenizer---")
    print("---Creating datasets---")
    ds_train = SLGText2UnitsDatasets(train_data_path, train_cod_root, train_face_root, is_3d=False,is_processed=True,is_sg_filter=True,
                                trainable=True)
    ds_dev = SLGText2UnitsDatasets(dev_data_path, dev_cod_root, dev_face_root, trainable=False,is_3d=False,is_processed=True,is_sg_filter=True)
    ds_test = SLGText2UnitsDatasets(test_data_path, test_cod_root, test_face_root,trainable=False,is_3d=False,is_processed=True,is_sg_filter=True)

    if ds_train.is_3d or ds_train.is_processed:
        config['model']['pose_dim'] = int(config['model']['pose_dim']*1.5)  # 3Dの場合の入力サイズ
    postfix = ""
    if config["dataset_parameters"]["use_phoenixT"]:
        postfix = "_phoenixT"
    if config["dataset_parameters"]["use_csl-daily"]:
        postfix += "_csl_daily"
    if config["dataset_parameters"]["use_how2sign"]:
        postfix += "_how2sign"

    #config['loss_parameters'][
    #    'max_length'] = ds_train.show_max_length() * 1.5  # loss_parametersのmax_lengthをデータセットの最大長の1.5倍に設定
    print("Datasets created.")
    print(f"Number of training samples: {len(ds_train)}")
    print(f"Number of dev samples: {len(ds_dev)}")
    print(f"Number of test samples: {len(ds_test)}")
    # データローダーの作成
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=config['lr_parameters']['batch_size'], shuffle=True,
                                           num_workers=2, collate_fn=ds_train.collate_fn, drop_last=True)
    dl_dev = torch.utils.data.DataLoader(ds_dev, batch_size=config['lr_parameters']['batch_size'], shuffle=False,
                                         num_workers=2,
                                         collate_fn=ds_dev.collate_fn)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=config['lr_parameters']['batch_size'], shuffle=False,
                                          num_workers=2, collate_fn=ds_test.collate_fn)
    print("DataLoaders created.")
    # モデルの作成
    print("---Creating model---")
    #model = Text2Pose(config["model"]).float().to(device)
    loss_w=VQLossWeights()
    loss_w.recon_pos=config['loss_parameters']['recon_pos_weight']
    loss_w.recon_dir=config['loss_parameters']['recon_dir_weight']
    loss_w.vq=config['loss_parameters']['vq_weight']
    """
    model = VQVAE1D(
        in_dim=config["model"]["in_dim"],
        hidden=config["model"]["hidden_dim"],
        code_dim=config["model"]["code_dim"],
        n_codes=config["model"]["n_codes"],
        stride=config["model"]["stride"],
        n_res_blocks=config["model"]["n_res_blocks"],
        dropout=config["model"]["dropout"],
        rvq_stages=config["model"]["rvq_stages"],
        vq_beta=config["model"]["vq_beta"],
        loss_w=loss_w
    ).float().to(device)
    """
    """
    model = VQVAETransformer1D(
        in_dim=config["model"]["in_dim"],
        d_model=config["model"]["hidden_dim"],
        n_heads=config["model"]["n_heads"],
        code_dim=config["model"]["code_dim"],
        n_codes=config["model"]["n_codes"],
        stride=config["model"]["stride"],
        n_layers_enc=config["model"]["n_layers_enc"],
        n_layers_dec=config["model"]["n_layers_dec"],
        ff_mult=config["model"]["ff_mult"],
        dropout=config["model"]["dropout"],
        rvq_stages=config["model"]["rvq_stages"],
        vq_beta=config["model"]["vq_beta"],
        loss_w=loss_w
    ).float().to(device)
    """
    model = VQVAETransformer1DSeparated(
        pose_d_model=config["model"]['separated_vae']['pose_d_model'],
        hand_d_model=config["model"]['separated_vae']['hand_d_model'],
        extra_d_model=config["model"]['separated_vae']['extra_d_model'],
        n_pose_layers_enc=config["model"]['separated_vae']['n_pose_layers_enc'],
        n_hand_layers_enc=config["model"]['separated_vae']['n_hand_layers_enc'],
        n_extra_layers_enc=config["model"]['separated_vae']['n_extra_layers_enc'],
        n_pose_layers_dec=config["model"]['separated_vae']['n_pose_layers_dec'],
        n_hand_layers_dec=config["model"]['separated_vae']['n_hand_layers_dec'],
        n_extra_layers_dec=config["model"]['separated_vae']['n_extra_layers_dec'],
        n_pose_heads=config["model"]['separated_vae']['n_pose_heads'],
        n_hand_heads=config["model"]['separated_vae']['n_hand_heads'],
        n_extra_heads=config["model"]['separated_vae']['n_extra_heads'],
        pose_code_dim=config["model"]['separated_vae']['pose_code_dim'],
        hand_code_dim=config["model"]['separated_vae']['hand_code_dim'],
        extra_code_dim=config["model"]['separated_vae']['extra_code_dim'],
        n_pose_codes=config["model"]['separated_vae']['n_pose_codes'],
        n_hand_codes=config["model"]['separated_vae']['n_hand_codes'],
        n_extra_codes=config["model"]['separated_vae']['n_extra_codes'],
        stride=config["model"]["stride"],
        ff_mult=config["model"]["ff_mult"],
        dropout=config["model"]["dropout"],
        rvq_stages=config["model"]["rvq_stages"],
        vq_beta=config["model"]["vq_beta"],
        loss_w=loss_w
    ).float().to(device)
    # モデルの保存
    if checkpoint != None and checkpoint.split(".")[-1] == "cpt":
        model.load_state_dict(torch.load(checkpoint, weights_only=False, map_location=device)["model_state_dict"])
        config['init_epoch'] = torch.load(checkpoint, weights_only=False)["epoch"] + 1
        # checkpointにあるキーの値をconfigの対応するキーに代入
        for key in config.keys():
            if key in torch.load(checkpoint, weights_only=False).keys():
                config[key] = torch.load(checkpoint, weights_only=False)[key]
    elif checkpoint != None and checkpoint.split(".")[-1] == "pth":
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        config['init_epoch'] = 0
    else:
        config['init_epoch'] = 0
    print(f"Number of parameters:{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("Model created.")
    # optimizer,criterion,lr_schedulerの作成
    print("---Creating optimizer, criterion, and lr_scheduler---")
    optimizer = optim.AdamW(model.parameters(), lr=config["lr_parameters"]['learning_rate'],
                           weight_decay=config["lr_parameters"]['weight_decay'])

    #criterion = Text2Pose_criterion(config['loss_parameters'])
    if config['lr_parameters']['scheduler_type'] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["lr_parameters"]["epoch"],
                                                         eta_min=config["lr_parameters"]["min_lr"])
    elif config['lr_parameters']['scheduler_type'] == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingLR(optimizer, warmup_epochs=int(config["lr_parameters"]["epoch"]*0.02),
                                      max_epochs=config["lr_parameters"]["epoch"],
                                      warmup_start_lr=config["lr_parameters"]["min_lr"],
                                      eta_min=config["lr_parameters"]["min_lr"])
    elif config['lr_parameters']['scheduler_type'] == "StepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['lr_parameters']['milestones'],
                                                 gamma=config['lr_parameters']['gamma'])
    print("Optimizer, criterion, and lr_scheduler created.")
    # 学習の実行
    print("---Starting training/evaluation---")
    trainer = VQVAESeparateTrainer(config, scheduler)
    if mode == "train":
        if checkpoint != None and checkpoint.split(".")[-1] == "cpt":
            # id名を取得
            with open(f"{save_path}/wandb_id.txt", "r") as f:
                id = f.read()
            wandb.init(project="SLG_TEXT2POSE", id=id, resume="allow", entity="tkeda-jean-tokyo-city-university")
        else:
            wandb.init(project="SLG_TEXT2POSE", entity="tkeda-jean-tokyo-city-university")
            id = wandb.run.id
            # id名を保存
            with open(f"{save_path}/wandb_id.txt", "w") as f:
                f.write(id)
        trainer.fit(model, optimizer, scheduler, None, dl_train, dl_dev, dl_test, device,
                    early_stopping=None)
    elif mode == "visualize":
        trainer.visualize(model, dl_test)
    else:
        trainer.eval(model, None, dl_test, device)
    print("Training/evaluation finished.")


if __name__ == "__main__":
    command = ['sudo', 'systemctl', 'stop', 'systemd-oomd']
    print("OOM killerを無効化")
    # subprocess.run(command, input=("gazouken\n").encode(), check=True)
    print("無効化完了")
    # global LOG_DIR
    # "train"か"eval"を指定(変数名を考えて)
    mode = "train"
    checkpoint =None
    # subprocess.run(command, input=("gazouken\n").encode(), check=True)
    # print("無効化完了")
    start = time.time()
    print("Loading config...")
    with open(f"/home/caffe/work/SLG/Parameter/config_vqvae.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Config loaded.")
    # logディレクトリにContinurous_Sign以下のディレクトリ，ファイルをコピー
    if checkpoint != None:
        save_path = checkpoint.split("/")[:-2]
        save_path = "/".join(save_path)
        with open(f"{save_path}/config_vqvae.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        save_path = copy.deepcopy(SAVE_MODEL_DIR_KWT)
        while True:
            if os.path.exists(save_path):
                time.sleep(60)  # もし保存先のディレクトリが既に存在していたら、1分待ってから再度確認する(他のプロセスが保存している可能性があるため)
            else:
                 break
        log_create_dir(save_path)
        # copy_dir(PROJECT_DIR, save_path)
        shutil.copy(f"/home/caffe/work/SLG/Parameter/config_vqvae.yaml", f"{save_path}/config_vqvae.yaml")

    config['save_path'] = save_path
    main(config, mode, checkpoint=checkpoint)
    # print("Process time: ", time.time() - start)
