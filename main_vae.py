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
from models.module.VAE_diffusion import VAETransformerDiffusion
from models.module.VAE_Transformer import VAE_Transformer
from SLG_datasets.SLG_datasets_Units import SLGText2UnitsDatasets
from loader import *
from Parameter.Parameter import *
from trainer.VQVAE_trainer import VQVAETrainer
from trainer.VAE_trainer import VAETrainer
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
        if config['dataset_parameters']['is_processed']:
            train_cod_root[0] = SKELETON_TRAIN_DATADIR_T_PROCESSED
            dev_cod_root[0] = SKELETON_DEV_DATADIR_T_PROCESSED
            test_cod_root[0] = SKELETON_TEST_DATADIR_T_PROCESSED

            train_face_root[0] = FACE_TRAIN_DATADIR_T_PROCESSED
            dev_face_root[0] = FACE_DEV_DATADIR_T_PROCESSED
            test_face_root[0] = FACE_TEST_DATADIR_T_PROCESSED
            is_3d=True
        else:
            train_cod_root[0] = SKELETON_TRAIN_DATADIR_T_3D
            dev_cod_root[0] = SKELETON_DEV_DATADIR_T_3D
            test_cod_root[0] = SKELETON_TEST_DATADIR_T_3D

            train_face_root[0] = FACE_TRAIN_DATADIR_T_3D
            dev_face_root[0] = FACE_DEV_DATADIR_T_3D
            test_face_root[0] = FACE_TEST_DATADIR_T_3D
            is_3d=True

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
        if config['dataset_parameters']['is_processed']:
            train_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_PROCESSED
            dev_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_PROCESSED
            test_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_PROCESSED

            train_face_root[1] = FACE_CSL_DAILY_DATADIR_PROCESSED
            dev_face_root[1] = FACE_CSL_DAILY_DATADIR_PROCESSED
            test_face_root[1] = FACE_CSL_DAILY_DATADIR_PROCESSED
            is_3d=True
        else:
            train_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_3D
            dev_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_3D
            test_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_3D

            train_face_root[1] = FACE_CSL_DAILY_DATADIR_3D
            dev_face_root[1] = FACE_CSL_DAILY_DATADIR_3D
            test_face_root[1] = FACE_CSL_DAILY_DATADIR_3D
            is_3d=True
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
        if config['dataset_parameters']['is_processed']:
            train_cod_root[2] = SKELETON_HOW2SIGN_TRAIN_DATADIR_PROCESSED
            dev_cod_root[2] = SKELETON_HOW2SIGN_DEV_DATADIR_PROCESSED
            test_cod_root[2] = SKELETON_HOW2SIGN_TEST_DATADIR_PROCESSED

            train_face_root[2] = FACE_HOW2SIGN_TRAIN_DATADIR_PROCESSED
            dev_face_root[2] = FACE_HOW2SIGN_DEV_DATADIR_PROCESSED
            test_face_root[2] = FACE_HOW2SIGN_TEST_DATADIR_PROCESSED
            is_3d=True
        else:
            train_cod_root[2] = SKELETON_HOW2SIGN_TRAIN_DATADIR_3D
            dev_cod_root[2] = SKELETON_HOW2SIGN_DEV_DATADIR_3D
            test_cod_root[2] = SKELETON_HOW2SIGN_TEST_DATADIR_3D

            train_face_root[2] = FACE_HOW2SIGN_TRAIN_DATADIR_3D
            dev_face_root[2] = FACE_HOW2SIGN_DEV_DATADIR_3D
            test_face_root[2] = FACE_HOW2SIGN_TEST_DATADIR_3D
            is_3d=True

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
        if config['dataset_parameters']['is_processed']:
            train_cod_root[3] = SKELETON_TRAIN_DATADIR_PROCESSED
            dev_cod_root[3] = SKELETON_DEV_DATADIR_PROCESSED
            test_cod_root[3] = SKELETON_TEST_DATADIR_PROCESSED

            train_face_root[3] = FACE_TRAIN_DATADIR_PROCESSED
            dev_face_root[3] = FACE_DEV_DATADIR_PROCESSED
            test_face_root[3] = FACE_TEST_DATADIR_PROCESSED
            is_3d=True
        else:
            train_cod_root[3] = SKELETON_TRAIN_DATADIR
            dev_cod_root[3] = SKELETON_DEV_DATADIR
            test_cod_root[3] = SKELETON_TEST_DATADIR

            train_face_root[3] = FACE_TRAIN_DATADIR
            dev_face_root[3] = FACE_DEV_DATADIR
            test_face_root[3] = FACE_TEST_DATADIR
            is_3d=False

        train_data_path += integrate_path(3, phoenix_train_path)
        dev_data_path += integrate_path(3, phoenix_dev_path)
        test_data_path += integrate_path(3, phoenix_test_path)
        i += 1
    if i == 0:
        raise ValueError("At least one dataset must be selected.")
    config['model']['decoder_num_lang'] = i + 1
    print("Datasets loaded.")
    print("保存場所:", save_path)
    print("Is GPU available?:", torch.cuda.is_available())

    device = config["device"] if torch.cuda.is_available() else "cpu"
    #deviceからgpuの名前を取得して表示
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(int(device[-1]))
        print("GPU name:", gpu_name)
    print("---Loading tokenizer---")
    print("---Creating datasets---")
    ds_train = SLGText2UnitsDatasets(train_data_path, train_cod_root, train_face_root, is_3d=is_3d,is_processed=config['dataset_parameters']['is_processed'],is_sg_filter=False,is_coarse=False,
                                trainable=True)
    ds_dev = SLGText2UnitsDatasets(dev_data_path, dev_cod_root, dev_face_root, trainable=False,is_3d=is_3d,is_processed=config['dataset_parameters']['is_processed'],is_sg_filter=False,is_coarse=False)
    ds_test = SLGText2UnitsDatasets(test_data_path, test_cod_root, test_face_root,trainable=False,is_3d=is_3d,is_processed=config['dataset_parameters']['is_processed'],is_sg_filter=False,is_coarse=False)

    if ds_train.is_3d:
        config['model']['encoder']['q_input_dim'] = int(config['model']['encoder']['q_input_dim']*1.5)  # 3Dの場合の入力サイズ
        config['model']['encoder']['kv_input_dim'] = int(config['model']['encoder']['kv_input_dim']*1.5)  # 3Dの場合の出力サイズ
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
    config['model']['anchor_frame_path'] = ANCHOR_FRAME_PATH
    model = VAETransformerDiffusion(config['model']).to(device)

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
    trainer = VAETrainer(config, scheduler)
    #trainer=VQVAETrainer(config, scheduler)
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
        trainer.visualize(model, ds_test, device,is_3d=is_3d)
    elif mode == "visualize":
        trainer.visualize(model, ds_test, device,is_3d=is_3d)
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
    mode = "train"  # Change this to "test" when you want to test
    checkpoint =None
    # subprocess.run(command, input=("gazouken\n").encode(), check=True)
    # print("無効化完了")
    start = time.time()
    print("Loading config...")
    with open(f"/home/caffe/work/SLG/Parameter/config_diffusion.yaml", "r") as f:
        config = yaml.safe_load(f)
    config['model']['text_cond'] = False
    print("Config loaded.")
    # logディレクトリにContinurous_Sign以下のディレクトリ，ファイルをコピー
    if checkpoint != None:
        if checkpoint.split(".")[-1] == "cpt":
            save_path=checkpoint.split("/")[:-1]
        else:
            save_path = checkpoint.split("/")[:-2]
        save_path = "/".join(save_path)
        with open(f"{save_path}/config_diffusion.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        while True:
            dt_now = datetime.datetime.now()
            save_path = copy.deepcopy(f"/media/caffe/data_storage/CSLR/keyword_models/train/{dt_now.strftime('%Y/%m%d/%H%M')}")
            print("保存先のパス:", save_path)
            if os.path.exists(save_path):
                time.sleep(10)  # もし保存先のディレクトリが既に存在していたら、1分待ってから再度確認する(他のプロセスが保存している可能性があるため)
            else:
                 break
        log_create_dir(save_path)
        # copy_dir(PROJECT_DIR, save_path)
        shutil.copy(f"/home/caffe/work/SLG/Parameter/config_diffusion.yaml", f"{save_path}/config_diffusion.yaml")
        copy_dir(PROJECT_DIR, save_path)
        shutil.rmtree(f"{save_path}/Continurous_Sign/wandb")  # wandbディレクトリはコピー後に削除(ログが混ざるのを防ぐため)

    config['save_path'] = save_path
    main(config, mode, checkpoint=checkpoint)
    # print("Process time: ", time.time() - start)
