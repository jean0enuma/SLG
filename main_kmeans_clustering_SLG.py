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
from models.module.VQ_VAE_Transformer import VQVAETransformer1D
from SLG_datasets.SLG_datasets_Units import SLGText2UnitsDatasets
from loader import *
from loader.sign_motion_tokenizer import SignMotionTokenizer
from Parameter.Parameter import *
from trainer.VQVAE_trainer import VQVAETrainer
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
SEED=0

np.random.seed(SEED)
from utils import *
torch.autograd.set_detect_anomaly(False)
# seedを固定
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"


def integrate_path(id, path_list):
    integrated_path = []
    for path in path_list:
        integrated_path.append((id, path))
    return integrated_path


def save_tokenizer(config):
    save_path=config["tokenizer_save_path"]
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
    sign_tokenizer=SignMotionTokenizer(
        k_bodyloc=config['tokenizer_parameters']['k_bodyloc'],
        k_loc=config['tokenizer_parameters']['k_loc'],
        k_mov=config['tokenizer_parameters']['k_mov'],
        k_hs=config['tokenizer_parameters']['k_hs'],
        k_rel=config['tokenizer_parameters']['k_rel'],
        fps=25,
        use_xy_only=True,
        hold_speed_thresh=config['tokenizer_parameters']['hold_speed_thresh'],
        seed=SEED,
        # ★追加：bodyに使う関節（あなたの skeleton に合わせて変える）
        body_joint_ids=(0, 2, 3, 4, 5),
        left_sh_id=2,
        right_sh_id=3,
    )
    i=0
    for id, data_path in train_data_path:
        i+=1
        if id==3:
            data_path="/".join(data_path.split("/")[:-1])
        print(f"{i}/{len(train_data_path)} Processing:", data_path)
        file_name=data_path.split("/")[-1].split(".mp4")[0]
        face_data_path=f"{train_face_root[id]}/{file_name}.csv"
        cod_data_path=f"{train_cod_root[id]}/{file_name}.csv"
        face_data=np.loadtxt(f"{face_data_path}",delimiter=",",dtype=np.float32)
        cod_data=np.loadtxt(f"{cod_data_path}",delimiter=",",dtype=np.float32)
        data = nan_interpolate(cod_data)
        # data=nan_interpolate_zero(data)
        if np.isnan(data).any():
            print("nan")
        #data = average_movint(data)
        new_data = np.zeros((2, data.shape[0], data.shape[1]// 3))
        new_data[0] = data[:, 0::3]  # data:(2,frame,point)
        new_data[1] = data[:, 1::3]
        new_hand_data = new_data[:, :, 6:]
        zero_mask = np.all(new_hand_data == 0, axis=(0, 2))  # shape: (S,)
        nan_indexes = np.where(zero_mask)[0]
        new_data = np.delete(new_data, nan_indexes, axis=1)
        new_body_data = new_data[:, :, :6]
        #new_data_face = np.delete(new_data_face, nan_indexes, axis=1)
        new_hand_data = np.delete(new_hand_data, nan_indexes, axis=1)
        new_body_data = np.delete(new_body_data, nan_indexes, axis=1)
        left_data=new_hand_data[:, :,:new_hand_data.shape[2]//2]
        right_data=new_hand_data[:,:, new_hand_data.shape[2]//2:]
        sample_frame=face_data.shape[0]
        new_body_data=new_data.transpose((1,2,0)) #(frame,point,2)
        left_data=left_data.transpose((1,2,0)) #(frame,point,2)
        right_data=right_data.transpose((1,2,0)) #(frame,point,2)
        sign_tokenizer.fit_codebooks([(new_body_data, left_data, right_data)],sample_frames=sample_frame, max_iter=30)

    #tokenizerの保存
    sign_tokenizer.save(save_path)
    print("Tokenizers created.")

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
    postfix = ""
    if config["dataset_parameters"]["use_phoenixT"]:
        postfix = "_phoenixT"
    if config["dataset_parameters"]["use_csl-daily"]:
        postfix += "_csl_daily"
    if config["dataset_parameters"]["use_how2sign"]:
        postfix += "_how2sign"
    if config["dataset_parameters"]["use_phoenix"]:
        postfix += "_phoenix"
    save_path = f"/home/caffe/work/SLG/Parameter/tokenizer_config_{postfix}.npz"
    config["tokenizer_save_path"]=save_path
    save_tokenizer(config)
    # print("Process time: ", time.time() - start)
