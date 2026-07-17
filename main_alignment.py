####
#既存モデルで単語分割を行うためのコード
###
import warnings
from tqdm import tqdm

import torch.utils.data

from Parameter.Parameter_learning_env import *
from Parameter.file_control import *
from loader.bounding_loader import clip_video_bbox
from loader.clip_parts import clip_parts

warnings.simplefilter('ignore')
import torch.optim as optim
from models.scheduler.scheduler import CosineAnnealingLR
from torchvision.transforms.v2 import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ColorJitter, \
    RandAugment, GaussianBlur

import time
from models.text2pose import Text2Pose
from models.Corrnet_plus.corrnet_plus import SLRModel
from SLG_datasets.phoenix_datasets_CTC import Phoenix_datasets_CTC
from trainer.VAE_sync_trainer import VAESyncTrainer
from models.module.VAE_sync import VAE_sync

from loader import *
from Parameter.Parameter import *
from trainer.Text2VAE_trainer import Text2VAETrainer
import csv, json
import wandb
import copy
from transformers import AutoTokenizer
import cv2

cv2.setNumThreads(0)

import numpy as np
import subprocess
import yaml
import faulthandler

faulthandler.enable()

np.random.seed(0)
from utils import *
from loader.coordinate_preprocess import nan_interpolate
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
def main(dataset, checkpoint=None, skeleton=False):
    print("Is GPU available?:", torch.cuda.is_available())

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("---Loading datasets---")

    transform = {}
    transform["train"] = Compose([
        CenterCrop(224),
        # RandomRotation(10),
        #RandomHorizontalFlip(0.5),
        #TemporalRescale(max_len=230, min_len=32, temp_scaling=0.2)
    ])
    transform["eval"] = Compose([
        CenterCrop(224),
        # Resize((224,224)),
        # ToTensor(),
        # UniformTemporalSubsample(200)
    ])
    train_path, dev_path, test_path, train_corpus, dev_corpus, test_corpus = datasets_loader(dataset)
    # train_path= train_path[:100] #デバッグ用
    # dev_path= dev_path[:200] #デバッグ用
    # test_path= test_path[:200] #デバッグ用

    if dataset == "phoenix":
        ext="png"
        gloss_filename = "gloss_dict.npy"
        tmp = np.load(f"./Parameter/{gloss_filename}", allow_pickle=True).item()
        #skeleton_train_path=SKELETON
        #skeleton_dev_path=SKELETON_DEV_DATADIR_3D
        #skeleton_test_path=SKELETON_TEST_DATADIR_3D
        gloss2class = {}
        class2gloss = {}
        for k, v in tmp.items():
            gloss2class[k] = v[0]
            class2gloss[v[0]] = k
    elif dataset == "phoenixT":
        ext="png"
        gloss_filename = "gloss_dict_T.npy"
        tmp = np.load(f"./Parameter/{gloss_filename}", allow_pickle=True).item()
        skeleton_train_path=SKELETON_TRAIN_DATADIR_T_3D
        skeleton_dev_path=SKELETON_DEV_DATADIR_T_3D
        skeleton_test_path=SKELETON_TEST_DATADIR_T_3D
        skeleton_train_face_path=FACE_TRAIN_DATADIR_T_3D
        skeleton_dev_face_path=FACE_DEV_DATADIR_T_3D
        skeleton_test_face_path=FACE_TEST_DATADIR_T_3D
        gloss2class = {}
        class2gloss = {}
        for k, v in tmp.items():
            gloss2class[k] = v
            class2gloss[v] = k
    elif dataset == "CSL-Daily":
        ext="jpg"
        skeleton_train_path=SKELETON_CSL_DAILY_DATADIR_3D
        skeleton_dev_path=SKELETON_CSL_DAILY_DATADIR_3D
        skeleton_test_path=SKELETON_CSL_DAILY_DATADIR_3D
        skeleton_train_face_path=FACE_CSL_DAILY_DATADIR_3D
        skeleton_dev_face_path=FACE_CSL_DAILY_DATADIR_3D
        skeleton_test_face_path=FACE_CSL_DAILY_DATADIR_3D
        gloss_filename = "gloss_dict_CSL-Daily.npy"
        tmp = np.load(f"./Parameter/{gloss_filename}", allow_pickle=True).item()
        gloss2class = {}
        class2gloss = {}
        for k, v in tmp.items():
            gloss2class[k] = v[0]
            class2gloss[v[0]] = k

    else:
        raise ValueError("DATASET must be phoenix or phoenixT")


    print(f"# of train data:{len(train_path)}")
    print(f"# of eval data:{len(dev_path)}")
    max_targets_length = max([len(i.split(" ")) for i in train_corpus["annotation"].values])

    print(f"最大ラベル長:{max_targets_length}")
    print(f"# of train data:{len(train_path)}")
    # max_dev_target_length = max([len(i.split(" ")) for i in dev_corpus["annotation"].values])
    print(f"# of eval data:{len(dev_path)}")

    num_class = len(gloss2class)
    print(f"Number of classes:{num_class}")


    for path in train_path:
        try:
            print(path)
        except UnicodeEncodeError:
            print("UnicodeEncodeError! Removing...")
            train_path.remove(path)
            print(f"Removed this path from train_path")
    print("Datasets loaded.")
    print("---Loading tokenizer---")
    print("---Creating datasets---")
    #gloss2class, class2gloss = convert_gloss_to_class(pd.concat([train_corpus]))
    num_class = len(gloss2class)
    print(f"グロス数:{num_class}")
    # 訓練用の設定
    ds_train = Phoenix_datasets_CTC(train_path, train_corpus, gloss2class, transform["train"], trainable=False,ext=ext)
    ds_eval = Phoenix_datasets_CTC(dev_path, dev_corpus, gloss2class, transform["eval"], trainable=False,ext=ext)
    ds_test = Phoenix_datasets_CTC(test_path, test_corpus, gloss2class, transform["eval"], trainable=False,ext=ext)
    # trainデータローダーの定義
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=2, shuffle=True, num_workers=2 ,
        collate_fn=ds_train.collate_fn, drop_last=True,
        pin_memory=True)
    dl_eval= torch.utils.data.DataLoader(
        ds_eval, batch_size=2, shuffle=False, num_workers=2,
        collate_fn=ds_eval.collate_fn, drop_last=False,
        pin_memory=True)
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size=2, shuffle=False, num_workers=2,
        collate_fn=ds_test.collate_fn, drop_last=False,
        pin_memory=True)

    model=SLRModel(num_classes=len(gloss2class)+1,gloss_dict=gloss2class,conv_type=2,use_bn=1)
    if dataset=="phoenix":
        if skeleton:
            save_path_skeleton = "/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/Alignment_words_skeleton"
            save_path_face="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/Alignment_words_face"
        save_path="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/Alignment_words"
        model.load_state_dict(torch.load("/home/caffe/work/SLG/Parameter/phoenix2014_dev_18.00.pt",weights_only=False)["model_state_dict"])
    elif dataset=="phoenixT":
        if skeleton:
            save_path_skeleton = "/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Alignment_words_skeleton_midpoint"
            save_path_face= "/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Alignment_words_face_midpoint"
        save_path="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Alignment_words_midpoint"
        model.load_state_dict(torch.load("/home/caffe/work/SLG/Parameter/phoenix2014-T_dev_17.20.pt",weights_only=False)["model_state_dict"])
    elif dataset=="CSL-Daily":
        if skeleton:
            save_path_skeleton  = "/media/caffe/data_storage/CSL-Daily/sentence/Alignment_words_skeleton"
            save_path_face= "/media/caffe/data_storage/CSL-Daily/sentence/Alignment_words_face"
        save_path="/media/caffe/data_storage/CSL-Daily/sentence/Alignment_words"
        model.load_state_dict(torch.load("/home/caffe/work/SLG/Parameter/CSL_Daily_dev_28.60.pt",weights_only=False)["model_state_dict"])
    model.to(device)
    #save_pathを作成
    split=["train","dev","test"]
    if os.path.exists(save_path):
        print(f"{save_path} already exists. Removing...")
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    if skeleton:
        if os.path.exists(save_path_skeleton):
            print(f"{save_path_skeleton} already exists. Removing...")
            shutil.rmtree(save_path_skeleton)
        os.makedirs(save_path_skeleton, exist_ok=True)
        if os.path.exists(save_path_face):
            print(f"{save_path_face} already exists. Removing...")
            shutil.rmtree(save_path_face)
        os.makedirs(save_path_face, exist_ok=True)

    for s in split:
        save_path_split=f"{save_path}/{s}"
        os.makedirs(save_path_split, exist_ok=True)


        if skeleton:
            save_path_skeleton_face_split=f"{save_path_face}/{s}"
            save_path_skeleton_split = f"{save_path_skeleton}/{s}"
            os.makedirs(save_path_skeleton_face_split, exist_ok=True)
            os.makedirs(save_path_skeleton_split, exist_ok=True)
        if s=="train":
            dl=dl_train
            skeleton_dir=f"{skeleton_train_path}"
            skeleton_face_dir=f"{skeleton_train_face_path}"
        elif s=="dev":
            dl=dl_eval
            skeleton_dir=f"{skeleton_dev_path}"
            skeleton_face_dir=f"{skeleton_dev_face_path}"
        elif s=="test":
            dl=dl_test
            skeleton_dir=f"{skeleton_test_path}"
            skeleton_face_dir=f"{skeleton_test_face_path}"
        #データを単語ごとに分割するためのコード
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl.dataset) // dl.batch_size):
            #TODO: batchの中身を確認して、必要な前処理を行う
            data,input_length,targets,target_length,data_path,true_length=batch
            data=data.to(device)
            input_length=input_length.to(device)
            targets=targets.to(device)
            target_length=target_length.to(device)
            alignment=model.alignment(data,input_length.clone(),orig_len=true_length,min_clip=16,mode="midpoint",overlap=False)
            #Alignmentをもとに，動画を単語ごとに分割するコードを書く
            #Alignment:[[{'gloss': 'GLOSS1', 't_end': int,'t_start': int},...],...]
            c=0
            prev_dir=""
            prev_frames=0
            prev_data_path=""
            for i in range(len(data)):
                o_skeleton_file=f"{skeleton_dir}/{data_path[i]}.csv"
                o_skeleton_face_file=f"{skeleton_face_dir}/{data_path[i]}.csv"
                skeleton_data=np.loadtxt(o_skeleton_file, delimiter=",", skiprows=1)
                max_len = max(input_length).item()
                left_pad = 6
                right_pad=max_len-skeleton_data.shape[0]-left_pad
                #パディング
                skeleton_data = np.concatenate((np.tile(skeleton_data[0], (left_pad, 1)), skeleton_data, np.tile(skeleton_data[-1], (right_pad, 1))), axis=0)
                #補間
                skeleton_data=np.where(skeleton_data==0,np.nan,skeleton_data)
                skeleton_data=nan_interpolate(skeleton_data)
                skeleton_data=np.where(skeleton_data==0,np.nan,skeleton_data)
                skeleton_face_data=np.loadtxt(o_skeleton_face_file, delimiter=",", skiprows=1)
                skeleton_face_data=np.concatenate((np.tile(skeleton_face_data[0], (left_pad, 1)), skeleton_face_data, np.tile(skeleton_face_data[-1], (right_pad, 1))), axis=0)
                skeleton_face_data=np.where(skeleton_face_data==0,np.nan,skeleton_face_data)
                skeleton_face_data=nan_interpolate(skeleton_face_data,limit_area="both")
                skeleton_face_data=np.where(skeleton_face_data==0,np.nan,skeleton_face_data)
                video=data[i][:input_length[i]].cpu().numpy().transpose(0,2,3,1) #(T,C,H,W) -> (T,H,W,C)
                #skeleton_data=skeleton_data[:input_length[i]]
                alignment_i=alignment[i]
                #gloss_i=targets[i][:target_length[i]].cpu().numpy()
                for j in range(len(alignment_i)):
                    t_start=alignment_i[j]['t_start']
                    t_end=alignment_i[j]['t_end']
                    gloss=alignment_i[j]['gloss']
                    save_dir=f"{save_path_split}/{gloss2class[gloss]}_{gloss}"
                    os.makedirs(save_dir, exist_ok=True)
                    if skeleton:
                        save_skeleton_dir=f"{save_path_skeleton_split}/{gloss2class[gloss]}_{gloss}"
                        os.makedirs(save_skeleton_dir, exist_ok=True)
                        save_face_dir=f"{save_path_skeleton_face_split}/{gloss2class[gloss]}_{gloss}"
                        os.makedirs(save_face_dir, exist_ok=True)
                        #単語ごとに骨格データを保存
                        skeleton_clip=skeleton_data[t_start:t_end]
                        skeleton_face_clip=skeleton_face_data[t_start:t_end]

                    video_clip= video[t_start:t_end]
                    if skeleton and (len(skeleton_clip)!=len(video_clip) or len(skeleton_face_clip)!=len(video_clip)):
                        print(f"Warning: skeleton_clip length {len(skeleton_clip)} or skeleton_face_clip length {len(skeleton_face_clip)} does not match video_clip length {len(video_clip)} for gloss {gloss} in data_path {data_path[i]}. Skipping this clip.")
                        continue
                    video_clip = (video_clip+1)*127.5
                    #opencvで保存するためにuint8に変換
                    video_clip = video_clip.astype(np.uint8)
                    if prev_dir == save_dir and prev_data_path == data_path[i]:
                        #地続きの動画の場合は連番を続ける
                        out_path=f"{save_dir}/{data_path[i]}_{c-1}"
                        #連番画像で保存
                        for k in range(len(video_clip)):
                            cv2.imwrite(f"{out_path}/{prev_frames+k}.png", video_clip[k][:,:,::-1])
                        prev_frames+=len(video_clip)
                        save_face_dir = f"{save_path_skeleton_face_split}/{gloss2class[gloss]}_{gloss}"
                        os.makedirs(save_face_dir, exist_ok=True)
                        if skeleton:
                            # 地続きの動画の場合は連番を続ける
                            out_path = f"{save_skeleton_dir}/{data_path[i]}_{c - 1}.csv"
                            out_data = np.loadtxt(out_path, delimiter=",", skiprows=1)
                            out_face_path = f"{save_face_dir}/{data_path[i]}_{c - 1}.csv"
                            out_face_data = np.loadtxt(out_face_path, delimiter=",", skiprows=1)
                            if len(out_data.shape) == 1:
                                out_data = out_data[np.newaxis, :]
                            if len(out_face_data.shape) == 1:
                                out_face_data = out_face_data[np.newaxis, :]
                            if len(skeleton_clip.shape) == 1:
                                skeleton_clip = skeleton_clip[np.newaxis, :]
                            if len(skeleton_face_clip.shape) == 1:
                                skeleton_face_clip = skeleton_face_clip[np.newaxis, :]

                            if out_data.size == 0:
                                out_data = skeleton_clip
                                out_face_data = skeleton_face_clip
                            else:
                                try:
                                    out_data = np.concatenate((out_data, skeleton_clip), axis=0)
                                except ValueError:
                                    print(
                                        f"ValueError: out_data shape: {out_data.shape}, skeleton_clip shape: {skeleton_clip.shape}")
                                    print(f"out_data: {out_data}")
                                    print(f"skeleton_clip: {skeleton_clip}")
                                    raise
                                out_face_data = np.concatenate((out_face_data, skeleton_face_clip), axis=0)
                            np.savetxt(out_path, out_data, delimiter=",")
                            np.savetxt(out_face_path, out_face_data, delimiter=",")

                    else:
                        prev_dir = save_dir
                        #動画を保存
                        out_path=f"{save_dir}/{data_path[i]}_{c}"
                        os.makedirs(out_path, exist_ok=True)
                        #連番画像で保存
                        for k in range(len(video_clip)):
                            cv2.imwrite(f"{out_path}/{k}.png", video_clip[k][:,:,::-1])
                        out_path = f"{save_skeleton_dir}/{data_path[i]}_{c}.csv"
                        out_face_path = f"{save_face_dir}/{data_path[i]}_{c}.csv"
                        np.savetxt(out_path, skeleton_clip, delimiter=",")
                        np.savetxt(out_face_path, skeleton_face_clip, delimiter=",")
                        prev_frames=len(video_clip)
                        prev_data_path = data_path[i]
                        c+=1






                #np.save(f"{save_path}/{data_path[i]}_{j}_{gloss}.npy", video_clip)

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
    print("Config loaded.")
    #dataset="phoenix"
    #main(dataset,checkpoint=checkpoint,skeleton=True)
    dataset="phoenixT"
    main(dataset,checkpoint=checkpoint,skeleton=True)
    #dataset="CSL-Daily"
    #main(dataset,checkpoint=checkpoint,skeleton=True)
    # print("Process time: ", time.time() - start)
    #time.sleep(10)
    #print("全学習が終了しました．PCをシャットダウンします...")
    #os.system("shutdown -h now")
