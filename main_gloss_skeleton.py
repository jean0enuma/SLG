import os,shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import (
    AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor,
    AutoModelForMultimodalLM,BitsAndBytesConfig,AutoModelForCausalLM
)


import torch
import os,shutil
import cv2,glob,json
import pandas as pd
from Parameter.Parameter import *
from loader.data_loader import *
import numpy as np
from scipy.interpolate import PchipInterpolator

def pchip_fill(coords: np.ndarray, max_gap: int | None = None) -> np.ndarray:
    """
    時間軸(T)に沿って欠損(NaN)を PCHIP 補間で埋める。

    Parameters
    ----------
    coords : np.ndarray, shape (T, 3, F)
        T フレーム, 3 座標 (x, y, z), F キーポイント。欠損は NaN。
    max_gap : int or None
        連続欠損がこのフレーム数を超える内部ギャップは補間せず NaN のまま残す。
        None なら内部ギャップは全て埋める。

    Returns
    -------
    np.ndarray, shape (T, 3, F)
        内部の NaN を埋めたコピー。先頭・末尾の欠損は外挿せず
        最近傍値でクランプ(ホールド)する。
    """
    T = coords.shape[0]
    out = coords.astype(np.float64, copy=True)
    t = np.arange(T)

    for c in range(coords.shape[1]):
        for k in range(coords.shape[2]):
            y = out[:, c, k]
            valid = ~np.isnan(y)
            n = int(valid.sum())

            if n == T:
                continue                 # 欠損なし
            if n == 0:
                continue                 # 補間の手がかりなし
            if n == 1:
                y[:] = y[valid][0]       # 1点のみ -> 定数ホールド
                continue

            tv, yv = t[valid], y[valid]

            # 有効区間内のみ補間。範囲外は NaN のまま返させる
            filled = PchipInterpolator(tv, yv, extrapolate=False)(t)

            # 先頭・末尾は外挿せず最近傍でホールド
            first, last = tv[0], tv[-1]
            filled[:first] = yv[0]
            filled[last + 1:] = yv[-1]

            # 長い内部ギャップは NaN のまま残す(任意)
            if max_gap is not None:
                idx = np.flatnonzero(np.isnan(y))   # 元の欠損位置
                if idx.size:
                    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                    for run in runs:
                        # 端のホールド区間は対象外。内部の長欠損のみ NaN へ戻す
                        if run[0] > first and run[-1] < last and run.size > max_gap:
                            filled[run] = np.nan

            out[:, c, k] = filled

    return out.astype(coords.dtype, copy=False)
from loader.coordinate_preprocess import *
def visualize_skeleton(cod_data,save_path,color=(0,255,0)):
    #cod_data shape: (frames,3,F)
    for i in range(cod_data.shape[0]):
        frame_data=cod_data[i]
        img=np.ones((512,512,3),dtype=np.uint8)*255
        for j in range(frame_data.shape[1]):
            x=int(frame_data[0][j]*512)
            y=int(frame_data[1][j]*512)
            cv2.circle(img,(x,y),5,color,-1)
        cv2.imwrite(f"{save_path}/{i}.png",img)
def sign_language_description(dataset,save_path):
    if dataset=="CSL-Daily":
        ext="jpg"
    else:
        ext="png"
    train_path, dev_path, test_path, gloss2class, class2gloss, video2gloss=islr_datasets_loader(dataset)
    train_c_path,dev_c_path,test_c_path, train_corpus, dev_corpus, test_corpus=datasets_loader(dataset)
    if dataset=="phoenixT":
        train_cod_root=f"{WORDS_DATADIR_T_SKELETON}/train"
        train_face_cod_root=f"{WORDS_DATADIR_T_SKELETON_FACE}/train"
        dev_cod_root=f"{WORDS_DATADIR_T_SKELETON}/dev"
        dev_face_cod_root=f"{WORDS_DATADIR_T_SKELETON_FACE}/dev"
        test_cod_root=f"{WORDS_DATADIR_T_SKELETON}/test"
        test_face_cod_root=f"{WORDS_DATADIR_T_SKELETON_FACE}/test"

        train_continuous_cod_root=f"{SKELETON_TRAIN_DATADIR_T_3D}"
        train_continuous_cod_face_root=f"{FACE_TRAIN_DATADIR_T_3D}"
        dev_continuous_cod_root=f"{SKELETON_DEV_DATADIR_T_3D}"
        dev_continuous_cod_face_root=f"{FACE_DEV_DATADIR_T_3D}"
        test_continuous_cod_root=f"{SKELETON_TEST_DATADIR_T_3D}"
        test_continuous_cod_face_root=f"{FACE_TEST_DATADIR_T_3D}"
    elif dataset=="CSL-Daily":
        train_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON}/train"
        train_face_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON_FACE}/train"
        dev_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON}/dev"
        dev_face_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON_FACE}/dev"
        test_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON}/test"
        test_face_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON_FACE}/test"

        train_continuous_cod_root=f"{SKELETON_CSL_DAILY_DATADIR_3D}"
        train_continuous_cod_face_root=f"{FACE_CSL_DAILY_DATADIR_3D}"
        dev_continuous_cod_root=f"{SKELETON_CSL_DAILY_DATADIR_3D}"
        dev_continuous_cod_face_root=f"{FACE_CSL_DAILY_DATADIR_3D}"
        test_continuous_cod_root=f"{SKELETON_CSL_DAILY_DATADIR_3D}"
        test_continuous_cod_face_root=f"{FACE_CSL_DAILY_DATADIR_3D}"
    else:
        raise ValueError("Invalid dataset name")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path,exist_ok=True)
    for video_path in train_c_path:
        print(f"Processing video: {video_path}")
        video_name=os.path.basename(video_path).split(".")[0]
        cod_path=f"{train_continuous_cod_root}/{video_name}.csv"
        face_cod_path=f"{train_continuous_cod_face_root}/{video_name}.csv"
        sequence = train_corpus[train_corpus["id"] == video_name]["annotation"].values[0]
        sequence=sequence.split(" ")
        gt_cod_data=np.loadtxt(cod_path,delimiter=",")
        gt_face_cod_data=np.loadtxt(face_cod_path,delimiter=",")
        gt_cod_data,gt_face_cod_data,gt_hand_cod_data,gt_body_cod_data=coordinate_preprocess_3d(gt_cod_data,gt_face_cod_data,is_face_connect=False,is_delete_nan=True)
        gt_cod_data=gt_cod_data.transpose(1,0,2)
        glosses_skeleton_list=[]
        gt_center_data = gt_cod_data[:, :, 1]  # (3,T)
        gt_shoulder_length = np.sqrt(
            (gt_cod_data[0, :, 2] - gt_cod_data[0, :, 3]) ** 2 + (gt_cod_data[1, :, 2] - gt_cod_data[1, :, 3]) ** 2 + (
                    gt_cod_data[2, :, 2] - gt_cod_data[2, :, 3]) ** 2)  # (T,)

        for i,gloss in enumerate(sequence):
            train_cod_path=glob.glob(f"{train_cod_root}/*_{gloss}/*.csv")
            if i==0:
                br=False
                while br==False:
                    select_cod_path=train_cod_path[random.randint(0,len(train_cod_path)-1)]
                    gloss_cod_data = np.loadtxt(select_cod_path, delimiter=",")
                    if len(gloss_cod_data.shape)> 1:
                        br=True
                    else:
                        print(f"Selected gloss {gloss} has more than one frame. Selecting again...")
                gloss_dir = os.path.basename(os.path.dirname(select_cod_path))
                face_data = np.loadtxt(f"{train_face_cod_root}/{gloss_dir}/{os.path.basename(select_cod_path)}",
                                       delimiter=",")
                cod_data, face_cod_data, hand_cod_data, body_cod_data = coordinate_preprocess_3d(gloss_cod_data,
                                                                                                 face_data,
                                                                                                 is_face_connect=False,
                                                                                                 is_delete_nan=False)
                center_data = cod_data[:, :, 1]
                shoulder_length = np.sqrt(
                    (cod_data[0, :, 2] - cod_data[0, :, 3]) ** 2 + (cod_data[1, :, 2] - cod_data[1, :, 3]) ** 2 + (
                                cod_data[2, :, 2] - cod_data[2, :, 3]) ** 2)
                cod_data = np.where(cod_data == 0, np.nan, cod_data)
                cod_data -= center_data[:, :, np.newaxis]
                cod_data /= shoulder_length[np.newaxis, :, np.newaxis]
                pad = np.ones((cod_data.shape[0], 5, cod_data.shape[-1])) * -100
                cod_data = np.concatenate([cod_data, pad], axis=1)
                glosses_skeleton_list.append(cod_data.transpose(1, 0, 2))
            else:
                prev_skeleton_data=glosses_skeleton_list[-1][:-5]
                left_wrist_cod=prev_skeleton_data[:,:,6]
                right_wrist_cod=prev_skeleton_data[:,:,27]
                left_hand_cod=prev_skeleton_data[:,:,6:27]
                left_hand_cod=prev_skeleton_data[:,:,6:27]
                right_hand_cod=prev_skeleton_data[:,:,27:48]
                #left_wrist_codのうちnanを除外する
                #left_wrist_cod:(T,3,1)
                left_wrist_cod=left_wrist_cod[~np.isnan(left_wrist_cod).any(axis=1)]
                left_hand_cod=left_hand_cod[~np.isnan(left_hand_cod).any(axis=(1,2))]
                if len(left_wrist_cod)==0:
                    left_wrist_cod=np.array([0,0,0])
                    left_hand_cod=np.zeros((3,21))
                else:
                    left_wrist_cod=left_wrist_cod[-1]
                    left_hand_cod=left_hand_cod[-1]
                right_wrist_cod=right_wrist_cod[~np.isnan(right_wrist_cod).any(axis=1)]
                right_hand_cod=right_hand_cod[~np.isnan(right_hand_cod).any(axis=(1,2))]
                if len(right_wrist_cod)==0:
                    right_wrist_cod=np.array([0,0,0])
                    right_hand_cod=np.zeros((3,21))
                else:
                    right_wrist_cod=right_wrist_cod[-1]
                    right_hand_cod=right_hand_cod[-1]
                #left_wrist_codとright_wrist_codの平均を計
                cod_data_list=[]
                left_cod_wrist_list=[]
                right_cod_wrist_list=[]
                left_cod_hand_list=[]
                right_cod_hand_list=[]
                for cod_path in train_cod_path:
                    gloss_cod_data=np.loadtxt(cod_path,delimiter=",")
                    gloss_dir = os.path.basename(os.path.dirname(cod_path))
                    face_data = np.loadtxt(f"{train_face_cod_root}/{gloss_dir}/{os.path.basename(cod_path)}",
                                           delimiter=",")

                    try:
                        if len(gloss_cod_data.shape) == 1:
                            gloss_cod_data = gloss_cod_data.reshape(1, -1)
                            face_data = face_data.reshape(1, -1)
                        cod_data, face_cod_data, hand_cod_data, body_cod_data = coordinate_preprocess_3d(gloss_cod_data,
                                                                                                         face_data,
                                                                                                         is_face_connect=False,
                                                                                                         is_delete_nan=True,
                                                                                                         is_limit_area=True)
                    except:
                        print(f"Error processing {cod_path}. Skipping...")
                        continue
                    center_data = cod_data[:, :, 1]
                    shoulder_length = np.sqrt(
                        (cod_data[0, :, 2] - cod_data[0, :, 3]) ** 2 + (cod_data[1, :, 2] - cod_data[1, :, 3]) ** 2 + (
                                cod_data[2, :, 2] - cod_data[2, :, 3]) ** 2)
                    cod_data = np.where(cod_data == 0, np.nan, cod_data)
                    cod_data -= center_data[:, :, np.newaxis]
                    cod_data /= shoulder_length[np.newaxis, :, np.newaxis]
                    cod_data=cod_data.transpose(1,0,2)
                    cod_data_list.append(cod_data)
                    cod_left_wrist=cod_data[:,:,6]
                    cod_left_hand=cod_data[:,:,6:27]
                    cod_left_wrist=cod_left_wrist[~np.isnan(cod_left_wrist).any(axis=1)]
                    cod_left_hand=cod_left_hand[~np.isnan(cod_left_hand).any(axis=(1,2))]
                    if len(cod_left_wrist)==0:
                        cod_left_wrist=np.array([[0,0,0]])
                        cod_left_hand=np.zeros((1,3,21))
                    left_cod_wrist_list.append(cod_left_wrist[0])
                    left_cod_hand_list.append(cod_left_hand[0])
                    cod_right_wrist=cod_data[:,:,27]
                    cod_right_hand=cod_data[:,:,27:48]
                    cod_right_wrist=cod_right_wrist[~np.isnan(cod_right_wrist).any(axis=1)]
                    cod_right_hand=cod_right_hand[~np.isnan(cod_right_hand).any(axis=(1,2))]
                    if len(cod_right_wrist)==0:
                        cod_right_wrist=np.array([[0,0,0]])
                        cod_right_hand=np.zeros((1,3,21))
                    right_cod_wrist_list.append(cod_right_wrist[0])
                    right_cod_hand_list.append(cod_right_hand[0])
                left_cod_wrist_array=np.stack(left_cod_wrist_list,axis=0)
                right_cod_wrist_array=np.stack(right_cod_wrist_list,axis=0)
                left_cod_hand_array=np.stack(left_cod_hand_list,axis=0)
                right_cod_hand_array=np.stack(right_cod_hand_list,axis=0)
                #left_wriist_codとleft_cod_wrist_arrayの距離,right_wriist_codとright_cod_wrist_arrayの距離を計算し、最小のものを選択する
                left_distance=np.linalg.norm(left_wrist_cod[np.newaxis]-left_cod_wrist_array,axis=1)
                right_distance=np.linalg.norm(right_wrist_cod-right_cod_wrist_array,axis=1)
                right_hand_distance=np.linalg.norm(right_hand_cod-right_cod_hand_array,axis=1).sum(-1)
                left_hand_distance=np.linalg.norm(left_hand_cod-left_cod_hand_array,axis=1).sum(-1)
                try:
                    distance=left_distance+right_distance
                    #distance=distance+left_hand_distance+right_hand_distance
                except Exception as e:
                    print(e)
                    print(f"Error calculating distance for gloss {gloss}. Skipping...")
                    continue
                min_index=np.argmin(distance,axis=0)
                try:
                    select_cod_data=cod_data_list[min_index]
                except Exception as e:
                    print(e)
                    print(f"Error selecting cod data for gloss {gloss}. Skipping...")
                    continue
                pad = np.ones((5,select_cod_data.shape[1] , select_cod_data.shape[-1])) * -100
                select_cod_data = np.concatenate([select_cod_data, pad], axis=0)
                glosses_skeleton_list.append(select_cod_data)


        glosses_skeleton_array=np.concatenate(glosses_skeleton_list,axis=0)[:-5]
        #0をnanに置き換える
        glosses_skeleton_array*=(np.max(gt_shoulder_length)+np.min(gt_shoulder_length))/2
        glosses_skeleton_array+=gt_center_data.mean(axis=0)[np.newaxis,:,np.newaxis]
        glosses_skeleton_array=np.where(np.isnan(glosses_skeleton_array),0,glosses_skeleton_array)
        glosses_skeleton_array=np.where(glosses_skeleton_array<-10,np.nan,glosses_skeleton_array)
        glosses_skeleton_array=pchip_fill(glosses_skeleton_array,max_gap=10)
        glosses_skeleton_array=np.where(np.isnan(glosses_skeleton_array),0,glosses_skeleton_array)
        gt_seq=" ".join(sequence)
        pred_save_path=f"{save_path}/{video_name}/pred_{gt_seq}"
        os.makedirs(pred_save_path,exist_ok=True)
        visualize_skeleton(glosses_skeleton_array,pred_save_path,color=(0,255,0))
        gt_save_path=f"{save_path}/{video_name}/gt_{gt_seq}"
        os.makedirs(gt_save_path, exist_ok=True)
        visualize_skeleton(gt_cod_data,gt_save_path,color=(255,0,0))

if __name__=="__main__":
    dataset="phoenixT"
    save_path=f"/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/glosses_sequence"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sign_language_description(dataset,save_path)