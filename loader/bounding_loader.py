import numpy as np
import pandas as pd
import json
import cv2
import torch
from loader.coordinate_preprocess import *

def detect_outliers_bboxes(data, window_size=5, threshold_position=0.1, threshold_size=0.1):
    T, num_columns = data.shape
    result = np.copy(data)  # 結果を保存するためにコピーを作成

    # 各フレームに対して外れ値をチェック
    for i in range(window_size, T - window_size):
        # 近傍フレームの平均を計算
        neighbors_mean = np.nanmean(data[i - window_size:i + window_size + 1], axis=0)

        # 現在のフレームとの絶対差を計算
        diff = np.abs(data[i] - neighbors_mean)

        # 座標の変化とサイズの変化を別々に計算
        position_diff = np.fmax(diff[[0,2]], diff[[1,3]])  # x_min, x_max と y_min, y_max の変化を比較
        size_diff = np.abs((data[i, 1] - data[i, 0]) - (neighbors_mean[1] - neighbors_mean[0])) + \
                    np.abs((data[i, 3] - data[i, 2]) - (neighbors_mean[3] - neighbors_mean[2]))  # ボックスの幅と高さの変化

        # 変化が閾値を超えた場合にNaNに置き換える
        if np.any(position_diff > threshold_position) or size_diff > threshold_size:
            result[i] = np.nan

    return result
def add_margin(bbox_list, margin_rate=0.2):
    """
    バウンディングボックスにマージンを追加する
    :param bbox_list: バウンディングボックスのリスト
    :param margin: 追加するマージンの大きさ
    :return: マージンを追加したバウンディングボックスのリスト
    """
    bbox_list = np.copy(bbox_list)  # 結果を保存するためにコピーを作成
    margin=(bbox_list[:,[1,3]]-bbox_list[:,[0,2]])*margin_rate
    bbox_list[:, 0] -= margin[:,0]  # x_min
    bbox_list[:, 1] -= margin[:,1]  # x_max
    bbox_list[:, 2] += margin[:,0]  # y_min
    bbox_list[:, 3] += margin[:,1]  # y_max
    return bbox_list # x_max
def bbox_list2array(box_list):
    bbox_array=np.zeros((len(box_list),4))
    for i,t in enumerate(box_list):
        if t==None:
            bbox_array[i]=np.nan
        else:
            bbox_array[i]=np.array(t)
    return bbox_array
def judge_left_or_right(hand1_list,hand2_list,hand_data):
    T=hand_data.shape[1]
    left_cod=hand_data[:,:,0:21]
    right_cod=hand_data[:,:,21:]
    left_center_x=np.mean(left_cod[0],axis=1)
    right_center_x=np.mean(right_cod[0],axis=1)
    left_center_y=np.mean(left_cod[1],axis=1)
    right_center_y=np.mean(right_cod[1],axis=1)
    hand1_center_x=np.mean(hand1_list[:,0:2],axis=1)
    hand2_center_x=np.mean(hand2_list[:,0:2],axis=1)
    hand1_center_y=np.mean(hand1_list[:,2:],axis=1)
    hand2_center_y=np.mean(hand2_list[:,2:],axis=1)
    hand1_left_distance=np.sqrt((hand1_center_x-left_center_x)**2+(hand1_center_y-left_center_y)**2)
    hand1_right_distance=np.sqrt((hand1_center_x-right_center_x)**2+(hand1_center_y-right_center_y)**2)
    hand2_left_distance=np.sqrt((hand2_center_x-left_center_x)**2+(hand2_center_y-left_center_y)**2)
    hand2_right_distance=np.sqrt((hand2_center_x-right_center_x)**2+(hand2_center_y-right_center_y)**2)
    left_bbox_array=np.zeros((T,4))
    right_bbox_array=np.zeros((T,4))
    for t in range(T):
        print(hand1_left_distance[t],hand1_right_distance[t],hand2_left_distance[t],hand2_right_distance[t])
        if hand1_left_distance[t]<=hand1_right_distance[t] and hand2_left_distance[t]>=hand2_right_distance[t]:
            left_bbox_array[t]=hand1_list[t]
            right_bbox_array[t]=hand2_list[t]
        elif hand1_left_distance[t]>=hand1_right_distance[t] and hand2_left_distance[t]<=hand2_right_distance[t]:
            left_bbox_array[t]=hand2_list[t]
            right_bbox_array[t]=hand1_list[t]
        else:
            left_bbox_array[t]=hand1_list[t]
            right_bbox_array[t]=hand2_list[t]
    return left_bbox_array,right_bbox_array

def bbox_preprocess(bbox_dict,hand_data):
    """
    バウンディングボックスの情報を読み込む
    :param bbox_path: バウンディングボックスの情報が記述されたファイルのパス
    :return: バウンディングボックスの情報を格納した辞書
    """
    face_bbox = bbox_list2array(bbox_dict["face"])
    left_hand_bbox = bbox_list2array(bbox_dict["hand_2"])
    right_hand_bbox = bbox_list2array(bbox_dict["hand_1"])#(T, 4)
    # バウンディングボックスの情報を格納する辞書
    face_bbox = detect_outliers_bboxes(face_bbox)
    left_hand_bbox = detect_outliers_bboxes(left_hand_bbox)
    right_hand_bbox = detect_outliers_bboxes(right_hand_bbox)
    bbox_list=np.concatenate([face_bbox,left_hand_bbox,right_hand_bbox],axis=1)
    df=pd.DataFrame(bbox_list)
    df.interpolate(limit_area="inside",inplace=True)
    #もしもnanがあれば0で埋める
    df.fillna(0, inplace=True)
    bbox_array=df.values
    face_bbox=bbox_array[:,0:4]
    hand1_bbox=bbox_array[:,4:8]
    hand2_bbox=bbox_array[:,8:]
    left_hand_bbox,right_hand_bbox=judge_left_or_right(hand1_bbox,hand2_bbox,hand_data)
    left_hand_bbox=np.where(left_hand_bbox==0,1.0,left_hand_bbox)#左手は通例むかって右下にあるので、もしも0なら1に変換
    #face_bbox=add_margin(face_bbox)
    #left_hand_bbox=add_margin(left_hand_bbox)
    #right_hand_bbox=add_margin(right_hand_bbox)
    bbox_array=np.stack([face_bbox,left_hand_bbox,right_hand_bbox],axis=1)#(T, 3, 4)
    return bbox_array#(T, 3, 4)

def load_bbox(kind_dataset:str):
    with open(f"/home/jean/work/MAE_csr/bbox/bbox_phoenix14_{kind_dataset}.json") as f:
        bbox_dict = json.load(f)
    return bbox_dict

def clip_video_bbox(video,box_array,img_size=(256,256),clip_size=(64,64)):
    #video:torch.tensor((T, H, W, C))
    #box_array:np.array((T, 3, 4))
    T=box_array.shape[0]
    box_array[:,:,0:2]*=img_size[1]
    box_array[:,:,2:]*=img_size[0]
    #もしvideoがtorch.tensorならnumpyに変換
    if isinstance(video,torch.Tensor):
        video=video.numpy()
    face_clip=np.zeros((T,clip_size[0],clip_size[1],3))
    left_hand_clip=np.zeros((T,clip_size[0],clip_size[1],3))
    right_hand_clip=np.zeros((T,clip_size[0],clip_size[1],3))
    t_mask=np.ones((T,3))
    for t in range(T):
        face_bbox=box_array[t,0]#(4,)
        left_hand_bbox=box_array[t,1]
        right_hand_bbox=box_array[t,2]
        if int(face_bbox[0])==int(face_bbox[1]) or int(face_bbox[2])==int(face_bbox[3]):
            t_mask[t,0]=0
            face_img=np.zeros((clip_size[0],clip_size[1],3))
        else:
            face_img=video[t,int(face_bbox[2]):int(face_bbox[3]),int(face_bbox[0]):int(face_bbox[1])]
            face_img=cv2.resize(face_img,(clip_size[1],clip_size[0]),interpolation=cv2.INTER_LANCZOS4)

        if int(left_hand_bbox[0])==int(left_hand_bbox[1]) or int(left_hand_bbox[2])==int(left_hand_bbox[3]):
            t_mask[t,1]=0
            left_hand_img=np.zeros((clip_size[0],clip_size[1],3))
        else:
            left_hand_img=video[t,int(left_hand_bbox[2]):int(left_hand_bbox[3]),int(left_hand_bbox[0]):int(left_hand_bbox[1])]
            left_hand_img = cv2.resize(left_hand_img, (clip_size[1], clip_size[0]), interpolation=cv2.INTER_LANCZOS4)

        if int(right_hand_bbox[0])==int(right_hand_bbox[1]) or int(right_hand_bbox[2])==int(right_hand_bbox[3]):
            t_mask[t,2]=0
            right_hand_img=np.zeros((clip_size[0],clip_size[1],3))
        else:
            right_hand_img=video[t,int(right_hand_bbox[2]):int(right_hand_bbox[3]),int(right_hand_bbox[0]):int(right_hand_bbox[1])]
            right_hand_img=cv2.resize(right_hand_img,(clip_size[1],clip_size[0]),interpolation=cv2.INTER_LANCZOS4)
        face_clip[t]=face_img
        left_hand_clip[t]=left_hand_img
        right_hand_clip[t]=right_hand_img
    return face_clip,left_hand_clip,right_hand_clip,t_mask

if __name__=="__main__":
    from loader import image2video
    import shutil,os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    num=4

    file={1:"01April_2010_Thursday_heute_default-0",2:"19July_2009_Sunday_tagesschau_default-9",3:"15July_2010_Thursday_tagesschau_default-0",4:"23May_2011_Monday_heute_default-9",}
    shutil.rmtree(f"/home/jean/work/MAE_csr/example/{num}_bbox_face",ignore_errors=True)
    shutil.rmtree(f"/home/jean/work/MAE_csr/example/{num}_bbox_left",ignore_errors=True)
    shutil.rmtree(f"/home/jean/work/MAE_csr/example/{num}_bbox_right",ignore_errors=True)
    os.makedirs(f"/home/jean/work/MAE_csr/example/{num}_bbox_face",exist_ok=True)
    os.makedirs(f"/home/jean/work/MAE_csr/example/{num}_bbox_left",exist_ok=True)
    os.makedirs(f"/home/jean/work/MAE_csr/example/{num}_bbox_right",exist_ok=True)
    os.makedirs(f"/home/jean/work/MAE_csr/example/{num}_bbox_anchor",exist_ok=True)
    img=image2video(f"/home/jean/work/MAE_csr/example/{num}",img_size=(256,256))/255
    data = np.loadtxt(f"/home/jean/work/MAE_csr/example/cod/{file[num]}.csv", delimiter=",")
    data_face=np.loadtxt(f"/home/jean/work/MAE_csr/example/face_cod/{file[num]}.csv",delimiter=",")
    data,face_data,hand_data,body_data=coordinate_preprocess(data,data_face)
    bbox=load_bbox("train")[file[num]]
    bbox_array=bbox_preprocess(bbox,hand_data)
    face_clip,left_clip,right_clip,t_mask=clip_video_bbox(img,bbox_array)
    bbox_array[:,:,0:2]*=224
    bbox_array[:,:,2:]*=224#(T, 3, 4)
    for i in range(face_clip.shape[0]):
        plt.figure()
        plt.imshow(face_clip[i])
        plt.savefig(f"/home/jean/work/MAE_csr/example/{num}_bbox_face/{i}.png")
        plt.close()
        plt.figure()
        plt.imshow(left_clip[i])
        plt.savefig(f"/home/jean/work/MAE_csr/example/{num}_bbox_left/{i}.png")
        plt.close()
        plt.figure()
        plt.imshow(right_clip[i])
        plt.savefig(f"/home/jean/work/MAE_csr/example/{num}_bbox_right/{i}.png")
        plt.imshow(img[i])
        ax=plt.gca()
        rect=patches.Rectangle((bbox_array[i,0,0],bbox_array[i,0,2]),bbox_array[i,0,1]-bbox_array[i,0,0],bbox_array[i,0,3]-bbox_array[i,0,2],linewidth=1,edgecolor="g",facecolor="none")
        ax.add_patch(rect)
        rect=patches.Rectangle((bbox_array[i,1,0],bbox_array[i,1,2]),bbox_array[i,1,1]-bbox_array[i,1,0],bbox_array[i,1,3]-bbox_array[i,1,2],linewidth=1,edgecolor="r",facecolor="none")
        ax.add_patch(rect)
        rect=patches.Rectangle((bbox_array[i,2,0],bbox_array[i,2,2]),bbox_array[i,2,1]-bbox_array[i,2,0],bbox_array[i,2,3]-bbox_array[i,2,2],linewidth=1,edgecolor="r",facecolor="none")
        ax.add_patch(rect)
        plt.savefig(f"/home/jean/work/MAE_csr/example/{num}_bbox_anchor/{i}.png")





