import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os

from mpmath import limit
from networkx.algorithms.centrality import harmonic_centrality
import math
# List of connections between landmarks (0 to 20)
connections = [(0, 10), (10, 11), (10, 12), (11, 13), (12, 14)]
hand_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # Little finger
    (5, 9), (9, 13), (13, 17),
]

all_connections = connections +[(13,17)]+ [(i + 17, j + 17) for i, j in hand_connections] +[(14,38)]+ [(i + 38, j + 38) for i, j in
                                                                                   hand_connections]
#print(all_connections)
hand_points = [(i + 17, j + 17) for i, j in hand_connections] + [(i + 38, j + 38) for i, j in hand_connections]
body_points = [0, 10, 11, 12, 13, 14]
# righteyepoints=[33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,33]17
# lefteyepoints=[362,398,384,385,386,387,388,466,388,263,249,390,373,374,380,381,382]17
# mouthpoints=[0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61,185,40,39,37]20
# contourpoints=[127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,435,361,323,454,356]24
righteye_connections = [(33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133),
                        (133, 155), (155, 154), (154, 153), (153, 145), (145, 144), (144, 163), (163, 7), (7, 33),
                        (33, 246)]
lefteye_connections = [(362, 398), (398, 384), (384, 385), (385, 386), (386, 387), (387, 388), (388, 466), (466, 388),
                       (388, 263), (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382),
                       (382, 362)]
contour_connectrions = [(127, 234), (234, 93), (93, 132), (132, 58), (58, 172), (172, 136), (136, 150), (150, 149),
                        (149, 176), (176, 148), (148, 152), (152, 377), (377, 400), (400, 378), (378, 379), (379, 365),
                        (365, 397), (397, 288), (288, 435), (435, 361), (361, 323), (323, 454), (454, 356)]
mouth_connections = [(0, 267), (267, 269), (269, 270), (270, 409), (409, 291), (291, 375), (375, 321), (321, 405),
                     (405, 314), (314, 17), (17, 84), (84, 181), (181, 91), (91, 146), (146, 61), (61, 185), (185, 40),
                     (40, 39), (39, 37), (37, 0)]
face_connections = righteye_connections + lefteye_connections + mouth_connections + contour_connectrions

def sort_connections(connections):
    #all_connectionsは元データのインデックスを表すが，データからconnectionに対応するデータを取得すると，インデックスが変わる
    #all_connectionsのインデックスを変換後のデータに対応させる
    #all_connectionsにあるインデックスを重複無しで抽出
    unique_indices = []
    for conn in connections:
        for idx in conn:
            if idx not in unique_indices:
                unique_indices.append(idx)
    #print(unique_indices)
    index_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(unique_indices)}
    #print(index_mapping)
    #all_connectionsのインデックスを変換
    sorted_connections = []
    for conn in connections:
        sorted_conn = (index_mapping[conn[0]], index_mapping[conn[1]])
        sorted_connections.append(sorted_conn)
    return sorted_connections
def nan_interpolate(data,limit_area="inside"):
    """
    nanを補間する
    :param data:(frame,point)
    :return:補間後のdata
    """
    df = pd.DataFrame(data)
    # 内挿
    # df.interpolate(limit_direction="both", inplace=True)
    if limit_area=="inside":
        df.interpolate(limit_area="inside", inplace=True,)
    else:
        df.interpolate(inplace=True)
    # もしもnanがあれば0で埋める
    df.fillna(0, inplace=True)
    return np.array(df)


def nan_interpolate_zero(data):
    df = pd.DataFrame(data)
    df.fillna(0, inplace=True)
    return np.array(df)


def average_movint(data, window_size=5):
    """
    移動平均を取る
    :param data:
    :return:
    """
    data=np.where(data==0,np.nan,data)
    df = pd.DataFrame(data)
    df.rolling(window=window_size, center=True).mean()
    data=np.array(df)
    return np.where(np.isnan(data),0,data)

def coordinate_preprocess_face(data_face,is_face_connect=False):
    data_face = nan_interpolate(data_face, limit_area="both")
    data_face = average_movint(data_face)
    new_data_face = np.zeros((2, data_face.shape[0], data_face.shape[1] // 2))
    new_data_face[0] = data_face[:, 0::2]
    new_data_face[1] = data_face[:, 1::2]
    if is_face_connect:
        connection_indexes=connection_to_set(face_connections)
        new_data_face=new_data_face[:,:,connection_indexes]
    return new_data_face
def coordinate_preprocess(data, data_face,is_face_connect=False,is_sg_filter=False):
    data = nan_interpolate(data)
    #data=nan_interpolate_zero(data)
    if np.isnan(data).any():
        print("nan")
    data = average_movint(data)
    new_data = np.zeros((2, data.shape[0], data.shape[1] // 2))
    new_data[0] = data[:, 0::2]  # data:(2,frame,point)
    new_data[1] = data[:, 1::2]
    new_data = np.delete(new_data, [i for i in range(17, 23)], axis=2)
    # 10番目の点は11,12番目の中間点
    new_data[0, :, 10] = (new_data[0, :, 11] + new_data[0, :, 12]) / 2
    new_data[1, :, 10] = (new_data[1, :, 11] + new_data[1, :, 12]) / 2
    connection_indexes = connection_to_set(all_connections)
    hand_indexes = connection_to_set(hand_points)
    new_hand_data = new_data[:, :, hand_indexes]
    #new_hand_data=np.concatenate(fillna_left_right(new_hand_data),axis=2)
    new_hand_data=np.stack([nan_interpolate(new_hand_data[0]),nan_interpolate(new_hand_data[1])],axis=0)
    #new_hand_dataの(2,S,F)のうち，すべてが0のSのインデックスを取得
    zero_mask = np.all(new_hand_data == 0, axis=(0, 2))  # shape: (S,)
    nan_indexes = np.where(zero_mask)[0]
    new_body_data = new_data[:, :, body_points]
    new_data = new_data[:, :, connection_indexes]
    data_face = nan_interpolate(data_face, limit_area="both")
    data_face = average_movint(data_face)
    new_data_face = np.zeros((2, data_face.shape[0], data_face.shape[1] // 2))
    new_data_face[0] = data_face[:, 0::2]
    new_data_face[1] = data_face[:, 1::2]
    if is_face_connect:
        connection_indexes=connection_to_set(face_connections)
        new_data_face=new_data_face[:,:,connection_indexes]
    #new_data,new_data_face,new_hand_data,new_body_dataのうち，nan_indexesに対応するフレームをすべて削除する
    new_data = np.delete(new_data, nan_indexes, axis=1)
    new_data_face = np.delete(new_data_face, nan_indexes, axis=1)
    new_hand_data = np.delete(new_hand_data, nan_indexes, axis=1)
    new_body_data = np.delete(new_body_data, nan_indexes, axis=1)
    if is_sg_filter:
        new_data=apply_savgol_filter(new_data)
        new_data_face=apply_savgol_filter(new_data_face)
        new_hand_data=apply_savgol_filter(new_hand_data)
        new_body_data=apply_savgol_filter(new_body_data)
    return new_data, new_data_face, new_hand_data, new_body_data
def coordinate_preprocess_3d(data, data_face,is_face_connect=False,is_sg_filter=False):
    data = nan_interpolate(data)
    #data=nan_interpolate_zero(data)
    if np.isnan(data).any():
        print("nan")
    #data = average_movint(data)
    new_data = np.zeros((3, data.shape[0], data.shape[1] // 3))
    new_data[0] = data[:, 0::3]  # data:(2,frame,point)
    new_data[1] = data[:, 1::3]
    new_data[2]=-data[:, 2::3]
    new_data = np.delete(new_data, [i for i in range(17, 23)], axis=2)

    # 10番目の点は11,12番目の中間点
    new_data[0, :, 10] = (new_data[0, :, 11] + new_data[0, :, 12]) / 2
    new_data[1, :, 10] = (new_data[1, :, 11] + new_data[1, :, 12]) / 2
    new_data[2, :, 10] = (new_data[2, :, 11] + new_data[2, :, 12]) / 2
    connection_indexes = connection_to_set(all_connections)
    hand_indexes = connection_to_set(hand_points)
    new_hand_data = new_data[:, :, hand_indexes]
    #new_hand_data=np.concatenate(fillna_left_right(new_hand_data),axis=2)
    new_hand_data=np.stack([nan_interpolate(new_hand_data[0]),nan_interpolate(new_hand_data[1]),nan_interpolate(new_hand_data[2])],axis=0)
    #new_hand_dataの(2,S,F)のうち，すべてが0のSのインデックスを取得
    zero_mask = np.all(new_hand_data == 0, axis=(0, 2))  # shape: (S,)
    nan_indexes = np.where(zero_mask)[0]
    left_criterion=new_data[2, :,15]
    right_criterion=new_data[2, :,16]
    new_data[2,:,17:38]+=left_criterion[:, np.newaxis]
    new_data[2,:,38:]+=right_criterion[:, np.newaxis]
    #new_dataを正規化

    #new_hand_data[2, :, :21]+=left_criterion[:, np.newaxis]
    #new_hand_data[2, :, 21:]+=right_criterion[:, np.newaxis]
    new_body_data = new_data[:, :, body_points]
    new_data = new_data[:, :, connection_indexes]
    data_face = nan_interpolate(data_face, limit_area="both")
    #data_face = average_movint(data_face)
    new_data_face = np.zeros((3, data_face.shape[0], data_face.shape[1] // 3))
    new_data_face[0] = data_face[:, 0::3]
    new_data_face[1] = data_face[:, 1::3]
    new_data_face[2] = -data_face[:, 2::3]
    if is_face_connect:
        connection_indexes=connection_to_set(face_connections)
        new_data_face=new_data_face[:,:,connection_indexes]
    #new_data,new_data_face,new_hand_data,new_body_dataのうち，nan_indexesに対応するフレームをすべて削除する
    new_data = np.delete(new_data, nan_indexes, axis=1)
    new_data_face = np.delete(new_data_face, nan_indexes, axis=1)
    new_hand_data = np.delete(new_hand_data, nan_indexes, axis=1)
    new_body_data = np.delete(new_body_data, nan_indexes, axis=1)
    if is_sg_filter:
        new_data=apply_savgol_filter(new_data)
        new_data_face=apply_savgol_filter(new_data_face)
        new_hand_data=apply_savgol_filter(new_hand_data)
        new_body_data=apply_savgol_filter(new_body_data)
    return new_data, new_data_face, new_hand_data, new_body_data
def normalize_hand(data):
    data= np.where(data==0,np.nan,data)
    max_x=np.nanmax(data[0],axis=1)#(フレーム,)
    min_x=np.nanmin(data[0],axis=1)
    max_y=np.nanmax(data[1],axis=1)
    min_y=np.nanmin(data[1],axis=1)
    center_x=(max_x+min_x)/2
    center_y=(max_y+min_y)/2
    data[0]=data[0]-center_x[:, np.newaxis]
    data[1]=data[1]-center_y[:, np.newaxis]
    scale_width=np.sqrt((data[0,:,5]-data[0,:,17])**2+(data[1,:,5]-data[1,:,17])**2)
    scale_length=np.sqrt((data[0,:,0]-data[0,:,9])**2+(data[1,:,0]-data[1,:,9])**2)
    scale=(scale_width+scale_length)/2
    data[0]=data[0]/scale[:, np.newaxis]
    data[1]=data[1]/scale[:, np.newaxis]
    if data.shape[0]==3:
        data[2]=data[2]/scale[:, np.newaxis]
    data=np.where(np.isnan(data),0,data)
    center=np.array([center_x,center_y])#(2,フレーム)
    return data,center,scale
def normalize_mad(data,root,mad=None,is_3d=True):
    #一度0をnanにしてから，10番をrootにして，10番のz座標を0にする
    data= np.where(data==0,np.nan,data)
    center=data[:,:,root]#(2,frame)
    #root番のz座標を0にする
    data= data - center
    #x,yのみで 10との距離を計算する
    mad=np.nanmedian(np.sqrt(data[0]**2+data[1]**2),axis=1) if mad is None else mad#(frame,)
    #madを平滑化する
    mad=valid_convolve(mad,5)
    eps=1e-8
    data[0]= data[0]/(mad[:, np.newaxis]+eps)
    data[1]=data[1]/(mad[:,np.newaxis]+eps)
    if is_3d:
        data[2]=data[2]/(mad[:,np.newaxis]+eps)
    #nanを0にする
    data=np.where(np.isnan(data),0,data)
    return data,center
def valid_convolve(xx,size):
    b=np.ones(size)/size
    xx_mean=np.convolve(xx,b,mode="same")
    n_conv=math.ceil(size/2)
    xx_mean[0]*=size/n_conv
    for i in range(1,n_conv):
        xx_mean[i]*=size/(n_conv+i)
        xx_mean[-i]*=size/(i+n_conv-(size%2))
    return xx_mean


def get_savgol_coeffs(window_size=5, poly_order=2):
    """
    最小二乗法を用いてサビツキー・ゴレイ・フィルタの係数を算出する
    (A^T * A)^-1 * A^T の第1行目が平滑化用の係数となる
    """
    m = (window_size - 1) // 2
    # 窓内の相対的な位置 t = [-2, -1, 0, 1, 2] (window_size=5の場合)
    t = np.arange(-m, m + 1)

    # デザイン行列 A の作成 (A_ij = t^j)
    # 2次多項式なら各行が [1, t, t^2] となる行列
    A = np.vander(t, poly_order + 1, increasing=True)

    # 擬似逆行列 (A^T * A)^-1 * A^T を計算
    # その第1行目が、中心点 (t=0) における多項式の係数 a0 (平滑化値) に相当する
    coeffs = np.linalg.pinv(A)[0]
    return coeffs


def apply_savgol_filter(data, window_size=5, poly_order=2):
    """
    (N, T, J) のデータに対して T 軸(axis=1)方向にフィルタを適用する
    N: チャンネル数, T: 系列長, J: 次元数
    """
    N, T, J = data.shape
    coeffs = get_savgol_coeffs(window_size, poly_order)
    m = (window_size - 1) // 2

    # 出力用配列の初期化
    smoothed_data = np.zeros_like(data)

    # 各チャンネル(N)と各次元(J)に対して独立に適用
    for n in range(N):
        for j in range(J):
            series = data[n, :, j]

            # 端の処理：反射パディング（edge）を行うことで出力を系列長 T に合わせる
            padded_series = np.pad(series, (m, m), mode='edge')

            # 畳み込み演算（coeffsは対称なので反転不要だが、念のため反転して適用）
            smoothed_data[n, :, j] = np.convolve(padded_series, coeffs[::-1], mode='valid')

    return smoothed_data
def coordinate_preprocess_nointerpolate(data, data_face,is_face_connect=False):
    data = nan_interpolate(data)
    #data=nan_interpolate_zero(data)
    if np.isnan(data).any():
        print("nan")
    #data = average_movint(data)
    new_data = np.zeros((2, data.shape[0], data.shape[1] // 2))
    new_data[0] = data[:, 0::2]  # data:(2,frame,point)
    new_data[1] = data[:, 1::2]
    new_data = np.delete(new_data, [i for i in range(17, 23)], axis=2)
    # 10番目の点は11,12番目の中間点
    new_data[0, :, 10] = (new_data[0, :, 11] + new_data[0, :, 12]) / 2
    new_data[1, :, 10] = (new_data[1, :, 11] + new_data[1, :, 12]) / 2
    connection_indexes = connection_to_set(all_connections)
    hand_indexes = connection_to_set(hand_points)
    new_hand_data = new_data[:, :, hand_indexes]
    new_hand_data=np.concatenate(fillna_left_right(new_hand_data),axis=2)
    df_left = pd.DataFrame(new_hand_data[0])
    df_left.fillna(0, inplace=True)
    np_left=np.array(df_left)
    df_right = pd.DataFrame(new_hand_data[1])
    df_right.fillna(0, inplace=True)
    np_right=np.array(df_right)
    new_hand_data = np.stack([np_left,np_right],axis=0)
    #new_hand_data=np.stack([nan_interpolate(new_hand_data[0]),nan_interpolate(new_hand_data[1])],axis=0)
    new_body_data = new_data[:, :, body_points]
    new_data = new_data[:, :, connection_indexes]
    data_face = nan_interpolate(data_face, limit_area="both")
    data_face = average_movint(data_face)
    new_data_face = np.zeros((2, data_face.shape[0], data_face.shape[1] // 2))
    new_data_face[0] = data_face[:, 0::2]
    new_data_face[1] = data_face[:, 1::2]
    if is_face_connect:
        connection_indexes=connection_to_set(face_connections)
        new_data_face=new_data_face[:,:,connection_indexes]
    return new_data, new_data_face, new_hand_data, new_body_data

def connection_to_set(connections):
    connect_indexes = []
    for c in connections:
        connect_indexes.extend(list(c))
    return sorted(list(set(connect_indexes)))


def hand_patch_indexes(hand_data, patch_size=[8, 8], img_size=[224, 224]):
    """
    right_hand_data:(2,frame,point)
    left_hand_data:(2,frame,point)
    patch_size:(y,x)
    img_size:(y,x)
    """
    num_patches_y = img_size[0] // patch_size[0]
    num_patches_x = img_size[1] // patch_size[1]
    patch_indexes = []
    for i in range(hand_data.shape[1]):
        hand_data[0, i] = hand_data[0, i] * img_size[1]
        hand_data[1, i] = hand_data[1, i] * img_size[0]
        max_hand_x = int(np.max(hand_data[0, i, 21:]))
        max_hand_y = int(np.max(hand_data[1, i, 21:]))
        min_hand_x = int(np.min(hand_data[0, i, 21:]))
        min_hand_y = int(np.min(hand_data[1, i, 21:]))
        hand_width = max_hand_x - min_hand_x
        hand_height = max_hand_y - min_hand_y

        # left_patches
        left_patch_index_x = [int(max_hand_x / patch_size[1] + 0.5) + 1, int(min_hand_x / patch_size[1] + 0.5) - 1]
        if left_patch_index_x[0] > num_patches_x:
            left_patch_index_x[0] = num_patches_x
        if left_patch_index_x[1] < 0:
            left_patch_index_x[1] = 0

        left_patch_index_y = [int(max_hand_y / patch_size[0] + 0.5) + 1, int(min_hand_y / patch_size[0] + 0.5) - 1]
        if left_patch_index_y[0] > num_patches_y:
            left_patch_index_y[0] = num_patches_y
        if left_patch_index_y[1] < 0:
            left_patch_index_y[1] = 0
        # right_patches
        max_hand_x = int(np.max(hand_data[0, i, :21]))
        max_hand_y = int(np.max(hand_data[1, i, :21]))
        min_hand_x = int(np.min(hand_data[0, i, :21]))
        min_hand_y = int(np.min(hand_data[1, i, :21]))
        hand_width = max_hand_x - min_hand_x
        hand_height = max_hand_y - min_hand_y
        right_patch_index_x = [int(max_hand_x / patch_size[1] + 0.5) + 1, int(min_hand_x / patch_size[1] + 0.5) - 1]
        if right_patch_index_x[0] > num_patches_x:
            right_patch_index_x[0] = num_patches_x
        if right_patch_index_x[1] < 0:
            right_patch_index_x[1] = 0
        right_patch_index_y = [int(max_hand_y / patch_size[1] + 0.5) + 1, int(min_hand_y / patch_size[0] + 0.5) - 1]
        if right_patch_index_y[0] > num_patches_y:
            right_patch_index_y[0] = num_patches_y
        if right_patch_index_y[1] < 0:
            right_patch_index_y[1] = 0
        # patch_indexes
        right_patch_indexes = []
        for i in range(right_patch_index_y[1], right_patch_index_y[0]):
            for j in range(right_patch_index_x[1], right_patch_index_x[0]):
                right_patch_indexes.append(i * num_patches_x + j)
        left_patch_indexes = []
        for i in range(left_patch_index_y[1], left_patch_index_y[0]):
            for j in range(left_patch_index_x[1], left_patch_index_x[0]):
                left_patch_indexes.append(i * num_patches_x + j)

        patch_indexes.append(list(set(right_patch_indexes + left_patch_indexes)))
    return patch_indexes


def face_patch_indexes(face_data, patch_size=[16, 16], img_size=[224, 224]):
    num_patches_y = img_size[0] // patch_size[0]
    num_patches_x = img_size[1] // patch_size[1]
    patch_indexes = []
    for i in range(face_data.shape[1]):
        face_data[0, i] = face_data[0, i] * img_size[1]
        face_data[1, i] = face_data[1, i] * img_size[0]
        max_face_x = int(np.max(face_data[0, i]))
        max_face_y = int(np.max(face_data[1, i]))
        min_face_x = int(np.min(face_data[0, i]))
        min_face_y = int(np.min(face_data[1, i]))

        face_patch_x = [int((max_face_x / patch_size[1] + 0.5)) + 1, int(min_face_x / patch_size[0] + 0.5) - 1]
        if face_patch_x[0] > num_patches_x:
            face_patch_x[0] = num_patches_x
        if face_patch_x[1] < 0:
            face_patch_x[1] = 0
        face_patch_y = [int((max_face_y / patch_size[1] + 0.5)) + 1, int(min_face_y / patch_size[0]) - 1]
        if face_patch_y[0] > num_patches_y:
            face_patch_y[0] = num_patches_y
        if face_patch_y[1] < 0:
            face_patch_y[1] = 0

        face_patch_indexes = []
        for i in range(face_patch_y[1], face_patch_y[0]):
            for j in range(face_patch_x[1], face_patch_x[0]):
                face_patch_indexes.append(i * num_patches_x + j)
        patch_indexes.append(face_patch_indexes)
    return patch_indexes


def hand_face_boudingbox(hand_data, face_data, patch_size=(16, 16), img_size=(224, 224)):
    """
    手と顔のバウンディングボックスを取得
    :param hand_data: (2,T,F)
    :param face_data: (2,T,F)
    :return: lefthand_bounding(2,T,2),righthand_bounding,face_bounding(2,T,2)
    """
    lefthand_boundings = np.zeros((2, hand_data.shape[1], 2))
    righthand_boundings = np.zeros((2, hand_data.shape[1], 2))
    face_boundings = np.zeros((2, face_data.shape[1], 2))
    hand_data[0] = hand_data[0] * img_size[1]
    hand_data[1] = hand_data[1] * img_size[0]
    face_data[0] = face_data[0] * img_size[1]
    face_data[1] = face_data[1] * img_size[0]
    num_patches_y = img_size[0] // patch_size[0]
    num_patches_x = img_size[1] // patch_size[1]
    for i in range(hand_data.shape[1]):
        max_left_x = np.max(hand_data[0, i, :21])
        max_left_y = np.max(hand_data[1, i, :21])
        min_left_x = np.min(hand_data[0, i, :21])
        min_left_y = np.min(hand_data[1, i, :21])
        hand_width = max_left_x - min_left_x
        hand_height = max_left_y - min_left_y
        # left_patches
        if hand_width == 0:
            left_patch_index_x = [0, 0]
        else:
            left_patch_index_x = [int(max_left_x / patch_size[1] + 0.5) + 1, int(min_left_x / patch_size[1] + 0.5) - 1]
            if left_patch_index_x[0] > num_patches_x:
                left_patch_index_x[0] = num_patches_x
            elif left_patch_index_x[1] < 0:
                left_patch_index_x[1] = 0
            if left_patch_index_x[1] < 0:
                left_patch_index_x[1] = 0
            elif left_patch_index_x[1] > num_patches_x:
                left_patch_index_x[1] = num_patches_x
        if hand_height == 0:
            left_patch_index_y = [0, 0]
        else:
            left_patch_index_y = [int(max_left_y / patch_size[0] + 0.5) + 1, int(min_left_y / patch_size[0] + 0.5) - 1]
            if left_patch_index_y[0] > num_patches_y:
                left_patch_index_y[0] = num_patches_y
            elif left_patch_index_y[1] < 0:
                left_patch_index_y[1] = 0
            if left_patch_index_y[1] < 0:
                left_patch_index_y[1] = 0
            elif left_patch_index_y[1] > num_patches_y:
                left_patch_index_y[1] = num_patches_y

        lefthand_boundings[0, i] = [left_patch_index_x[1] * patch_size[1], left_patch_index_x[0] * patch_size[1]]
        lefthand_boundings[1, i] = [left_patch_index_y[1] * patch_size[0], left_patch_index_y[0] * patch_size[0]]

        max_right_x = np.max(hand_data[0, i, 21:])
        max_right_y = np.max(hand_data[1, i, 21:])
        min_right_x = np.min(hand_data[0, i, 21:])
        min_right_y = np.min(hand_data[1, i, 21:])
        hand_width = max_right_x - min_right_x
        hand_height = max_right_y - min_right_y
        if hand_width == 0:
            right_patch_index_x = [0, 0]
        else:
            right_patch_index_x = [int(max_right_x / patch_size[1] + 0.5) + 1, int(min_right_x / patch_size[1] + 0.5) - 1]
            if right_patch_index_x[0] > num_patches_x:
                right_patch_index_x[0] = num_patches_x
            elif right_patch_index_x[1] < 0:
                right_patch_index_x[1] = 0
            if right_patch_index_x[1] < 0:
                right_patch_index_x[1] = 0
            elif right_patch_index_x[1] > num_patches_x:
                right_patch_index_x[1] = num_patches_x
        if hand_height == 0:
            right_patch_index_y = [0, 0]
        else:
            right_patch_index_y = [int(max_right_y / patch_size[0] + 0.5) + 1, int(min_right_y / patch_size[0] + 0.5) - 1]
            if right_patch_index_y[0] > num_patches_y:
                right_patch_index_y[0] = num_patches_y
            elif right_patch_index_y[1] < 0:
                right_patch_index_y[1] = 0
            if right_patch_index_y[1] < 0:
                right_patch_index_y[1] = 0
            elif right_patch_index_y[1] > num_patches_y:
                right_patch_index_y[1] = num_patches_y

        righthand_boundings[0, i] = [right_patch_index_x[1] * patch_size[1], right_patch_index_x[0] * patch_size[1]]
        righthand_boundings[1, i] = [right_patch_index_y[1] * patch_size[0], right_patch_index_y[0] * patch_size[0]]

        max_face_x = np.nanmax(face_data[0, i])
        max_face_y = np.nanmax(face_data[1, i])
        min_face_x = np.nanmin(face_data[0, i])
        min_face_y = np.nanmin(face_data[1, i])
        face_width = max_face_x - min_face_x
        face_height = max_face_y - min_face_y
        if face_width == 0:
            face_patch_x = [0, 0]
        else:
            face_patch_x = [int((max_face_x / patch_size[1] + 0.5)) , int(min_face_x / patch_size[1] + 0.5) ]
            if face_patch_x[0] > num_patches_x:
                face_patch_x[0] = num_patches_x
            if face_patch_x[1] < 0:
                face_patch_x[1] = 0
            elif face_patch_x[1] > num_patches_x:
                face_patch_x[1] = num_patches_x
        if face_height == 0:
            face_patch_y = [0, 0]
        else:
            face_patch_y = [int((max_face_y / patch_size[0] + 0.5)) , int(min_face_y / patch_size[0]+0.5)]
            if face_patch_y[0] > num_patches_y:
                face_patch_y[0] = num_patches_y
            if face_patch_y[1] < 0:
                face_patch_y[1] = 0
            if face_patch_y[1] > num_patches_y:
                face_patch_y[1] = num_patches_y

        face_boundings[0, i] = [face_patch_x[1] * patch_size[1], face_patch_x[0] * patch_size[1]]
        face_boundings[1, i] = [face_patch_y[1] * patch_size[0], face_patch_y[0] * patch_size[0]]
    return lefthand_boundings, righthand_boundings, face_boundings

def unsharp_masking(img,kx,ky,sigx,sigy,k):
    """
    画像にアンシャープマスキングを適用
    :param img: (H,W,3)
    :param kx: x方向のカーネル
    :param ky: y方向のカーネル
    :param sigx: x方向の標準偏差
    :param sigy: y方向の標準偏差
    :param k: 強度
    :return: アンシャープマスキングを適用した画像
    """
    img_copy=img.copy()
    img_copy=cv2.GaussianBlur(img_copy,(kx,ky),sigx,sigy)
    diff_img=img-img_copy
    img_k=diff_img*k
    result=img+img_k
    return result
def clip_hand_face(img, left_bounding, right_bounding, face_bounding, clip_size=(32, 32)):
    left_bounding = left_bounding.astype(np.int32)
    right_bounding = right_bounding.astype(np.int32)
    face_bounding = face_bounding.astype(np.int32)
    img_face = np.zeros((img.shape[0], clip_size[0], clip_size[0], 3))  # (T,H,W,3)
    img_left = np.zeros((img.shape[0], clip_size[1], clip_size[1], 3))
    img_right = np.zeros((img.shape[0], clip_size[1], clip_size[1], 3))
    r_mask = []
    l_mask = []
    f_mask = []

    for i in range(img.shape[0]):
        if face_bounding[1, i, 0] == face_bounding[1, i, 1] or face_bounding[0, i, 0] == face_bounding[0, i, 1]:
            img_face[i] = np.zeros((clip_size[0], clip_size[0], 3))
            f_mask.append(0)
        else:
            img_face[i] = cv2.resize(
                img[i, face_bounding[1, i, 0]:face_bounding[1, i, 1], face_bounding[0, i, 0]:face_bounding[0, i, 1]],
                dsize=(clip_size[0], clip_size[0]))
            #img_face[i]=cv2.GaussianBlur(img_face[i],(3,3),0,0)
            #img_face[i]=unsharp_masking(img_face[i],3,3,0,0,1.5)
            f_mask.append(1)
        if left_bounding[1, i, 0] == left_bounding[1, i, 1] or left_bounding[0, i, 0] == left_bounding[0, i, 1]:
            img_left[i] = np.zeros((clip_size[1], clip_size[1], 3))
            l_mask.append(0)
        else:
            img_left[i] = cv2.resize(
                img[i, left_bounding[1, i, 0]:left_bounding[1, i, 1], left_bounding[0, i, 0]:left_bounding[0, i, 1]],
                dsize=(clip_size[1], clip_size[1]))
            #img_left[i]=cv2.GaussianBlur(img_left[i],(3,3),0,0)
            #img_left[i]=unsharp_masking(img_left[i],3,3,0,0,1.5)
            l_mask.append(1)
        if right_bounding[1, i, 0] == right_bounding[1, i, 1] or right_bounding[0, i, 0] == right_bounding[0, i, 1]:
            img_right[i] = np.zeros((clip_size[1], clip_size[1], 3))
            r_mask.append(0)
        else:
            img_right[i] = cv2.resize(img[i, right_bounding[1, i, 0]:right_bounding[1, i, 1],
                                      right_bounding[0, i, 0]:right_bounding[0, i, 1]], dsize=(clip_size[1], clip_size[1]))
            #img_right[i]=cv2.GaussianBlur(img_right[i],(3,3),0,0)
            #img_right[i]=unsharp_masking(img_right[i],3,3,0,0,1.0)

            r_mask.append(1)
    mask = np.stack([np.array(f_mask), np.array(l_mask), np.array(r_mask)], axis=1)
    return img_face, img_left, img_right, mask
def display_skeleton(img,hand_data, face_data, img_size=(224, 224)):
    for i in range(img.shape[0]):
        img_copy=img[i].copy()
        for j in range(hand_data.shape[2]):
            x=int(hand_data[0,i,j]*img_size[1])
            y=int(hand_data[1,i,j]*img_size[0])
            cv2.circle(img_copy,(x,y),2,(255,0,0),-1)
        for j in range(face_data.shape[2]):
            x=int(face_data[0,i,j]*img_size[1])
            y=int(face_data[1,i,j]*img_size[0])
            cv2.circle(img_copy,(x,y),2,(0,255,0),-1)
        cv2.imshow("skeleton",img_copy)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
def fillna_left_right(hand_data):
    #left_dataとright_dataの0をnanに変換
    left_data=hand_data[:,:,:21]
    right_data=hand_data[:,:,21:]
    left_data[left_data==0]=np.nan
    right_data[right_data==0]=np.nan

    df_left_x=pd.DataFrame(left_data[0])
    df_left_y=pd.DataFrame(left_data[1])
    df_right_x=pd.DataFrame(right_data[0])
    df_right_y=pd.DataFrame(right_data[1])
    #left_xとleft_xのnanを前方および後方補間
    df_left_x.fillna(method="ffill",inplace=True)
    df_left_x.fillna(method="bfill",inplace=True)
    df_right_x.fillna(method="ffill",inplace=True)
    df_right_x.fillna(method="bfill",inplace=True)
    #left_yとleft_yのnanを1に置換
    df_left_y.fillna(1,inplace=True)
    df_right_y.fillna(1,inplace=True)
    left_data[0]=np.array(df_left_x)
    left_data[1]=np.array(df_left_y)
    right_data[0]=np.array(df_right_x)
    right_data[1]=np.array(df_right_y)
    return left_data,right_data



def compute_center(body_data, hand_data, face_data):
    b_center_x = body_data[0, :, 1]  # (T,)
    b_center_y = body_data[1, :, 1]
    b_center = np.stack([b_center_x, b_center_y], axis=0)  # (2,T)

    left_hand_data=hand_data[:,:,:21]
    right_hand_data=hand_data[:,:,21:]
    left_center_x = np.mean(left_hand_data[0], axis=1)
    left_center_y = np.mean(left_hand_data[1], axis=1)
    left_center = np.stack([left_center_x, left_center_y], axis=0)-b_center
    right_center_x = np.mean(right_hand_data[0], axis=1)
    right_center_y = np.mean(right_hand_data[1], axis=1)
    right_center = np.stack([right_center_x, right_center_y], axis=0)-b_center
    face_center_x = np.mean(face_data[0], axis=1)
    face_center_y = np.mean(face_data[1], axis=1)
    face_center = np.stack([face_center_x, face_center_y], axis=0)-b_center  # (2,T)
    b_center=np.zeros_like(b_center)
    # (2,T,4)
    return np.stack([face_center,left_center, right_center,b_center], axis=2)  # (2,T,4)


if __name__ == "__main__":
    from loader import image2video
    import shutil

    num = 1

    file = {1: "11August_2011_Thursday_heute-3196", 2: "12December_2011_Monday_heute-5817",
            3: "15July_2010_Thursday_tagesschau_default-0", 4: "23May_2011_Monday_heute_default-9",
            5:"01February_2011_Tuesday_heute_default-6"}
    shutil.rmtree(f"/home/caffe/work/MAE_csr/example/{num}_face", ignore_errors=True)
    shutil.rmtree(f"/home/caffe/work/MAE_csr/example/{num}_left", ignore_errors=True)
    shutil.rmtree(f"/home/caffe/work/MAE_csr/example/{num}_right", ignore_errors=True)
    os.makedirs(f"/home/caffe/work/MAE_csr/example/{num}_face", exist_ok=True)
    os.makedirs(f"/home/caffe/work/MAE_csr/example/{num}_left", exist_ok=True)
    os.makedirs(f"/home/caffe/work/MAE_csr/example/{num}_right", exist_ok=True)
    #img = image2video(f"/home/caffe/work/MAE_csr/example/{file[num]}", img_size=(224, 224),ext="png") / 255
    data = np.loadtxt(f"/home/caffe/work/MAE_csr/example/cod/{file[num]}.csv", delimiter=",")
    data_face = np.loadtxt(f"/home/caffe/work/MAE_csr/example/face_cod/{file[num]}.csv", delimiter=",")
    data, face_data, hand_data, body_data = coordinate_preprocess(data, data_face)
    display_skeleton(img,hand_data, face_data)

    print(data.shape)
    img_face, img_left, img_right, _ = clip_hand_face(img,
                                                      *hand_face_boudingbox(hand_data, face_data, patch_size=(6, 6)),
                                                      clip_size=(64, 32))
    for i in range(img_face.shape[0]):

        plt.figure()
        plt.imshow(img_face[i])
        plt.savefig(f"/home/caffe/work/MAE_csr/example/{num}_face/{i}.png")
        plt.close()
        plt.figure()
        plt.imshow(img_left[i])
        plt.savefig(f"/home/caffe/work/MAE_csr/example/{num}_left/{i}.png")
        plt.close()
        plt.figure()
        plt.imshow(img_right[i])
        plt.savefig(f"/home/caffe/work/MAE_csr/example/{num}_right/{i}.png")
