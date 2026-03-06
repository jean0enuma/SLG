from timm.layers import set_layer_config
import cv2
from loader.coordinate_preprocess import *
import matplotlib.pyplot as plt
import numpy as np
import math


def clip_parts(data,data_face,clip_size=(64,64)):
    new_data, new_data_face, new_hand_data, new_body_data=coordinate_preprocess(data,data_face)
    #body_image=create_skeleton_image(new_body_data,clip_size,parts="body")
    #hand_image=create_skeleton_image(new_hand_data,clip_size,parts="right_hand")
    face_image=create_skeleton_image(new_data_face,clip_size,parts="face")
def init_path_indexes(connection):
    new_indexes=[]
    for c in connection:
        new_indexes.extend(list(c))
    new_indexes=sorted(list(set(new_indexes)))
    new_connection=[(new_indexes.index(i),new_indexes.index(j)) for i,j in connection]
    return new_connection,new_indexes


# ガウスフィルタの定義
def gaussian_filter(ksize, sigma):
    """
    式 exp(−∥x−x′∥^2 / 2σ^2) に基づいたガウスフィルターを作成します。

    Args:
        ksize (int): フィルタのサイズ（奇数）。
        sigma (float): ガウス関数の標準偏差。

    Returns:
        2次元ガウスカーネル
    """
    # カーネルの中心を計算（ksizeは奇数なので整数）
    center = ksize // 2

    # カーネルを初期化
    kernel = np.zeros((ksize, ksize), dtype=np.float32)

    # ガウス関数に基づいてカーネルを計算
    for i in range(ksize):
        for j in range(ksize):
            x = i - center
            y = j - center
            kernel[i, j] = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # 正規化を行わない場合はそのまま
    return kernel
def create_skeleton_image(data,clip_size,parts="body",ksize=3,sigma=0):
    if parts=="body":
        s_connection,_=init_path_indexes(connections)
    elif parts=="left_hand":
        data=data[:,:,0:21]

        s_connection,_=init_path_indexes(hand_connections)
    elif parts=="right_hand":
        data=data[:,:,21:]
        s_connection,_=init_path_indexes(hand_connections)
    elif parts=="face":
        s_connection=face_connections
    else:
        exit(-1)
    length=data.shape[1]
    # ガウスフィルターを作成
    #custom_gaussian_kernel = gaussian_filter(ksize, sigma)

    for l in range(length):
        image = np.zeros((clip_size[0], clip_size[1], 3))
        x_margin=(np.max(data[0,l])-np.min(data[0,l]))*0.1
        y_margin=(np.max(data[1,l])-np.min(data[1,l]))*0.1
        max=np.max(data[1,l])
        min=np.min(data[1,l])
        data[0,l] = (data[0,l] - np.min(data[0,l])+x_margin) / (np.max(data[0,l]) - np.min(data[0,l])+2*x_margin)
        data[1,l] = ((data[1,l] - np.min(data[1,l])+y_margin) / (np.max(data[1,l]) - np.min(data[1,l])+2*y_margin))
        for i in range(len(s_connection)):
            x1=int(data[0,l,s_connection[i][0]]*clip_size[0])
            x2=int(data[0,l,s_connection[i][1]]*clip_size[0])
            y1=int(data[1,l,s_connection[i][0]]*clip_size[1])
            y2=int(data[1,l,s_connection[i][1]]*clip_size[1])
            cv2.line(image,(x1,y1),(x2,y2),(255,255,255),thickness=1,lineType=cv2.LINE_AA)
            cv2.circle(image,(x1,y1),1,(255,255,255),thickness=-1,lineType=cv2.LINE_AA)
            cv2.circle(image,(x2,y2),1,(255,255,255),thickness=-1,lineType=cv2.LINE_AA)
        #image = cv2.filter2D(image, -1, custom_gaussian_kernel)
        image=cv2.GaussianBlur(image,(ksize,ksize),sigmaX=sigma,sigmaY=sigma)
        plt.imshow(image/255)
        plt.show()
    return image

if __name__ == '__main__':
    cod_data=np.loadtxt("/home/jean/work/MAE_csr/example/cod/01April_2010_Thursday_heute_default-0.csv",delimiter=",")
    face_data=np.loadtxt("/home/jean/work/MAE_csr/example/face_cod/01April_2010_Thursday_heute_default-0.csv",delimiter=",")

    clip_parts(cod_data,face_data)


