"""
複数の画像から動画を作成する
"""
import cv2
import os
import glob
import shutil
import numpy as np
def image2video(image_path,img_size=None,ext="png"):
    """
    複数の画像から動画のnumpy配列を作成する
    :param image_path: 画像が保存されているディレクトリ
    :return:動画のnumpy配列
    """
    #画像を読み込む
    images=sorted(glob.glob(f"{image_path}/*.{ext}"))
    #画像のサイズを取得
    img=cv2.imread(images[0])
    if img_size is not None:
        height,width,layers=(img_size[0],img_size[1],3)
    else:
        height,width,layers=img.shape
    #動画データを保存するためのnumpy配列
    video=np.empty((len(images),height,width,3),dtype=np.uint8)
    #videoに画像を書き込む
    for i,image in enumerate(images):
        tmp=cv2.imread(image)
        if img_size is not None:
            tmp=cv2.resize(tmp,(img_size[1],img_size[0]),interpolation=cv2.INTER_LANCZOS4)
        #画像を表示
        #cv2.imshow("image",tmp)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        video[i]=cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)
    return video
def npy2video(npy_path,img_size=None):
    """
    複数のnpyファイルから動画のnumpy配列を作成する
    :param npy_path: npyファイルが保存されているディレクトリ
    :return:動画のnumpy配列
    """
    #npyファイルを読み込む
    npys=sorted(glob.glob(f"{npy_path}/*.npy"))
    #画像のサイズを取得
    img=np.load(npys[0])
    if img_size is not None:
        height,width,layers=(img_size[0],img_size[1],3)
    else:
        height,width,layers=img.shape
    #動画データを保存するためのnumpy配列
    video=np.empty((len(npys),height,width,3),dtype=np.uint8)
    #videoに画像を書き込む
    for i,npy in enumerate(npys):
        tmp=np.load(npy)
        if img_size is not None:
            tmp=cv2.resize(tmp,(img_size[1],img_size[0]),interpolation=cv2.INTER_LANCZOS4)
        #画像を表示
        #cv2.imshow("image",tmp)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        video[i]=tmp
    return video
def resize_mask(npy_path, img_size=(256, 256)):
    """
    npyファイルを指定したサイズにリサイズする
    :param npy_path: npyファイルのパス
    :param img_size: リサイズ後の画像サイズ (height, width)
    :return: リサイズされたnumpy配列
    """
    data = np.load(npy_path).astype(np.uint8)*255
    resized_data = []
    for frame in data:
        resized_frame = cv2.resize(frame, (img_size[1], img_size[0]), interpolation=cv2.INTER_LANCZOS4)/255
        resized_data.append(resized_frame)
    return np.array(resized_data).astype(np.bool)
def video2npy(video_path,img_size=None):
    """
    動画ファイルをnpyファイルに変換する
    :param video_path: 動画ファイルのパス
    :param img_size: リサイズ後の画像サイズ (height, width)
    :return: numpy配列
    """
    #動画を読み込む
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if img_size is not None:
            frame = cv2.resize(frame, (img_size[1], img_size[0]), interpolation=cv2.INTER_LANCZOS4)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames)
if __name__ == '__main__':
    data = video2npy("/media/caffe/data_storage/How2Sign/train/front_clip/0tZfBzet80M_4-5-rgb_front.mp4")
    print(data.shape)  # (T, H, W, C)