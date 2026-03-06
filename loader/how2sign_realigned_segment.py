from Parameter.Parameter import *
import cv2
import numpy as np
import glob
import pandas as pd
from loader.data_loader import how2sign_load_corpus
import os
import shutil
def segment_video(data_path,start,end):
    """
    動画データを指定されたフレームでセグメント化する関数
    data_path:動画データのパス
    start:開始時間(s)
    end:終了時間(s)
    return:セグメント化された動画データ(np.array)(T,H,W,C)
    """
    cap = cv2.VideoCapture(data_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frame= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start * fps)
    end_frame = int(end * fps)
    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= start_frame and frame_idx < end_frame:
            frames.append(frame)
        frame_idx += 1
        if frame_idx >= end_frame:
            break
    cap.release()
    return np.array(frames),fps  # (T, H, W, C)
if __name__ == "__main__":
    train_data_paths=sorted(glob.glob(f"{HOW2SIGN_TRAIN_DATADIR}/raw_videos/*.mp4"))
    dev_data_paths=sorted(glob.glob(f"{HOW2SIGN_DEV_DATADIR}/raw_videos/*.mp4"))
    test_data_paths=sorted(glob.glob(f"{HOW2SIGN_TEST_DATADIR}/raw_videos/*.mp4"))

    train_corpus=how2sign_load_corpus(HOW2SIGN_TEXT_TRAIN_PATH)
    dev_corpus=how2sign_load_corpus(HOW2SIGN_TEXT_DEV_PATH)
    test_corpus=how2sign_load_corpus(HOW2SIGN_TEXT_TEST_PATH)

    train_dir=f"{HOW2SIGN_TRAIN_DATADIR}/realigned_videos"
    dev_dir=f"{HOW2SIGN_DEV_DATADIR}/realigned_videos"
    test_dir=f"{HOW2SIGN_TEST_DATADIR}/realigned_videos"
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.mkdir(train_dir)
    if os.path.exists(dev_dir):
        shutil.rmtree(dev_dir)
    os.mkdir(dev_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.mkdir(test_dir)
    all_path=[train_data_paths,dev_data_paths,test_data_paths]
    for data_paths in all_path:
        print("Number of videos:",len(data_paths))
        for data_path in data_paths:
            file_name=data_path.split("/")[-1].split(".mp4")[0]
            segments=train_corpus[train_corpus['VIDEO_NAME']==file_name]
            for i,(idx,row) in enumerate(segments.iterrows()):
                start=float(row['START_REALIGNED'])
                end=float(row['END_REALIGNED'])
                new_clip_name=f"{row['SENTENCE_NAME']}.mp4"
                segment_data,fps=segment_video(data_path,start,end)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(f"{train_dir}/{new_clip_name}", fourcc, fps, (segment_data.shape[2], segment_data.shape[1]))
                for frame in segment_data:
                    out.write(frame)
                out.release()
                print(f"Saved: {train_dir}/{new_clip_name}")
