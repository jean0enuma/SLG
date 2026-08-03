import os,shutil

from torchmetrics.functional.text import perplexity

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
import torch
import cv2
import torch

import cv2,glob,json
import pandas as pd
from Parameter.Parameter import *
from loader.data_loader import *
from loader.coordinate_preprocess import coordinate_preprocess_3d

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def create_video(img_files_path,save_path,file_name="temp.mp4",ext="png"):
    img_files=sorted(glob.glob(f"{img_files_path}/*.{ext}"))
    if len(img_files)==0:
        print("No images found in the specified path.")
        return
    img=cv2.imread(img_files[0])
    height,width,layers=img.shape
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    video=cv2.VideoWriter(f"{save_path}/{file_name}",fourcc,25,(width,height))
    for img_file in img_files:
        img=cv2.imread(img_file)
        video.write(img)
    video.release()

def recreate_video(video_file_path,save_path,file_name="temp.mp4",fps=1):
    #fpsを1にして動画を再作成する
    cap=cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_file_path}")
        return
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    video=cv2.VideoWriter(f"{save_path}/{file_name}",fourcc,fps,(width,height))
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        video.write(frame)
    cap.release()
    video.release()
class VJEPAExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
        self.model = AutoModel.from_pretrained(
            "facebook/vjepa2-vitl-fpc64-256",
            device_map="cuda:0",
            attn_implementation="sdpa"
        )

    def forward(self, video_path):
        # opencvでフレーム数を取得
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        vr = VideoDecoder(video_path)
        frame_idx = np.arange(0,
                              frame_count)  # choosing some frames. here, you can define more complex sampling strategy
        video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
        video = self.processor(video, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**video)
        B,C,F=outputs.last_hidden_state.shape
        outputs=outputs.last_hidden_state.reshape(B,frame_count//2,C//(frame_count//2),F)
        return outputs
def save_crop_video(video_path,save_path,x_min,x_max,y_min,y_max,file_name="crop_video.mp4"):
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    video=cv2.VideoWriter(f"{save_path}/{file_name}",fourcc,fps,(128,128))
    i=0
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        img_x_min=int(x_min[i]*width)
        img_x_max=int(x_max[i]*width)
        img_y_min=int(y_min[i]*height)
        img_y_max=int(y_max[i]*height)
        if img_y_min==img_y_max or img_x_min==img_x_max:
            #すべての座標が同じ場合は、すべての画素を0にする
            crop_frame=np.zeros((128,128,3),dtype=np.uint8)
        else:
            crop_frame=frame[img_y_min:img_y_max,img_x_min:img_x_max]
            crop_frame=cv2.resize(crop_frame,(128,128))
        video.write(crop_frame)
        i+=1
    cap.release()
    video.release()
def face_hand_crop(hand_cod_data,face_cod_data,data_path,save_path):
    left_hand_cod_data = hand_cod_data[:, :, :21]
    right_hand_cod_data = hand_cod_data[:, :, 21:]
    left_x_max = np.max(left_hand_cod_data[0, :, :], axis=-1)
    left_x_min = np.min(left_hand_cod_data[0, :, :], axis=-1)
    left_y_max = np.max(left_hand_cod_data[1, :, :], axis=-1)
    left_y_min = np.min(left_hand_cod_data[1, :, :], axis=-1)

    right_x_max = np.max(right_hand_cod_data[0, :, :], axis=-1)
    right_x_min = np.min(right_hand_cod_data[0, :, :], axis=-1)
    right_y_max = np.max(right_hand_cod_data[1, :, :], axis=-1)
    right_y_min = np.min(right_hand_cod_data[1, :, :], axis=-1)

    face_x_max = np.max(face_cod_data[0, :, :], axis=-1)
    face_x_min = np.min(face_cod_data[0, :, :], axis=-1)
    face_y_max = np.max(face_cod_data[1, :, :], axis=-1)
    face_y_min = np.min(face_cod_data[1, :, :], axis=-1)
    # 動画をクロップして保存する
    new_left_x_min = np.maximum(np.zeros_like(left_x_min), left_x_min - (left_x_max - left_x_min) * 0.1)
    new_left_x_max = np.minimum(np.ones_like(left_x_max), left_x_max + (left_x_max - left_x_min) * 0.1)
    new_left_y_min = np.maximum(np.zeros_like(left_y_min), left_y_min - (left_y_max - left_y_min) * 0.1)
    new_left_y_max = np.minimum(np.ones_like(left_y_max), left_y_max + (left_y_max - left_y_min) * 0.1)
    new_right_x_min = np.maximum(np.zeros_like(right_x_min), right_x_min - (right_x_max - right_x_min) * 0.1)
    new_right_x_max = np.minimum(np.ones_like(right_x_max), right_x_max + (right_x_max - right_x_min) * 0.1)
    new_right_y_min = np.maximum(np.zeros_like(right_y_min), right_y_min - (right_y_max - right_y_min) * 0.1)
    new_right_y_max = np.minimum(np.ones_like(right_y_max), right_y_max + (right_y_max - right_y_min) * 0.1)
    new_face_x_min = np.maximum(np.zeros_like(face_x_min), face_x_min - (face_x_max - face_x_min) * 0.1)
    new_face_x_max = np.minimum(np.ones_like(face_x_max), face_x_max + (face_x_max - face_x_min) * 0.1)
    new_face_y_min = np.maximum(np.zeros_like(face_y_min), face_y_min - (face_y_max - face_y_min) * 0.1)
    new_face_y_max = np.minimum(np.ones_like(face_y_max), face_y_max + (face_y_max - face_y_min) * 0.1)
    save_crop_video(data_path, save_path, x_min=new_left_x_min, x_max=new_left_x_max, y_min=new_left_y_min,
                    y_max=new_left_y_max, file_name=f"tmp_video_left.mp4")
    save_crop_video(data_path, save_path, x_min=new_right_x_min, x_max=new_right_x_max, y_min=new_right_y_min,
                    y_max=new_right_y_max, file_name=f"tmp_video_right.mp4")
    save_crop_video(data_path, save_path, x_min=new_face_x_min, x_max=new_face_x_max, y_min=new_face_y_min,
                    y_max=new_face_y_max, file_name=f"tmp_video_face.mp4")


def sign_video_feature_extraction(dataset,save_path):

    # opencvでフレーム数を取得
    model=VJEPAExtractor()
    if dataset == "CSL-Daily":
        ext = "jpg"
    else:
        ext = "png"
    train_path, dev_path, test_path, gloss2class, class2gloss, video2gloss = islr_datasets_loader(dataset)
    if dataset=="phoenixT":
        train_cod_root=f"{WORDS_DATADIR_T_SKELETON}/train"
        train_face_cod_root=f"{WORDS_DATADIR_T_SKELETON_FACE}/train"
        dev_cod_root=f"{WORDS_DATADIR_T_SKELETON}/dev"
        dev_face_cod_root=f"{WORDS_DATADIR_T_SKELETON_FACE}/dev"
        test_cod_root=f"{WORDS_DATADIR_T_SKELETON}/test"
        test_face_cod_root=f"{WORDS_DATADIR_T_SKELETON_FACE}/test"
    elif dataset=="CSL-Daily":
        train_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON}/train"
        train_face_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON_FACE}/train"
        dev_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON}/dev"
        dev_face_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON_FACE}/dev"
        test_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON}/test"
        test_face_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON_FACE}/test"
    elif dataset=="AUTSL":
        train_cod_root=SKELETON_AUTSL_TRAIN_DATADIR_3D
        train_face_cod_root=FACE_AUTSL_TRAIN_DATADIR_3D
        dev_cod_root=SKELETON_AUTSL_DEV_DATADIR_3D
        dev_face_cod_root=FACE_AUTSL_DEV_DATADIR_3D
        test_cod_root=SKELETON_AUTSL_TEST_DATADIR_3D
        test_face_cod_root=FACE_AUTSL_TEST_DATADIR_3D
    else:
        raise ValueError("Invalid dataset name")
    if not os.path.exists(save_path):
        #shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)
    # textデータはjsonl形式で保存する
    train_save_path = f"{save_path}/train_video_feature"
    dev_save_path = f"{save_path}/dev_video_feature"
    test_save_path = f"{save_path}/test_video_feature"

    for data_path in train_path:
        print(f"Processing {data_path}...")
        gloss_dir=os.path.basename(os.path.dirname(data_path))
        file_name=data_path.split("/")[-1].split(".mp4")[0]
        video_name=data_path.split("/")[-1]
        if dataset=="AUTSL":
            face_data_path = f"{train_face_cod_root}/{file_name}.csv"
            cod_data_path = f"{train_cod_root}/{file_name}.csv"
        else:
            face_data_path = f"{train_face_cod_root}/{gloss_dir}/{file_name}.csv"
            cod_data_path = f"{train_cod_root}/{gloss_dir}/{file_name}.csv"
        face_data=np.loadtxt(f"{face_data_path}",delimiter=",",dtype=np.float32)
        cod_data=np.loadtxt(f"{cod_data_path}",delimiter=",",dtype=np.float32)
        cod_data, face_cod_data, hand_cod_data, body_cod_data = coordinate_preprocess_3d(cod_data,
                                                                                         face_data,
                                                                                         is_face_connect=False,is_delete_nan=False)

        if not os.path.exists(f"{train_save_path}/{gloss_dir}"):
            os.makedirs(f"{train_save_path}/{gloss_dir}", exist_ok=True)
        if os.path.basename(data_path).split(".")[1] != "mp4":
            data_id = os.path.basename(data_path)
            img_data_path = sorted(glob.glob(f"{data_path}/*.{ext}"))
            create_video(img_data_path, save_path, file_name=f"tmp_video.mp4")
            video_path = f"{save_path}/tmp_video.mp4"
        else:
            data_id = os.path.basename(data_path).split(".")[0]
            video_path = data_path
        if os.path.exists(f"{train_save_path}/{gloss_dir}/{data_id}.npy") and os.path.exists(f"{train_save_path}/{gloss_dir}/{data_id}_left.npy") and os.path.exists(f"{train_save_path}/{gloss_dir}/{data_id}_right.npy") and os.path.exists(f"{train_save_path}/{gloss_dir}/{data_id}_face.npy"):
            print(f"Skipping {data_path} as features already exist.")
            continue
        face_hand_crop(hand_cod_data, face_cod_data, video_path, save_path)
        feature = model(video_path)
        left_feature=model(f"{save_path}/tmp_video_left.mp4")
        right_feature=model(f"{save_path}/tmp_video_right.mp4")
        face_feature=model(f"{save_path}/tmp_video_face.mp4")
        feature=feature.squeeze(0).cpu().numpy()  # (T, S, D)
        left_feature=left_feature.squeeze(0).cpu().numpy()
        right_feature=right_feature.squeeze(0).cpu().numpy()
        face_feature=face_feature.squeeze(0).cpu().numpy()
        print("feature shape:", feature.shape)

        np.save(f"{train_save_path}/{gloss_dir}/{data_id}.npy", feature)
        np.save(f"{train_save_path}/{gloss_dir}/{data_id}_left.npy", left_feature)
        np.save(f"{train_save_path}/{gloss_dir}/{data_id}_right.npy", right_feature)
        np.save(f"{train_save_path}/{gloss_dir}/{data_id}_face.npy", face_feature)
    if os.path.exists(f"{save_path}/tmp_video.mp4"):
        os.remove(f"{save_path}/tmp_video.mp4")
    if os.path.exists(f"{save_path}/tmp_video_left.mp4"):
        os.remove(f"{save_path}/tmp_video_left.mp4")
    if os.path.exists(f"{save_path}/tmp_video_right.mp4"):
        os.remove(f"{save_path}/tmp_video_right.mp4")
    if os.path.exists(f"{save_path}/tmp_video_face.mp4"):
        os.remove(f"{save_path}/tmp_video_face.mp4")


    for data_path in dev_path:
        gloss_dir=os.path.basename(os.path.dirname(data_path))
        gloss_dir = os.path.basename(os.path.dirname(data_path))
        file_name = data_path.split("/")[-1].split(".mp4")[0]
        video_name = data_path.split("/")[-1]
        if dataset=="AUTSL":
            face_data_path = f"{dev_face_cod_root}/{file_name}.csv"
            cod_data_path = f"{dev_cod_root}/{file_name}.csv"
        else:
            face_data_path = f"{dev_face_cod_root}/{gloss_dir}/{file_name}.csv"
            cod_data_path = f"{dev_cod_root}/{gloss_dir}/{file_name}.csv"
        face_data = np.loadtxt(f"{face_data_path}", delimiter=",", dtype=np.float32)
        cod_data = np.loadtxt(f"{cod_data_path}", delimiter=",", dtype=np.float32)
        cod_data, face_cod_data, hand_cod_data, body_cod_data = coordinate_preprocess_3d(cod_data,
                                                                                         face_data,
                                                                                         is_face_connect=True, )
        C, T, J = cod_data.shape
        # face,handのそれぞれの最大値-最小値の1.2倍をクロップサイズとしてバウンティングボックスを決める
        if not os.path.exists(f"{dev_save_path}/{gloss_dir}"):
            os.makedirs(f"{dev_save_path}/{gloss_dir}", exist_ok=True)
        if os.path.basename(data_path).split(".")[1] != "mp4":
            data_id = os.path.basename(data_path)
            img_data_path = sorted(glob.glob(f"{data_path}/*.{ext}"))
            create_video(img_data_path, save_path, file_name="tmp_video.mp4")
            video_path = f"{save_path}/tmp_video.mp4"
        else:
            data_id = os.path.basename(data_path).split(".")[0]
            video_path = data_path
        face_hand_crop(hand_cod_data, face_cod_data, video_path, save_path)
        feature = model(video_path)
        left_feature=model(f"{save_path}/tmp_video_left.mp4")
        right_feature=model(f"{save_path}/tmp_video_right.mp4")
        face_feature=model(f"{save_path}/tmp_video_face.mp4")
        feature = feature.squeeze(0).cpu().numpy()  # (T, S, D)
        left_feature=left_feature.squeeze(0).cpu().numpy()
        right_feature=right_feature.squeeze(0).cpu().numpy()
        face_feature=face_feature.squeeze(0).cpu().numpy()
        print("feature shape:", feature.shape)

        np.save(f"{dev_save_path}/{gloss_dir}/{data_id}.npy", feature)
        np.save(f"{dev_save_path}/{gloss_dir}/{data_id}_left.npy", left_feature)
        np.save(f"{dev_save_path}/{gloss_dir}/{data_id}_right.npy", right_feature)
        np.save(f"{dev_save_path}/{gloss_dir}/{data_id}_face.npy", face_feature)
    if os.path.exists(f"{save_path}/tmp_video.mp4"):
        os.remove(f"{save_path}/tmp_video.mp4")
    if os.path.exists(f"{save_path}/tmp_video.mp4"):
        os.remove(f"{save_path}/tmp_video.mp4")
    if os.path.exists(f"{save_path}/tmp_video_left.mp4"):
        os.remove(f"{save_path}/tmp_video_left.mp4")
    if os.path.exists(f"{save_path}/tmp_video_right.mp4"):
        os.remove(f"{save_path}/tmp_video_right.mp4")
    if os.path.exists(f"{save_path}/tmp_video_face.mp4"):
        os.remove(f"{save_path}/tmp_video_face.mp4")

    for data_path in test_path:
        gloss_dir = os.path.basename(os.path.dirname(data_path))
        file_name = data_path.split("/")[-1].split(".mp4")[0]
        video_name = data_path.split("/")[-1]
        if dataset=="AUTSL":
            face_data_path = f"{test_face_cod_root}/{file_name}.csv"
            cod_data_path = f"{test_cod_root}/{file_name}.csv"
        else:
            face_data_path = f"{test_face_cod_root}/{gloss_dir}/{file_name}.csv"
            cod_data_path = f"{test_cod_root}/{gloss_dir}/{file_name}.csv"
        face_data = np.loadtxt(f"{face_data_path}", delimiter=",", dtype=np.float32)
        cod_data = np.loadtxt(f"{cod_data_path}", delimiter=",", dtype=np.float32)
        cod_data, face_cod_data, hand_cod_data, body_cod_data = coordinate_preprocess_3d(cod_data,
                                                                                         face_data,
                                                                                         is_face_connect=True, )
        C, T, J = cod_data.shape
        # face,handのそれぞれの最大値-最小値の1.2倍をクロップサイズとしてバウンティングボックスを決める


        if not os.path.exists(f"{test_save_path}/{gloss_dir}"):
            os.makedirs(f"{test_save_path}/{gloss_dir}", exist_ok=True)
        if os.path.basename(data_path).split(".")[1] != "mp4":
            data_id = os.path.basename(data_path)
            img_data_path = sorted(glob.glob(f"{data_path}/*.{ext}"))
            create_video(img_data_path, save_path, file_name="tmp_video.mp4")
            video_path = f"{save_path}/tmp_video.mp4"
        else:
            data_id = os.path.basename(data_path).split(".")[0]
            video_path = data_path
        face_hand_crop(hand_cod_data, face_cod_data, video_path, save_path)
        feature = model(video_path)
        left_feature = model(f"{save_path}/tmp_video_left.mp4")
        right_feature = model(f"{save_path}/tmp_video_right.mp4")
        face_feature = model(f"{save_path}/tmp_video_face.mp4")
        feature = feature.squeeze(0).cpu().numpy()  # (T, S, D)
        left_feature = left_feature.squeeze(0).cpu().numpy()
        right_feature = right_feature.squeeze(0).cpu().numpy()
        face_feature = face_feature.squeeze(0).cpu().numpy()
        print("feature shape:", feature.shape)

        np.save(f"{test_save_path}/{gloss_dir}/{data_id}.npy", feature)
        np.save(f"{test_save_path}/{gloss_dir}/{data_id}_left.npy", left_feature)
        np.save(f"{test_save_path}/{gloss_dir}/{data_id}_right.npy", right_feature)
        np.save(f"{test_save_path}/{gloss_dir}/{data_id}_face.npy", face_feature)
    if os.path.exists(f"{save_path}/tmp_video.mp4"):
        os.remove(f"{save_path}/tmp_video.mp4")
    if os.path.exists(f"{save_path}/tmp_video.mp4"):
        os.remove(f"{save_path}/tmp_video.mp4")
    if os.path.exists(f"{save_path}/tmp_video_left.mp4"):
        os.remove(f"{save_path}/tmp_video_left.mp4")
    if os.path.exists(f"{save_path}/tmp_video_right.mp4"):
        os.remove(f"{save_path}/tmp_video_right.mp4")
    if os.path.exists(f"{save_path}/tmp_video_face.mp4"):
        os.remove(f"{save_path}/tmp_video_face.mp4")
def visualize_feature(dataset,save_path):
    # featureを可視化する
    # opencvでフレーム数を取得
    model = VJEPAExtractor()
    if dataset == "CSL-Daily":
        ext = "jpg"
    else:
        ext = "png"
    train_path, dev_path, test_path, gloss2class, class2gloss, video2gloss = islr_datasets_loader(dataset)
    # textデータはjsonl形式で保存する
    train_save_path = f"{save_path}/train_video_feature"
    dev_save_path = f"{save_path}/dev_video_feature"
    test_save_path = f"{save_path}/test_video_feature"
    gloss_list=[]
    feature_list=[]
    for data_path in train_path:
        gloss_dir = os.path.basename(os.path.dirname(data_path))
        gloss_list.append(gloss_dir)
        if os.path.basename(data_path).split(".")[1] != "mp4":
            data_id = os.path.basename(data_path)
            img_data_path = sorted(glob.glob(f"{data_path}/*.{ext}"))
            create_video(img_data_path, save_path, file_name=f"tmp_video.mp4")
            video_path = f"{save_path}/tmp_video.mp4"
        else:
            data_id = os.path.basename(data_path).split(".")[0]
            video_path = data_path
        feature = model(video_path)
        feature_list = feature.cpu().mean(dim=1)  # (T, S, D)
    feature_list=torch.cat(feature_list, dim=0).numpy()  # (N, D)
    label_set=set(gloss_list)
    #tsneで次元削減して可視化する
    feature_2d = TSNE(
        n_components=2,
        perplexity=min(30, len(feature_list) - 1),
        init='pca',
        learning_rate="auto",
        random_state=42
    ).fit_transform(feature_list)
    plt.figure(figsize=(10, 10))
    for label in label_set:
        indices = [i for i, gloss in enumerate(gloss_list) if gloss == label]
        plt.scatter(feature_2d[indices, 0], feature_2d[indices, 1], label=label)
    plt.legend()
    plt.title("t-SNE Visualization of Video Features")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
if __name__=="__main__":
    sign_video_feature_extractoion("AUTSL",save_path="/media/caffe/data_storage/AUTSL/video_feature")
