import numpy as np
import cv2
import glob
import pandas as pd
from Parameter.Parameter import *
from sklearn.model_selection import train_test_split
import os,copy
import random,time

def create_corpus(data_path):
    gloss2class = {}
    class2gloss = {}
    id = 0
    for path in data_path:
        key = path.split("/")[-2]
        if key not in gloss2class.keys():
            class2gloss[id] = key
            gloss2class[key] = id
            id += 1
    return gloss2class, class2gloss
def create_corpus_phoenix(data_path):
    gloss2class = {}
    class2gloss = {}
    id = 0
    for path in data_path:
        key = path.split("/")[-1]
        if key not in gloss2class.keys():
            class2gloss[id] = key
            gloss2class[key] = id
            id += 1
    return gloss2class, class2gloss
def create_corpus_gsl(train_iso, dev_iso, test_iso):
    iso_data = pd.concat([train_iso, dev_iso, test_iso],ignore_index=True)
    path2gloss = {row[0]: row[1] for row in iso_data.values}
    gloss_list=sorted(list(set([gloss for gloss in path2gloss.values()])))
    gloss2class={gloss: i for i, gloss in enumerate(gloss_list)}
    class2gloss = {i: gloss for gloss, i in gloss2class.items()}
    path2class = {path: gloss2class[gloss] for path, gloss in path2gloss.items()}
    return gloss2class, class2gloss, path2class

def create_video2class_from_datapath(data_path, gloss2class):
    """
    データパスからvideo2glossを作成する
    :param data_path: データパス
    :return: video2gloss
    """
    video2gloss = {}
    for path in data_path:
        key = path.split("/")[-1]
        gloss= path.split("/")[-2]
        video2gloss[key] = gloss2class[gloss]
    return video2gloss
def create_video2class_from_datapath_phoenix(data_path, gloss2class):
    """
    データパスからvideo2glossを作成する
    :param data_path: データパス
    :return: video2gloss
    """
    video2gloss = {}
    for path in data_path:
        key = path.split("/")[-2:]
        key = "/".join(key)
        gloss= path.split("/")[-2]
        video2gloss[key] = gloss2class[gloss]
    return video2gloss
def Data_loader(file_path:list,segment_list:dict):
    """
    :param data_path: ファイルパス,phoenixの手話文の種類を示す
    :param segment_list: フレームごとのラベリングを示す
    :return:
    """
    video_path=glob.glob(f"{file_path}/1/*.png")
    #video_pathから画像を読み込む
    video=[]
    classes=[]
    for i,v in enumerate(video_path):
        img=cv2.imread(v)
        video.append(img)
        if segment_list!=None:
            if v[v.find("feature"):] in segment_list.keys():
                classes.append(int(segment_list[v[v.find("feature"):]]))
            else:
                if classes!=[]:
                    classes.append(classes[-1])
                else:
                    classes.append(-1)

    #videoは(260,210,3)のndarrayがフレーム数分格納されているリスト，これを(フレーム数,260,210,3)のndarrayに変換
    video=np.array(video)
    classes=np.array(classes)
    if segment_list!=None:
        return video,classes
    else:
        return video
def how2sign_load_corpus(file_path:str):
    """
    :param file_path: ファイルパス
    :return:
    """
    # corpusの読み込み
    corpus = pd.read_csv(file_path, sep='\t', header=None)
    # VIDEO_NAMEを基準にソート
    #corpusの最初の1行目をcolumn名にする
    columns= corpus.iloc[0]
    corpus.columns = columns
    #corpusの最初の1行目を削除
    corpus = corpus[1:]
    corpus = corpus.sort_values("VIDEO_NAME")

    return corpus
def datasets_loader(dataset:str):
    #手話動画データセットの読み込み
    #phoenixは画像の連番となっている
    if dataset=="how2sign":
        train_path =  sorted(glob.glob(f"{HOW2SIGN_TRAIN_DATADIR}/front_clip/*"))
        #train_path=preprocess_from_video(train_path, threshold_min_frame=16, threshold_max_frame=1000000)
        dev_path = sorted(glob.glob(f"{HOW2SIGN_DEV_DATADIR}/front_clip/*"))
        #dev_path=preprocess_from_video(dev_path, threshold_min_frame=16, threshold_max_frame=1000000)
        test_path = sorted(glob.glob(f"{HOW2SIGN_TEST_DATADIR}/front_clip/*"))
        #test_path=preprocess_from_video(test_path, threshold_min_frame=16, threshold_max_frame=1000000)
        train_corpus = how2sign_load_corpus(HOW2SIGN_TEXT_TRAIN_PATH)
        dev_corpus = how2sign_load_corpus(HOW2SIGN_TEXT_DEV_PATH)
        test_corpus = how2sign_load_corpus(HOW2SIGN_TEXT_TEST_PATH)
    elif dataset=="phoenix+phoenixT":
        train_path = sorted(glob.glob(f"{TRAIN_RESIZED_DATADIR}/*/1"))+sorted(glob.glob(f"{TRAIN_DATADIR_T}/*"))
        dev_path = sorted(glob.glob(f"{DEV_RESIZED_DATADIR}/*/1")) + sorted(glob.glob(f"{DEV_DATADIR_T}/*"))
        test_path = sorted(glob.glob(f"{TEST_RESIZED_DATADIR}/*/1")) + sorted(glob.glob(f"{TEST_DATADIR_T}/*"))
        # 13April_2011_Wednesday_tagesschau_default-14だけnanがでる(削除検討)
        # おそらく，フレーム数とラベルの長さが合っていない．途中で終わっている可能性あり
        train_path.remove(f"{TRAIN_RESIZED_DATADIR}/13April_2011_Wednesday_tagesschau_default-14/1")
        train_corpus = pd.read_csv(TEXT_TRAIN_PATH, delimiter="|")
        dev_corpus = pd.read_csv(TEXT_DEV_PATH, delimiter="|")
        test_corpus = pd.read_csv(TEXT_TEST_PATH, delimiter="|")
        train_corpusT = pd.read_csv(TEXT_TRAIN_PATH_T, delimiter="|")
        dev_corpusT = pd.read_csv(TEXT_DEV_PATH_T, delimiter="|")
        test_corpusT = pd.read_csv(TEXT_TEST_PATH_T, delimiter="|")
        train_corpusT = train_corpusT.rename(columns={"orth": "annotation", "name": "id"})
        dev_corpusT = dev_corpusT.rename(columns={"orth": "annotation", "name": "id"})
        test_corpusT = test_corpusT.rename(columns={"orth": "annotation", "name": "id"})
        train_corpus = pd.concat([train_corpus, train_corpusT])
        dev_corpus = pd.concat([dev_corpus, dev_corpusT])
        test_corpus = pd.concat([test_corpus, test_corpusT])

    elif dataset=="phoenix":
        train_path = sorted(glob.glob(f"{TRAIN_DATADIR}/*/1"))
        dev_path = sorted(glob.glob(f"{DEV_DATADIR}/*/1"))
        test_path = sorted(glob.glob(f"{TEST_DATADIR}/*/1"))
        # 13April_2011_Wednesday_tagesschau_default-14だけnanがでる(削除検討)
        # おそらく，フレーム数とラベルの長さが合っていない．途中で終わっている可能性あり
        train_path.remove(f"{TRAIN_DATADIR}/13April_2011_Wednesday_tagesschau_default-14/1")
        train_corpus = pd.read_csv(TEXT_TRAIN_PATH, delimiter="|")
        dev_corpus = pd.read_csv(TEXT_DEV_PATH, delimiter="|")
        test_corpus = pd.read_csv(TEXT_TEST_PATH, delimiter="|")
    elif dataset=="phoenixT":
        train_path = sorted(glob.glob(f"{TRAIN_DATADIR_T}/*"))
        dev_path = sorted(glob.glob(f"{DEV_DATADIR_T}/*"))
        test_path = sorted(glob.glob(f"{TEST_DATADIR_T}/*"))
        train_corpusT = pd.read_csv(TEXT_TRAIN_PATH_T, delimiter="|")
        dev_corpusT = pd.read_csv(TEXT_DEV_PATH_T, delimiter="|")
        test_corpusT = pd.read_csv(TEXT_TEST_PATH_T, delimiter="|")
        train_corpus = train_corpusT.rename(columns={"orth": "annotation", "name": "id"})
        dev_corpus = dev_corpusT.rename(columns={"orth": "annotation", "name": "id"})
        test_corpus = test_corpusT.rename(columns={"orth": "annotation", "name": "id"})
    elif dataset=="CSL-Daily":
        data_path=sorted(glob.glob(f"{CSL_DAILY_DATADIR}/*"))
        split=pd.read_csv(f"{CSL_DAILY_LABELS}/split_1.txt",delimiter="|")
        train_path=sorted([path for path in data_path if path.split("/")[-1] in split[split["split"]=="train"]["name"].values])
        dev_path=sorted([path for path in data_path if path.split("/")[-1] in split[split["split"]=="dev"]["name"].values])
        test_path=sorted([path for path in data_path if path.split("/")[-1] in split[split["split"]=="test"]["name"].values])
        corpus= pd.read_csv(f"{CSL_DAILY_LABELS}/video_map.txt",delimiter="|")
        train_corpus=corpus.rename(columns={"gloss":"annotation","name":"id"})
        dev_corpus=train_corpus
        test_corpus=train_corpus
    elif dataset=="GSL":
        train_corpus = pd.read_csv(f"{GSL_CSR_TARGET_DATADIR}/gsl_train_unseen.txt", delimiter="|",header=None)
        dev_corpus = pd.read_csv(f"{GSL_CSR_TARGET_DATADIR}/gsl_dev_unseen.txt", delimiter="|",header=None)
        test_corpus = pd.read_csv(f"{GSL_CSR_TARGET_DATADIR}/gsl_test_unseen.txt", delimiter="|",header=None)
        train_path=[f"{GSL_CSR_DATADIR}/{row[0]}" for _, row in train_corpus.iterrows()]
        train_path=preprocess_from_frames(train_path, threshold_min_frame=16, threshold_max_frame=1000000,ext="jpg")
        dev_path=[f"{GSL_CSR_DATADIR}/{row[0]}" for _, row in dev_corpus.iterrows()]
        dev_path=preprocess_from_frames(dev_path, threshold_min_frame=16, threshold_max_frame=1000000,ext="jpg")
        test_path=[f"{GSL_CSR_DATADIR}/{row[0]}" for _, row in test_corpus.iterrows()]
        test_path=preprocess_from_frames(test_path, threshold_min_frame=16, threshold_max_frame=1000000,ext="jpg")
    elif dataset=="LSA-T":
        data_path=sorted(glob.glob(f"{LSA_T_DATADIR}/*.mp4"))
        remove_path=[]
        for i in range(258,292):
            remove_path.append(f"{LSA_T_DATADIR}/noticias-en-lengua-de-senas-argentina-resumen-semanal-semana-santa-04042021_{i}.mp4")
        remove_path.append(f"{LSA_T_DATADIR}/noticias-en-lengua-de-senas-argentina-resumen-semanal-18042021_120.mp4")
        data_path=[i for i in data_path if i not in remove_path]
        train_path, test_path = train_test_split(data_path, test_size=0.2, random_state=42)
        test_path, dev_path = train_test_split(test_path, test_size=0.5, random_state=42)
        train_corpus=None
        dev_corpus=None
        test_corpus=None
    elif dataset=="CSL-news":
        data_path=sorted(glob.glob(f"{CSL_NEWS_DATADIR}/*.mp4"))
        #data_path=preprocess_from_video(data_path, threshold_min_frame=16, threshold_max_frame=10000)
        train_path, test_path = train_test_split(data_path, test_size=0.2, random_state=42)
        test_path, dev_path = train_test_split(test_path, test_size=0.5, random_state=42)
        train_corpus=None
        dev_corpus=None
        test_corpus=None


    else:
        raise ValueError("dataset is not defined")
    return train_path, dev_path, test_path, train_corpus, dev_corpus, test_corpus
def datasets_loader_T(dataset:str):
    #手話動画データセットの読み込み
    #phoenixは画像の連番となっている
    if dataset=="how2sign":
        train_path =  sorted(glob.glob(f"{HOW2SIGN_TRAIN_DATADIR}/front_clip/*"))
        #train_path=preprocess_from_video(train_path, threshold_min_frame=16, threshold_max_frame=1000000)
        dev_path = sorted(glob.glob(f"{HOW2SIGN_DEV_DATADIR}/front_clip/*"))
        #dev_path=preprocess_from_video(dev_path, threshold_min_frame=16, threshold_max_frame=1000000)
        test_path = sorted(glob.glob(f"{HOW2SIGN_TEST_DATADIR}/front_clip/*"))
        #test_path=preprocess_from_video(test_path, threshold_min_frame=16, threshold_max_frame=1000000)
        train_corpus = how2sign_load_corpus(HOW2SIGN_TEXT_TRAIN_PATH_OLD).rename(columns={"SENTENCE": "annotation", "SENTENCE_NAME": "id"})
        dev_corpus = how2sign_load_corpus(HOW2SIGN_TEXT_DEV_PATH_OLD).rename(columns={"SENTENCE": "annotation", "SENTENCE_NAME": "id"})
        test_corpus = how2sign_load_corpus(HOW2SIGN_TEXT_TEST_PATH_OLD).rename(columns={"SENTENCE": "annotation", "SENTENCE_NAME": "id"})
        #corpusから，END_REALIGNED-START_REALIGNEDが16以上のものだけをpathから抜き取る
        #train_corpus["duration"]=train_corpus["END_REALIGNED"].astype(float)-train_corpus["START_REALIGNED"].astype(float)
        train_corpus["duration"]=train_corpus["END"].astype(float)-train_corpus["START"].astype(float)

        #dev_corpus["duration"]=dev_corpus["END_REALIGNED"].astype(float)-dev_corpus["START_REALIGNED"].astype(float)
        dev_corpus["duration"]=dev_corpus["END"].astype(float)-dev_corpus["START"].astype(float)

        #test_corpus["duration"]=test_corpus["END_REALIGNED"].astype(float)-test_corpus["START_REALIGNED"].astype(float)
        test_corpus["duration"]=test_corpus["END"].astype(float)-test_corpus["START"].astype(float)

        train_corpus=train_corpus[train_corpus["duration"]>=16/23]
        dev_corpus=dev_corpus[dev_corpus["duration"]>=16/23]
        test_corpus=test_corpus[test_corpus["duration"]>=16/23]
        train_path=[f"{HOW2SIGN_TRAIN_DATADIR}/front_clip/{row['id']}" for _, row in train_corpus.iterrows() if row['id'] in [path.split("/")[-1].split('.mp4')[0] for path in train_path]]
        dev_path=[f"{HOW2SIGN_DEV_DATADIR}/front_clip/{row['id']}" for _, row in dev_corpus.iterrows() if row['id'] in [path.split("/")[-1] for path in dev_path]]
        test_path=[f"{HOW2SIGN_TEST_DATADIR}/front_clip/{row['id']}" for _, row in test_corpus.iterrows() if row['id'] in [path.split("/")[-1] for path in test_path]]
    elif dataset=="phoenixT":
        train_path = sorted(glob.glob(f"{TRAIN_DATADIR_T}/*"))
        dev_path = sorted(glob.glob(f"{DEV_DATADIR_T}/*"))
        test_path = sorted(glob.glob(f"{TEST_DATADIR_T}/*"))
        train_corpusT = pd.read_csv(TEXT_TRAIN_PATH_T, delimiter="|")
        dev_corpusT = pd.read_csv(TEXT_DEV_PATH_T, delimiter="|")
        test_corpusT = pd.read_csv(TEXT_TEST_PATH_T, delimiter="|")
        train_corpus = train_corpusT.rename(columns={"translation": "annotation", "name": "id"})
        dev_corpus = dev_corpusT.rename(columns={"translation": "annotation", "name": "id"})
        test_corpus = test_corpusT.rename(columns={"translation": "annotation", "name": "id"})
    elif dataset=="CSL-Daily":
        data_path=sorted(glob.glob(f"{CSL_DAILY_DATADIR}/*"))
        split=pd.read_csv(f"{CSL_DAILY_LABELS}/split_1.txt",delimiter="|")
        train_path=sorted([path for path in data_path if path.split("/")[-1] in split[split["split"]=="train"]["name"].values])
        dev_path=sorted([path for path in data_path if path.split("/")[-1] in split[split["split"]=="dev"]["name"].values])
        test_path=sorted([path for path in data_path if path.split("/")[-1] in split[split["split"]=="test"]["name"].values])
        corpus= pd.read_csv(f"{CSL_DAILY_LABELS}/video_map.txt",delimiter="|")
        train_corpus=corpus.rename(columns={"char":"annotation","name":"id"})
        dev_corpus=train_corpus
        test_corpus=train_corpus
    elif dataset=="LSA-T":
        data_path=sorted(glob.glob(f"{LSA_T_DATADIR}/*.mp4"))
        remove_path=[]
        for i in range(258,292):
            remove_path.append(f"{LSA_T_DATADIR}/noticias-en-lengua-de-senas-argentina-resumen-semanal-semana-santa-04042021_{i}.mp4")
        remove_path.append(f"{LSA_T_DATADIR}/noticias-en-lengua-de-senas-argentina-resumen-semanal-18042021_120.mp4")
        data_path=[i for i in data_path if i not in remove_path]
        train_path, test_path = train_test_split(data_path, test_size=0.2, random_state=42)
        test_path, dev_path = train_test_split(test_path, test_size=0.5, random_state=42)
        corpus=pd.read_csv(LSA_T_METADIR,delimiter="|")
        train_corpus=corpus.rename(columns={"label":"annotation"})
        dev_corpus=train_corpus
        test_corpus=train_corpus
    elif dataset=="CSL-news":
        data_path=sorted(glob.glob(f"{CSL_NEWS_DATADIR}/*.mp4"))
        #data_path=preprocess_from_video(data_path, threshold_min_frame=16, threshold_max_frame=10000)
        train_path, test_path = train_test_split(data_path, test_size=0.2, random_state=42)
        test_path, dev_path = train_test_split(test_path, test_size=0.5, random_state=42)
        train_corpus=None
        dev_corpus=None
        test_corpus=None

    elif dataset=="phoenix":
        train_path = sorted(glob.glob(f"{TRAIN_DATADIR}/*/1"))
        dev_path = sorted(glob.glob(f"{DEV_DATADIR}/*/1"))
        test_path = sorted(glob.glob(f"{TEST_DATADIR}/*/1"))
        # 13April_2011_Wednesday_tagesschau_default-14だけnanがでる(削除検討)
        # おそらく，フレーム数とラベルの長さが合っていない．途中で終わっている可能性あり
        train_path.remove(f"{TRAIN_DATADIR}/13April_2011_Wednesday_tagesschau_default-14/1")
        train_corpus = pd.read_csv(TEXT_TRAIN_PATH, delimiter="|")
        dev_corpus = pd.read_csv(TEXT_DEV_PATH, delimiter="|")
        test_corpus = pd.read_csv(TEXT_TEST_PATH, delimiter="|")
    else:
        raise ValueError("dataset is not defined")
    return train_path, dev_path, test_path, train_corpus, dev_corpus, test_corpus
def optical_datasets_loader(dataset:str):
    """
    optical flowのデータセットの読み込み
    :param dataset: データセット名
    :return: train_path, dev_path, test_path, train_corpus, dev_corpus, test_corpus
    """
    if dataset=="phoenix":
        train_path = TRAIN_DATADIR_OPTICAL
        dev_path = DEV_DATADIR_OPTICAL
        test_path = TEST_DATADIR_OPTICAL
    elif dataset=="phoenixT":
        train_path = TRAIN_DATADIR_OPTICAL_T
        dev_path = DEV_DATADIR_OPTICAL_T
        test_path = TEST_DATADIR_OPTICAL_T
    else:
        raise ValueError("dataset is not defined")
    return train_path, dev_path, test_path
def mask_datasets_loader(dataset:str):
    """
    segmentationのデータセットの読み込み
    :param dataset: データセット名
    :return: train_path, dev_path, test_path, train_corpus, dev_corpus, test_corpus
    """
    if dataset=="phoenix":
        train_path =TRAIN_BINARY_DATADIR
        dev_path = DEV_BINARY_DATADIR
        test_path = TEST_BINARY_DATADIR
    elif dataset=="phoenixT":
        train_path = TRAIN_BINARY_DATADIR_T
        dev_path = DEV_BINARY_DATADIR_T
        test_path =TEST_BINARY_DATADIR_T
    else:
        raise ValueError("dataset is not defined")
    return train_path, dev_path, test_path
def preprocess_from_video(video_path, threshold_min_frame=8, threshold_max_frame=96):
    """
    動画のフレームを読み込み，前処理を行う
    :param video_path: 動画のパス
    :return: 前処理後の動画のフレーム
    """
    new_video_path=[]
    for v_path in video_path:
        cap= cv2.VideoCapture(v_path)
        total_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frame < threshold_min_frame or total_frame > threshold_max_frame:
            continue
        new_video_path.append(v_path)
    if len(new_video_path)==0:
        raise ValueError("No video found after preprocessing")
    return new_video_path
def preprocess_from_frames(video_path, threshold_min_frame=8, threshold_max_frame=96,ext="png"):
    """
    動画のフレームを読み込み，前処理を行う
    :param video_path: 動画のパス
    :return: 前処理後の動画のフレーム
    """
    new_video_path=[]
    for v_path in video_path:
        total_frame=len(glob.glob(f"{v_path}/*.{ext}"))
        if total_frame < threshold_min_frame or total_frame > threshold_max_frame:
            continue
        new_video_path.append(v_path)
    if len(new_video_path)==0:
        raise ValueError("No video found after preprocessing")
    return new_video_path
def islr_datasets_loader(dataset:str):
    if dataset=="AUTSL":
        train_path = sorted(glob.glob(f"{AUTSL_TRAIN_DATADIR}/*/*.mp4"))
        train_path=preprocess_from_video(train_path)
        dev_path = sorted(glob.glob(f"{AUTSL_DEV_DATADIR}/*/*.mp4"))
        dev_path=preprocess_from_video(dev_path)
        test_path = sorted(glob.glob(f"{AUTSL_TEST_DATADIR}/*/*.mp4"))
        test_path=preprocess_from_video(test_path)
        data_path=train_path + dev_path + test_path

        gloss2class,class2gloss= create_corpus(data_path)
        video2class= create_video2class_from_datapath(data_path, gloss2class)
    elif dataset=="WLASL":
        data_path = sorted(glob.glob(f"{WLASL2000_DATADIR}/*/*.mp4"))
        gloss2class, class2gloss = create_corpus(data_path)
        train_path, test_path = train_test_split(data_path, test_size=0.2, random_state=42)
        #test_path, dev_path = train_test_split(test_path, test_size=0.5, random_state=42)
        video2class = create_video2class_from_datapath(data_path, gloss2class)
    elif dataset=="ASL_Citizen":
        #data_path= sorted(glob.glob(f"{ASL_CITIZEN_DATADIR}/*.mp4"))
        #data_path=preprocess_from_frames(data_path)
        train_corpus= pd.read_csv(f"{ASL_CITIZEN_TEXT_DATADIR}/train.csv")
        dev_corpus= pd.read_csv(f"{ASL_CITIZEN_TEXT_DATADIR}/val.csv")
        test_corpus= pd.read_csv(f"{ASL_CITIZEN_TEXT_DATADIR}/test.csv")
        #カラム名"Video file"と"Gloss"の列を辞書に変換
        video2gloss_train = {row["Video file"]: row["Gloss"] for _, row in train_corpus.iterrows()}
        video2gloss_dev = {row["Video file"]: row["Gloss"] for _, row in dev_corpus.iterrows()}
        video2gloss_test = {row["Video file"]: row["Gloss"] for _, row in test_corpus.iterrows()}
        video2gloss= {**video2gloss_train, **video2gloss_dev, **video2gloss_test}
        gloss_list=[]
        gloss2class= {}
        #もしglossが同じならば同じクラス番号を割り当てる．もしglossが異なるならば異なるクラス番号(最も大きい値で更新)を割り当てる
        for video, gloss in video2gloss_train.items():
            if gloss not in gloss2class:
                gloss2class[gloss] = len(gloss2class)
        for video, gloss in video2gloss_dev.items():
            if gloss not in gloss2class:
                gloss2class[gloss] = len(gloss2class)
        for video, gloss in video2gloss_test.items():
            if gloss not in gloss2class:
                gloss2class[gloss] = len(gloss2class)
        class2gloss= {v: k for k, v in gloss2class.items()}
        video2class={k: gloss2class[v] for k, v in video2gloss.items()}
        train_path= sorted([f"{ASL_CITIZEN_DATADIR}/{video}" for video in video2gloss_train.keys()])
        train_path= preprocess_from_video(train_path)
        test_path= sorted([f"{ASL_CITIZEN_DATADIR}/{video}" for video in video2gloss_dev.keys()])
        test_path= preprocess_from_video(test_path)
        #test_path= sorted([f"{ASL_CITIZEN_DATADIR}/{video}" for video in video2gloss_test.keys()])
        #test_path= preprocess_from_frames(test_path)

    elif dataset=="LSA64":
        data_path=sorted(glob.glob(f"{LSA64_DATADIR}/*/*.mp4"))
        gloss2class, class2gloss = create_corpus(data_path)
        train_path, test_path = train_test_split(data_path, test_size=0.2, random_state=42)
        video2class= create_video2class_from_datapath(data_path, gloss2class)
    elif dataset=="GSL":
        train_iso= pd.read_csv(f"{GSL_ISR_TARGET_DATADIR}/train_greek_iso.csv", delimiter="|",header=None)
        dev_iso= pd.read_csv(f"{GSL_ISR_TARGET_DATADIR}/dev_greek_iso.csv", delimiter="|",header=None)
        test_iso= pd.read_csv(f"{GSL_ISR_TARGET_DATADIR}/test_greek_iso.csv", delimiter="|",header=None)
        gloss2class, class2gloss, video2class = create_corpus_gsl(train_iso,dev_iso,test_iso)
        data_path = sorted(glob.glob(f"{GSL_ISR_DATADIR}/*/*"))
        train_path= []
        dev_path= []
        test_path= []
        for path in data_path:
            if len(glob.glob(f"{path}/*.jpg")) < 8:
                continue
            if '/'.join(path.split("/")[-2:]) in train_iso[0].values:
                train_path.append(path)
            if '/'.join(path.split("/")[-2:]) in dev_iso[0].values:
                dev_path.append(path)
            if '/'.join(path.split("/")[-2:]) in test_iso[0].values:
                test_path.append(path)
    elif dataset=="phoenix":
        remove_class={
            "795___ON__",
            "830___PU__",
            "790___OFF__",
            "246___EMOTION__",
            "510___LEFTHAND__",
            "1231_si",
        }
        data_path = []
        copy_path=sorted(glob.glob(f"{WORDS_DATADIR}/*"))
        for gloss_path in copy_path:
            num_file= len(glob.glob(f"{gloss_path}/*"))
            if num_file < 100:
                continue
            elif num_file > 790:#q1+四分位範囲*1.5以上は408個をランダムに選択
                gloss_path_list=random.sample(glob.glob(f"{gloss_path}/*"), k=408)
            else:
                gloss_path_list=glob.glob(f"{gloss_path}/*")
            for file_path in gloss_path_list:
                num_frame= len(glob.glob(f"{file_path}/*.png"))
                if num_frame < 8:
                    continue
                elif file_path.split("/")[-2] in remove_class:
                    continue
                else:
                    data_path.append(file_path)
        gloss2class, class2gloss = create_corpus(data_path)
        train_path, test_path = train_test_split(data_path, test_size=0.2, random_state=42)
        video2class = create_video2class_from_datapath_phoenix(data_path, gloss2class)
    else:
        raise ValueError("dataset is not defined")

    return train_path,  test_path,gloss2class, class2gloss,video2class
if __name__=="__main__":
    import glob
    from Parameter.Parameter import *
    file_path = "/media/caffe/data_storage/How2Sign/how2sign_realigned_val.csv"
    corpus = how2sign_load_corpus(file_path)
    train_path = sorted(glob.glob(f"{HOW2SIGN_DEV_DATADIR}/front_clip/*"))
    for idx in range(len(train_path)):
        file_name = train_path[idx].split("/")[-1]
        print(file_name)
        file_name=file_name.encode('CP932').decode('CP932')
        id = file_name.split(".")[0]
        # corpusのVIDEO_NAMEがidと一致する行を取得
        try:
            video_name = corpus[corpus["SENTENCE_NAME"] == id]["SENTENCE"].values[0]
        except:
            print(f"video_name not found: {id}")
            continue



