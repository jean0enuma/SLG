import torch
import torch.nn as nn
import datetime
dt_now=datetime.datetime.now()

#フレームレベルラベリングのパス
AUTO_SEGMENT_PATH="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/annotations/automatic/train.alignment"
AUTO_TRAIN_CLASS_PATH="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/annotations/automatic/trainingClasses.txt"
#テキストレベルのアノテーションのパス
TEXT_TRAIN_PATH="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/train.corpus.csv"
TEXT_DEV_PATH="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/dev.corpus.csv"
TEXT_TEST_PATH="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/test.corpus.csv"
#学習データのパス
TRAIN_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train"
TEST_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/test"
DEV_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/dev"
#256x256にリサイズしたデータのパス
TRAIN_RESIZED_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/train"
TEST_RESIZED_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/test"
DEV_RESIZED_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/dev"
#segmentationのデータのパス
TRAIN_SEGMENT_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/segmentation/train"
TEST_SEGMENT_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/segmentation/test"
DEV_SEGMENT_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/segmentation/dev"
#binary_maskのデータのパス
TRAIN_BINARY_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/binary_mask/train"
TEST_BINARY_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/binary_mask/test"
DEV_BINARY_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/binary_mask/dev"
#Phoenix-Tのデータのパス
TRAIN_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train"
TEST_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test"
DEV_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev"
#256x256にリサイズしたデータのパス(Phoenix-T)
#TODO: 256x256にリサイズしたデータを作成する
TRAIN_RESIZED_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px/train"
TEST_RESIZED_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px/test"
DEV_RESIZED_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px/dev"
#segmentationのデータのパス(Phoenix-T)
TRAIN_SEGMENT_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/segmentation/train"
TEST_SEGMENT_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/segmentation/test"
DEV_SEGMENT_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/segmentation/dev"
#binary_maskのデータのパス(Phoenix-T)
TRAIN_BINARY_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/binary_mask/train"
TEST_BINARY_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/binary_mask/test"
DEV_BINARY_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/binary_mask/dev"
#Phoenix-Tのテキストのデータのパス
TEXT_TRAIN_PATH_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv"
TEXT_DEV_PATH_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv"
TEXT_TEST_PATH_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv"
#skeleton_phoenix
SKELETON_TRAIN_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/Full_Frame_skeleton/train"
SKELETON_DEV_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/Full_Frame_skeleton/dev"
SKELETON_TEST_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/Full_Frame_skeleton/test"
SKELETON_TRAIN_DATADIR_PROCESSED="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/Full_Frame_skeleton_preprocessed/train"
SKELETON_DEV_DATADIR_PROCESSED="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/Full_Frame_skeleton_preprocessed/dev"
SKELETON_TEST_DATADIR_PROCESSED="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/Full_Frame_skeleton_preprocessed/test"
#face_phoenix
FACE_TRAIN_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/Full_Frame_face/train"
FACE_DEV_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/Full_Frame_face/dev"
FACE_TEST_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/Full_Frame_face/test"
FACE_TRAIN_DATADIR_PROCESSED="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/Full_Frame_face_preprocessed/train"
FACE_DEV_DATADIR_PROCESSED="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/Full_Frame_face_preprocessed/dev/"
FACE_TEST_DATADIR_PROCESSED="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/Full_Frame_face_preprocessed/test"
#skeleton_phoenixT
SKELETON_TRAIN_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_skeleton/train"
SKELETON_DEV_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_skeleton/dev"
SKELETON_TEST_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_skeleton/test"
SKELETON_TRAIN_DATADIR_T_3D="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_skeleton_3D/train"
SKELETON_DEV_DATADIR_T_3D="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_skeleton_3D/dev"
SKELETON_TEST_DATADIR_T_3D="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_skeleton_3D/test"
SKELETON_TRAIN_DATADIR_T_PROCESSED="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_skeleton_preprocessed/train"
SKELETON_DEV_DATADIR_T_PROCESSED="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_skeleton_preprocessed/dev"
SKELETON_TEST_DATADIR_T_PROCESSED="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_skeleton_preprocessed/test"
#face_phoenixT
FACE_TRAIN_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_face/train"
FACE_DEV_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_face/dev"
FACE_TEST_DATADIR_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_face/test"
FACE_TRAIN_DATADIR_T_3D="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_face_3D/train"
FACE_DEV_DATADIR_T_3D="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_face_3D/dev"
FACE_TEST_DATADIR_T_3D="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_face_3D/test"
FACE_TRAIN_DATADIR_T_PROCESSED="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_face_preprocessed/train"
FACE_DEV_DATADIR_T_PROCESSED="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_face_preprocessed/dev"
FACE_TEST_DATADIR_T_PROCESSED="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/Full_Frame_face_preprocessed/test"

WORDS_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/words_video"
SKELETON_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/words_video_pose"
FEATURE_DATADIR="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features"

#optical_flowのパス
TRAIN_DATADIR_OPTICAL="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/optical_flow_gm/train"
TEST_DATADIR_OPTICAL="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/optical_flow_gm/test"
DEV_DATADIR_OPTICAL="/media/caffe/data_storage/phoenix/phoenix2014-release/phoenix-2014-multisigner/features/optical_flow_gm/dev"

TRAIN_DATADIR_OPTICAL_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/optical_flow_gm/train"
TEST_DATADIR_OPTICAL_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/optical_flow_gm/test"
DEV_DATADIR_OPTICAL_T="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/optical_flow_gm/dev"
#保存するモデルのパス
SAVE_MODEL_DIR="/media/caffe/data_storage/CSLR/boundary_models"
SAVE_MODEL_DIR_KWT=f"/media/caffe/data_storage/CSLR/keyword_models/train/{dt_now.strftime('%Y/%m%d/%H%M')}"
SAVE_MODEL_DIR_KWT_TEST=f"/media/caffe/data_storage/CSLR/keyword_models/test/{dt_now.strftime('%Y/%m%d/%H%M')}"
#How2Signのデータのパス
HOW2SIGN_TRAIN_DATADIR="/media/caffe/data_storage/How2Sign/train"
HOW2SIGN_DEV_DATADIR="/media/caffe/data_storage/How2Sign/val"
HOW2SIGN_TEST_DATADIR="/media/caffe/data_storage/How2Sign/test"
SKELETON_HOW2SIGN_TRAIN_DATADIR="/media/caffe/data_storage/How2Sign/train/mediapipe_output"
SKELETON_HOW2SIGN_DEV_DATADIR="/media/caffe/data_storage/How2Sign/val/mediapipe_output"
SKELETON_HOW2SIGN_TEST_DATADIR="/media/caffe/data_storage/How2Sign/test/mediapipe_output"

SKELETON_HOW2SIGN_TRAIN_DATADIR_PROCESSED="/media/caffe/data_storage/How2Sign/train/mediapipe_output_processed"
SKELETON_HOW2SIGN_DEV_DATADIR_PROCESSED="/media/caffe/data_storage/How2Sign/val/mediapipe_output_processed"
SKELETON_HOW2SIGN_TEST_DATADIR_PROCESSED="/media/caffe/data_storage/How2Sign/test/mediapipe_output_processed"

FACE_HOW2SIGN_TRAIN_DATADIR="/media/caffe/data_storage/How2Sign/train/mediapipe_output_face"
FACE_HOW2SIGN_DEV_DATADIR="/media/caffe/data_storage/How2Sign/val/mediapipe_output_face"
FACE_HOW2SIGN_TEST_DATADIR="/media/caffe/data_storage/How2Sign/test/mediapipe_output_face"

FACE_HOW2SIGN_TRAIN_DATADIR_PROCESSED="/media/caffe/data_storage/How2Sign/train/mediapipe_output_face_processed"
FACE_HOW2SIGN_DEV_DATADIR_PROCESSED="/media/caffe/data_storage/How2Sign/val/mediapipe_output_face_processed"
FACE_HOW2SIGN_TEST_DATADIR_PROCESSED="/media/caffe/data_storage/How2Sign/test/mediapipe_output_face_processed"

HOW2SIGN_TEXT_TRAIN_PATH="/media/caffe/data_storage/How2Sign/how2sign_realigned_train.csv"
HOW2SIGN_TEXT_TRAIN_PATH_OLD="/media/caffe/data_storage/How2Sign/how2sign_train.csv"
HOW2SIGN_TEXT_DEV_PATH="/media/caffe/data_storage/How2Sign/how2sign_realigned_val.csv"
HOW2SIGN_TEXT_DEV_PATH_OLD="/media/caffe/data_storage/How2Sign/how2sign_val.csv"
HOW2SIGN_TEXT_TEST_PATH="/media/caffe/data_storage/How2Sign/how2sign_realigned_test.csv"
HOW2SIGN_TEXT_TEST_PATH_OLD="/media/caffe/data_storage/How2Sign/how2sign_test.csv"
#ASL-Citizenのデータのパス
ASL_CITIZEN_DATADIR="/media/caffe/data_storage/ASL_Citizen/ASL_Citizen/videos"
ASL_CITIZEN_TEXT_DATADIR="/media/caffe/data_storage/ASL_Citizen/ASL_Citizen/splits"


#このプロジェクトのパス
PROJECT_DIR="/home/caffe/work/MAE_csr"
#損失関数
LOSS_FUNCTION=nn.CrossEntropyLoss()
#モデルの種類
#S3D,skeleton_transformer
MODEL_TYPE="S3D"
#学習パラメータ
MEAN=[0.1307,0.1307,0.1307]
STD=[0.3081,0.3081,0.3081]



#フレーム数の下限
MIN_FRAMES=16
#テスト時のウィンドウサイズ
WINDOW_SIZE=16
#テスト時のストライド
WINDOW_STRIDE=1

# poseの繋がり
POSE_CONNECTION = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [6, 8], [8, 10], [10, 12], [10, 14], [5, 7], [7, 9], [9, 11],
                   [9, 13]]
# handの繋がり
HAND_CONNECTION = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
                   [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
#削除する骨格のインデックス
COD_DELETE=[1, 3, 4, 6, 9, 10, 21, 22]
POSE_IDX=23
HAND_IDX=21


#Skeleton_transformerのパラメータ
#レイヤーの数
ST_LAYERS=3
#ヘッドの数
ST_ALL_HEADS=13
ST_BODY_HEADS=4
ST_HAND_HEADS=7
#dropout
ST_DROPOUT=0.1

#KEYWORDの数
NUM_KEYWORDS={0:1232,10:400,50:179,100:122,200:74,500:24}
#クラスの閾値
THRESHOLD_CLASS=100

#クラスごとの動画数の平均
MEAN_NUM_VIDEOS={0:49,10:134,50:269,100:350,200:490,500:866}

#---pretrain parameter---
#事前学習用のデータのパス
#AUTSL
AUTSL_TRAIN_DATADIR="/media/caffe/data_storage/datasets/AUTSL/datasets/RGB/train_video"
AUTSL_DEV_DATADIR="/media/caffe/data_storage/datasets/AUTSL/datasets/RGB/val_video"
AUTSL_TEST_DATADIR="/media/caffe/data_storage/datasets/AUTSL/datasets/RGB/test_video"
WLASL2000_DATADIR="/media/caffe/data_storage/datasets/WLASL/movie_full"
AUTSL_TRAIN_COD_DIR="/media/caffe/data_storage/AUTSL/coordinates/train"
AUTSL_DEV_COD_DIR="/media/caffe/data_storage/AUTSL/coordinates/val"
AUTSL_TEST_COD_DIR="/media/caffe/data_storage/AUTSL/coordinates/test"
LSA64_DATADIR="/media/caffe/data_storage/LSA64/lsa64/lsa64_split"
GSL_ISR_DATADIR="/media/caffe/data_storage/GSL/ISR/Greek_isolated/GSL_isol"
GSL_ISR_TARGET_DATADIR="/media/caffe/data_storage/GSL/ISR/GSL_iso_files"
GSL_CSR_DATADIR="/media/caffe/data_storage/GSL/CSR/GSL_continuous"
GSL_CSR_TARGET_DATADIR="/media/caffe/data_storage/GSL/CSR/GSL_continuous_files/GSL SD"
LSA_T_DATADIR="/media/caffe/data_storage/LSA-T/cuts"
LSA_T_METADIR="/media/caffe/data_storage/LSA-T/meta.csv"
LSA_T_KEYDIR="/media/caffe/data_storage/LSA-T/keypoints.h5"
CSL_NEWS_DATADIR="/media/caffe/data_storage/CSL_news"
CSL_DAILY_DATADIR="/media/caffe/data_storage/CSL-Daily/sentence/frames_512x512"
SKELETON_CSL_DAILY_DATADIR="/media/caffe/data_storage/CSL-Daily/sentence/pose_512x512"
SKELETON_CSL_DAILY_DATADIR_PROCESSED="/media/caffe/data_storage/CSL-Daily/sentence/pose_512x512_processed"
FACE_CSL_DAILY_DATADIR="/media/caffe/data_storage/CSL-Daily/sentence/pose_face_512x512"
FACE_CSL_DAILY_DATADIR_PROCESSED="/media/caffe/data_storage/CSL-Daily/sentence/pose_face_512x512_processed"
CSL_DAILY_LABELS="/media/caffe/data_storage/CSL-Daily/sentence_label"
AUTSL_NUM_CLASS=226


