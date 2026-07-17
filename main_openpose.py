import subprocess
from pathlib import Path
import glob,os
from Parameter.Parameter import *
import pandas as pd
import numpy as np
import shutil,re,json
OPENPOSE_ROOT = Path("/home/caffe/work/openpose")

def _sorted_json_files(json_dir: Path) -> list:
    """ファイル名中の数値でソートしたJSONファイルのリストを返す"""
    files = list(json_dir.glob("*.json"))
    def frame_num(p: Path) -> int:
        nums = re.findall(r"\d+", p.stem)
        return int(nums[-1]) if nums else -1  # 末尾の数値をフレーム番号とみなす
    return sorted(files, key=frame_num)


def _extract_xy(json_path: Path, key: str, num_points: int,size: tuple) -> np.ndarray:
    """JSONから指定キーのキーポイントを読み、confidenceを除いた(x,y)を返す"""
    with open(json_path) as f:
        data = json.load(f)
    people = data.get("people", [])
    if len(people) == 0:  # 検出失敗フレームはゼロ埋め
        return np.zeros(num_points * 2, dtype=np.float32)
    kps = np.array(people[0].get(key, []), dtype=np.float32)
    if kps.size != num_points * 3:
        return np.zeros(num_points * 2, dtype=np.float32)
    kps = kps.reshape(-1, 3)[:, :2]  # (N, 3) -> (N, 2)  confidenceを除去
    kps[:, 0] /= size[0]  # x座標を正規化
    kps[:, 1] /= size[1]  # y座標を正
    return kps.flatten()  # (x1, y1, x2, y2, ...)


def merge_body_hand_to_csv(body_json_dir: str, hand_json_dir: str, output_dir: str,size: tuple = (210, 260),
                           csv_name: str = "keypoints.csv") -> Path:
    """
    body/handのOpenPose JSON連番ファイルを結合し、(T, F)のCSVとして保存する。

    F方向の構成 (計134次元):
        body 25点 (x,y) = 50
      + left hand 21点 (x,y) = 42
      + right hand 21点 (x,y) = 42

    Returns:
        保存したCSVファイルのパス
    """
    body_dir, hand_dir = Path(body_json_dir), Path(hand_json_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    body_files = _sorted_json_files(body_dir)
    hand_files = _sorted_json_files(hand_dir)

    if len(body_files) == 0:
        raise FileNotFoundError(f"No JSON files in {body_dir}")
    if len(body_files) != len(hand_files):
        raise ValueError(
            f"Frame count mismatch: body={len(body_files)}, hand={len(hand_files)}"
        )

    rows = []
    for bf, hf in zip(body_files, hand_files):
        body_xy  = _extract_xy(bf, "pose_keypoints_2d", 25,size)        # (50,)
        lhand_xy = _extract_xy(hf, "hand_left_keypoints_2d", 21,size)   # (42,)
        rhand_xy = _extract_xy(hf, "hand_right_keypoints_2d", 21,size)  # (42,)
        rows.append(np.concatenate([body_xy, lhand_xy, rhand_xy])) # (134,)

    arr = np.stack(rows)  # (T, 134)

    # 列名: body_x0, body_y0, ..., lhand_x0, ..., rhand_y20
    cols = (
        [f"body_{ax}{i}"  for i in range(25) for ax in ("x", "y")]
        + [f"lhand_{ax}{i}" for i in range(21) for ax in ("x", "y")]
        + [f"rhand_{ax}{i}" for i in range(21) for ax in ("x", "y")]
    )
    df = pd.DataFrame(arr, columns=cols)

    out_path = out_dir / csv_name
    df.to_csv(out_path, index=False,header=False)
    return out_path
def face_json_to_csv(face_json_dir: str, output_dir: str,
                     csv_name: str = "face_keypoints.csv",size: tuple =(210,260)) -> Path:
    """
    faceのOpenPose JSON連番ファイルを(T, F)のCSVとして保存する(ヘッダなし)。

    F方向の構成: face 70点 (x,y) = 140次元

    Returns:
        保存したCSVファイルのパス
    """
    face_dir = Path(face_json_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    face_files = _sorted_json_files(face_dir)
    if len(face_files) == 0:
        raise FileNotFoundError(f"No JSON files in {face_dir}")

    rows = [_extract_xy(f, "face_keypoints_2d", 70,size=size) for f in face_files]  # 各(140,)
    arr = np.stack(rows)  # (T, 140)

    out_path = out_dir / csv_name
    pd.DataFrame(arr).to_csv(out_path, index=False, header=False)
    return out_path
def run_openpose(video_path: str, json_dir: str):
    cmd = [
        str(OPENPOSE_ROOT / "build/examples/openpose/openpose.bin"),
        "--video", video_path,
        "--write_json", json_dir,
        "--hand", "--face",
        "--display", "0",
        "--render_pose", "0",
        "--number_people_max", "1",
    ]
    # モデルパスが相対参照なので cwd 指定が必須
    subprocess.run(cmd, cwd=OPENPOSE_ROOT, check=True)
def run_openpose_images_hand(image_dir: str, json_dir: str):
    Path(json_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        str(OPENPOSE_ROOT / "build/examples/openpose/openpose.bin"),
        "--image_dir", image_dir,
        "--write_json", json_dir,
        "--hand",
        "--display", "0",
        "--render_pose", "0",
        "--number_people_max", "1",
    ]
    subprocess.run(cmd, cwd=OPENPOSE_ROOT, check=True)
def run_openpose_images_face(image_dir: str, json_dir: str):
    Path(json_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        str(OPENPOSE_ROOT / "build/examples/openpose/openpose.bin"),
        "--image_dir", image_dir,
        "--write_json", json_dir,
        "--face",
        "--display", "0",
        "--render_pose", "0",
        "--number_people_max", "1",
    ]
    subprocess.run(cmd, cwd=OPENPOSE_ROOT, check=True)
def run_openpose_images_body(image_dir: str, json_dir: str):
    Path(json_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        str(OPENPOSE_ROOT / "build/examples/openpose/openpose.bin"),
        "--image_dir", image_dir,
        "--write_json", json_dir,
        "--display", "0",
        "--render_pose", "0",
        "--number_people_max", "1",
    ]
    subprocess.run(cmd, cwd=OPENPOSE_ROOT, check=True)
if __name__=="__main__":
    #run_openpose("/home/caffe/work/openpose/examples/media/video.avi", "/home/caffe/work/openpose/examples/media/sign001/")
    dataset = "phoenixT"  # "phoenixT" or "CSL-Daily"
    if dataset=="phoenixT":
        train_data_root=TRAIN_DATADIR_T
        dev_data_root=DEV_DATADIR_T
        test_data_root=TEST_DATADIR_T
        train_data_root_face=FACE_TRAIN_DATADIR_T
        dev_data_root_face=FACE_DEV_DATADIR_T
        test_data_root_face=FACE_TEST_DATADIR_T
        size=(210,260)
        skeleton_train_root=SKELETON_TRAIN_DATADIR_T_OPENPOSE
        skeleton_dev_root=SKELETON_DEV_DATADIR_T_OPENPOSE
        skeleton_test_root=SKELETON_TEST_DATADIR_T_OPENPOSE
        skeleton_train_root_face=FACE_TRAIN_DATADIR_T_OPENPOSE
        skeleton_dev_root_face=FACE_DEV_DATADIR_T_OPENPOSE
        skeleton_test_root_face=FACE_TEST_DATADIR_T_OPENPOSE
        """
        if os.path.exists(skeleton_train_root):
            shutil.rmtree(skeleton_train_root)
        if os.path.exists(skeleton_dev_root):
            shutil.rmtree(skeleton_dev_root)
        if os.path.exists(skeleton_test_root):
            shutil.rmtree(skeleton_test_root)
        """
        if os.path.exists(skeleton_train_root_face):
            shutil.rmtree(skeleton_train_root_face)
        if os.path.exists(skeleton_dev_root_face):
            shutil.rmtree(skeleton_dev_root_face)
        if os.path.exists(skeleton_test_root_face):
            shutil.rmtree(skeleton_test_root_face)
        os.makedirs(skeleton_train_root, exist_ok=True)
        os.makedirs(skeleton_dev_root, exist_ok=True)
        os.makedirs(skeleton_test_root, exist_ok=True)


        train_data_dirs=sorted(glob.glob(f"{train_data_root}/*"))
        dev_data_dirs=sorted(glob.glob(f"{dev_data_root}/*"))
        test_data_dirs=sorted(glob.glob(f"{test_data_root}/*"))
        train_data_face_dirs=sorted(glob.glob(f"{train_data_root_face}/*"))
        dev_data_face_dirs=sorted(glob.glob(f"{dev_data_root_face}/*"))
        test_data_face_dirs=sorted(glob.glob(f"{test_data_root_face}/*"))
        print(f"==train==")
        for train_data_dir in train_data_dirs:
            print(f"Processing {train_data_dir}...")
            file_name=os.path.basename(train_data_dir)
            #openposeを実行してjsonファイルを出力
            #run_openpose_images_body(train_data_dir, f"{skeleton_train_root}/temp_body")
            #run_openpose_images_hand(train_data_dir, f"{skeleton_train_root}/temp_hand")
            run_openpose_images_face(train_data_dir, f"{skeleton_train_root_face}/temp_face")
            #body,handのjsonファイルを結合して1つのjsonファイルにする
            #merge_body_hand_to_csv(f"{skeleton_train_root}/temp_body", f"{skeleton_train_root}/temp_hand", f"{skeleton_train_root}", csv_name=f"{file_name}.csv",size=size)
            face_json_to_csv(f"{skeleton_train_root_face}/temp_face", f"{skeleton_train_root_face}", csv_name=f"{file_name}.csv",size=size)
            #shutil.rmtree(f"{skeleton_train_root}/temp_body")
            #shutil.rmtree(f"{skeleton_train_root}/temp_hand")
            shutil.rmtree(f"{skeleton_train_root_face}/temp_face")
        print(f"==dev==")
        for dev_data_dir in dev_data_dirs:
            print(f"Processing {dev_data_dir}...")
            file_name=os.path.basename(dev_data_dir)
            #openposeを実行してjsonファイルを出力
            #run_openpose_images_body(dev_data_dir, f"{skeleton_dev_root}/temp_body")
            #run_openpose_images_hand(dev_data_dir, f"{skeleton_dev_root}/temp_hand")
            run_openpose_images_face(dev_data_dir, f"{skeleton_dev_root_face}/temp_face")
            #body,handのjsonファイルを結合して1つのjsonファイルにする
            #merge_body_hand_to_csv(f"{skeleton_dev_root}/temp_body", f"{skeleton_dev_root}/temp_hand", f"{skeleton_dev_root}", csv_name=f"{file_name}.csv",size=size)
            face_json_to_csv(f"{skeleton_dev_root_face}/temp_face", f"{skeleton_dev_root_face}", csv_name=f"{file_name}.csv",size=size)
            #shutil.rmtree(f"{skeleton_dev_root}/temp_body")
            #shutil.rmtree(f"{skeleton_dev_root}/temp_hand")
            shutil.rmtree(f"{skeleton_dev_root_face}/temp_face")
        print(f"==test==")
        for test_data_dir in test_data_dirs:
            print(f"Processing {test_data_dir}...")
            file_name=os.path.basename(test_data_dir)
            #openposeを実行してjsonファイルを出力
            #run_openpose_images_body(test_data_dir, f"{skeleton_test_root}/temp_body")
            #run_openpose_images_hand(test_data_dir, f"{skeleton_test_root}/temp_hand")
            run_openpose_images_face(test_data_dir, f"{skeleton_test_root_face}/temp_face")
            #body,handのjsonファイルを結合して1つのjsonファイルにする
            #merge_body_hand_to_csv(f"{skeleton_test_root}/temp_body", f"{skeleton_test_root}/temp_hand", f"{skeleton_test_root}", csv_name=f"{file_name}.csv")
            face_json_to_csv(f"{skeleton_test_root_face}/temp_face", f"{skeleton_test_root_face}", csv_name=f"{file_name}.csv",size=size)
            #shutil.rmtree(f"{skeleton_test_root_face}/temp_body")
            #shutil.rmtree(f"{skeleton_test_root_face}/temp_hand")
            shutil.rmtree(f"{skeleton_test_root_face}/temp_face")





    elif dataset=="CSL-Daily":
        data_root=CSL_DAILY_DATADIR
        skeleton_data_root=SKELETON_CSL_DAILY_DATADIR_OPENPOSE
    else:
        raise ValueError("Invalid dataset name")


