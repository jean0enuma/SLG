import os,shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
import openai
openai.api_key="<>"
class sign_language_description_generator:
    def __init__(self):
        model_id = "NemoStation/Marlin-2B"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        self.model.compile()
        self.processor = AutoProcessor.from_pretrained(model_id)





    def generate_marlin(self,video_path):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that accurately describes sign language movements in detail."
             "Focus on describing the hand shapes, positions, movements, finger configuration, palm orientations, and non-manual movements such as facial expressions and gaze.  Based on the observed video,provide a step by step description that is detailed as possible.."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, },  # ローカルパス or URL
                    {"type": "text", "text": '''
                        You are an expert in sign language linguistics. Observe the following and describe the movements structurally.
    
        [Description Criteria] For each time interval:
        - Hand shape (dominant hand / non-dominant hand)(e.g., flat hand, fist, open hand, etc.)
        - Finger configuration (e.g., which fingers are extended, bent, etc.)
        - Position (in front of or above which part of the body)
        - Movement (direction, path, repetition)
        - Palm orientation
        - Non-manual movements (facial expressions, mouth shape, gaze, head/body)(low priority)
    
        [Perspective] All left/right references must be consistent with “the direction as seen from the camera’s perspective.” Forcus on [Events] more than [Scene].
        [Format] 
        Scene: ... 
        Events:
        <0.0-0.5>:...
        [Constraints] Describe only observable facts. Mark anything that cannot be determined as “Unknown.”
        Write any inferences about meaning only in the “Interpretation” column; do not mix them with the movement descriptions.
                        '''},
                ],
            }
        ]
        # opencvでフレーム数を取得
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            num_frames=frame_count,
            fps=None
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(**inputs,
                                          max_new_tokens=2048,
                                          min_new_tokens=128,
                                          repetition_penalty=1.15,
                                          do_sample=True,
                                            temperature=0.7,
                                            top_p=0.9,
                                           top_k=50,

                                          )
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        # Parse output
        return response
def interpolate_frames(video: np.ndarray, target_frames: int) -> np.ndarray:
    """
    線形補間でフレーム数を増加させる。

    Args:
        video: 入力動画。形状 (T, ...) で先頭軸がフレーム。例: (T, H, W, C)
        target_frames: 出力フレーム数 (T より大きい値)

    Returns:
        形状 (target_frames, ...) の補間済み動画。dtype は入力に合わせる。
    """
    T = video.shape[0]
    if target_frames < 2 or T < 2:
        raise ValueError("T と target_frames は 2 以上が必要です")

    orig_dtype = video.dtype
    work = video.astype(np.float32)

    # 元フレーム位置 [0, T-1] 上に新しいサンプル点を等間隔に配置
    positions = np.linspace(0, T - 1, target_frames)
    idx0 = np.floor(positions).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, T - 1)

    # 重みを後続次元にブロードキャストできる形へ
    w = (positions - idx0).reshape(-1, *([1] * (video.ndim - 1)))

    out = (1.0 - w) * work[idx0] + w * work[idx1]

    # uint8 等の整数型なら丸めてクリップして戻す
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        out = np.clip(np.round(out), info.min, info.max)
    return out.astype(orig_dtype)
def create_video(img_files_path,save_path,file_name="temp.mp4",ext="png"):
    img_files=sorted(glob.glob(f"{img_files_path}/*.{ext}"))
    if len(img_files)==0:
        print("No images found in the specified path.")
        return
    img=cv2.imread(img_files[0])
    height,width,layers=img.shape
    img_list=[]
    for img_file in img_files:
        img=cv2.imread(img_file)
        img_list.append(img)
    video_data=interpolate_frames(np.array(img_list), target_frames=int(2*len(img_list)))
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    video=cv2.VideoWriter(f"{save_path}/{file_name}",fourcc,1,(width,height))
    for img in video_data:
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
    img_list=[]
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        img_list.append(frame)
    video_data=interpolate_frames(np.array(img_list), target_frames=int(2*len(img_list)))
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    video=cv2.VideoWriter(f"{save_path}/{file_name}",fourcc,fps,(width,height))
    for img in video_data:
        video.write(img)
    cap.release()
    video.release()

def sign_language_description(dataset,save_path):
    descriptor=sign_language_description_generator()
    if dataset=="CSL-Daily":
        ext="jpg"
    else:
        ext="png"
    train_path, dev_path, test_path, gloss2class, class2gloss, video2gloss=islr_datasets_loader(dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    #textデータはjsonl形式で保存する
    train_save_path=f"{save_path}/train_sign_description.jsonl"
    dev_save_path=f"{save_path}/dev_sign_description.jsonl"
    test_save_path=f"{save_path}/test_sign_description.jsonl"
    """
    train_text=[]
    for data_path in train_path:
        gloss_dir=os.path.basename(os.path.dirname(data_path))
        print(f"Processing {data_path}...")
        if os.path.basename(data_path).split(".")[1]!="mp4":
            data_id=os.path.basename(data_path)
            img_data_path=sorted(glob.glob(f"{data_path}/*.{ext}"))
            create_video(img_data_path,save_path,file_name=f"tmp.mp4")
            video_path=f"{save_path}/tmp.mp4"
        else:
            data_id=os.path.basename(data_path).split(".")[0]
            recreate_video(data_path,save_path,file_name=f"tmp.mp4",fps=1)
            video_path=f"{save_path}/tmp.mp4"
        response=descriptor.generate_marlin(video_path)
        print(f"Generated description for {data_id}:\n {response}...")  # Print first 100 characters of the response
        train_text.append({"data_id": data_id, "description": response,"is_flip":False,"gloss_dir":gloss_dir})
        #左右反転の場合も取り入れる
        #responseのrightとleftを入れ替える
        response_flip=response.replace("right","|temp|").replace("left","right").replace("|temp|","left")
        response_flip=response_flip.replace("Right","|temp|").replace("Left","Right").replace("|temp|","Left")
        response_flip=response_flip.replace("RIGHT","|temp|").replace("LEFT","RIGHT").replace("|temp|","LEFT")
        print(f"Generated flipped description for {data_id}:\n {response_flip}...")
        train_text.append({"data_id": data_id, "description": response_flip,"is_flip":True,"gloss_dir":gloss_dir})
    df=pd.DataFrame(train_text)
    df.to_json(train_save_path,  force_ascii=False, lines=True, orient='records')
    """
    train_text=[]
    for data_path in dev_path:
        gloss_dir=os.path.basename(os.path.dirname(data_path))
        print(f"Processing {data_path}...")
        if os.path.basename(data_path).split(".")[1]!="mp4":
            data_id=os.path.basename(data_path)
            img_data_path=sorted(glob.glob(f"{data_path}/*.{ext}"))
            create_video(img_data_path,save_path,file_name="tmp.mp4")
            video_path=f"{save_path}/tmp.mp4"
        else:
            data_id=os.path.basename(data_path).split(".")[0]
            recreate_video(data_path, save_path, file_name=f"tmp.mp4", fps=3)
            video_path = f"{save_path}/tmp.mp4"
        response=descriptor.generate_marlin(video_path)
        train_text.append({"data_id": data_id, "description": response,"is_flip":False,"gloss_dir":gloss_dir})
        #左右反転の場合も取り入れる
        #responseのrightとleftを入れ替える
        response_flip=response.replace("right","|temp|").replace("left","right").replace("|temp|","left")
        response_flip=response_flip.replace("Right","|temp|").replace("Left","Right").replace("|temp|","Left")
        response_flip=response_flip.replace("RIGHT","|temp|").replace("LEFT","RIGHT").replace("|temp|","LEFT")
        print(f"Generated flipped description for {data_id}:\n {response_flip}...")
        train_text.append({"data_id": data_id, "description": response_flip,"is_flip":True,"gloss_dir":gloss_dir})
    df=pd.DataFrame(train_text)
    df.to_json(dev_save_path,  force_ascii=False, lines=True, orient='records')
    train_text=[]
    for data_path in test_path:
        gloss_dir=os.path.basename(os.path.dirname(data_path))
        print(f"Processing {data_path}...")
        if os.path.basename(data_path).split(".")[1]!="mp4":
            data_id=os.path.basename(data_path)
            img_data_path=sorted(glob.glob(f"{data_path}/*.{ext}"))
            create_video(img_data_path,save_path,file_name="tmp.mp4")
            video_path=f"{save_path}/tmp.mp4"
        else:
            data_id=os.path.basename(data_path).split(".")[0]
            recreate_video(data_path, save_path, file_name=f"tmp.mp4", fps=3)
            video_path = f"{save_path}/tmp.mp4"
        response=descriptor.generate_marlin(video_path)
        train_text.append({"data_id": data_id, "description": response,"is_flip":False,"gloss_dir":gloss_dir})
        #左右反転の場合も取り入れる
        #responseのrightとleftを入れ替える
        response_flip=response.replace("right","|temp|").replace("left","right").replace("|temp|","left")
        response_flip=response_flip.replace("Right","|temp|").replace("Left","Right").replace("|temp|","Left")
        response_flip=response_flip.replace("RIGHT","|temp|").replace("LEFT","RIGHT").replace("|temp|","LEFT")
        print(f"Generated flipped description for {data_id}:\n {response_flip}...")
        train_text.append({"data_id": data_id, "description": response_flip,"is_flip":True,"gloss_dir":gloss_dir})
    df=pd.DataFrame(train_text)
    df.to_json(test_save_path,  force_ascii=False, lines=True, orient='records')
    os.remove(f"{save_path}/tmp.mp4")
def sign_language_description_T(dataset,save_path):
    descriptor=sign_language_description_generator()
    if dataset=="CSL-Daily":
        ext="jpg"
    else:
        ext="png"
    train_path, dev_path, test_path, train_target_corpus, dev_target_corpus, test_target_corpus=datasets_loader_T(dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    #textデータはjsonl形式で保存する
    train_save_path=f"{save_path}/train_sign_description.jsonl"
    dev_save_path=f"{save_path}/dev_sign_description.jsonl"
    test_save_path=f"{save_path}/test_sign_description.jsonl"

    train_text=[]
    for data_path in train_path:
        file_name=os.path.basename(data_path)
        seq=train_target_corpus[train_target_corpus["id"] == file_name]["annotation"].values[0]
        print(f"Processing {data_path}...")
        if os.path.basename(data_path)[-3:]!="mp4":
            data_id=os.path.basename(data_path)
            create_video(data_path,save_path,file_name=f"tmp.mp4",ext=ext)
            video_path=f"{save_path}/tmp.mp4"
        else:
            data_id=os.path.basename(data_path).split(".")[0]
            recreate_video(data_path,save_path,file_name=f"tmp.mp4",fps=1)
            video_path=f"{save_path}/tmp.mp4"
        response=descriptor.generate_marlin(video_path)
        print(f"Generated description for {data_id}:\n {response}...")  # Print first 100 characters of the response
        train_text.append({"data_id": data_id, "description": response,"is_flip":False,"seq":seq})
        #左右反転の場合も取り入れる
        #responseのrightとleftを入れ替える
        response_flip=response.replace("right","|temp|").replace("left","right").replace("|temp|","left")
        response_flip=response_flip.replace("Right","|temp|").replace("Left","Right").replace("|temp|","Left")
        response_flip=response_flip.replace("RIGHT","|temp|").replace("LEFT","RIGHT").replace("|temp|","LEFT")
        print(f"Generated flipped description for {data_id}:\n {response_flip}...")
        train_text.append({"data_id": data_id, "description": response_flip,"is_flip":True,"seq":seq})
    df=pd.DataFrame(train_text)
    df.to_json(train_save_path,  force_ascii=False, lines=True, orient='records')

    train_text=[]
    for data_path in dev_path:
        file_name = os.path.basename(data_path)
        seq = dev_target_corpus[dev_target_corpus["id"] == file_name]["annotation"].values[0]
        print(f"Processing {data_path}...")
        if os.path.basename(data_path)[-3:] != "mp4":
            data_id = os.path.basename(data_path)
            create_video(data_path, save_path, file_name=f"tmp.mp4", ext=ext)
            video_path = f"{save_path}/tmp.mp4"
        else:
            data_id = os.path.basename(data_path).split(".")[0]
            recreate_video(data_path, save_path, file_name=f"tmp.mp4", fps=1)
            video_path = f"{save_path}/tmp.mp4"
        response = descriptor.generate_marlin(video_path)
        print(f"Generated description for {data_id}:\n {response}...")  # Print first 100 characters of the response
        train_text.append({"data_id": data_id, "description": response, "is_flip": False, "seq": seq})
        # 左右反転の場合も取り入れる
        # responseのrightとleftを入れ替える
        response_flip = response.replace("right", "|temp|").replace("left", "right").replace("|temp|", "left")
        response_flip = response_flip.replace("Right", "|temp|").replace("Left", "Right").replace("|temp|", "Left")
        response_flip = response_flip.replace("RIGHT", "|temp|").replace("LEFT", "RIGHT").replace("|temp|", "LEFT")
        print(f"Generated flipped description for {data_id}:\n {response_flip}...")
        train_text.append({"data_id": data_id, "description": response_flip, "is_flip": True, "seq": seq})
    df=pd.DataFrame(train_text)
    df.to_json(dev_save_path,  force_ascii=False, lines=True, orient='records')
    train_text=[]
    for data_path in test_path:
        file_name = os.path.basename(data_path)
        seq = test_target_corpus[test_target_corpus["id"] == file_name]["annotation"].values[0]
        print(f"Processing {data_path}...")
        if os.path.basename(data_path)[-3:] != "mp4":
            data_id = os.path.basename(data_path)
            create_video(data_path, save_path, file_name=f"tmp.mp4", ext=ext)
            video_path = f"{save_path}/tmp.mp4"
        else:
            data_id = os.path.basename(data_path).split(".")[0]
            recreate_video(data_path, save_path, file_name=f"tmp.mp4", fps=1)
            video_path = f"{save_path}/tmp.mp4"
        response = descriptor.generate_marlin(video_path)
        print(f"Generated description for {data_id}:\n {response}...")  # Print first 100 characters of the response
        train_text.append({"data_id": data_id, "description": response, "is_flip": False, "seq": seq})
        # 左右反転の場合も取り入れる
        # responseのrightとleftを入れ替える
        response_flip = response.replace("right", "|temp|").replace("left", "right").replace("|temp|", "left")
        response_flip = response_flip.replace("Right", "|temp|").replace("Left", "Right").replace("|temp|", "Left")
        response_flip = response_flip.replace("RIGHT", "|temp|").replace("LEFT", "RIGHT").replace("|temp|", "LEFT")
        print(f"Generated flipped description for {data_id}:\n {response_flip}...")
        train_text.append({"data_id": data_id, "description": response_flip, "is_flip": True, "seq": seq})
    df=pd.DataFrame(train_text)
    df.to_json(test_save_path,  force_ascii=False, lines=True, orient='records')
    os.remove(f"{save_path}/tmp.mp4")
def extract_class_centroid(dataset,save_path):
    if dataset=="CSL-Daily":
        ext="jpg"
    else:
        ext="png"
    train_path, dev_path, test_path, gloss2class, class2gloss, video2gloss=islr_datasets_loader(dataset)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    #textデータはjsonl形式で保存する
    train_save_path=f"{save_path}/train_class_centroid.jsonl"
    dev_save_path=f"{save_path}/dev_class_centroid.jsonl"
    test_save_path=f"{save_path}/test_class_centroid.jsonl"
    #train_save_pathをロード
    if os.path.exists(train_save_path):
        train_df=pd.read_json(train_save_path,lines=True)
    else:
        raise ValueError(f"{train_save_path} does not exist. Please run sign_language_description first.")
    gloss_embedding_dict={}
    for index,row in train_df.iterrows():
        gloss_dir=row["gloss_dir"]
        description=row["description"]
        if gloss_dir not in gloss_embedding_dict:
            gloss_embedding_dict[gloss_dir]=[]
        response=openai.Embedding.create(input=description, model="text-embedding-ada-002")
        embeds=[d["embedding"] for d in response["data"]]
        gloss_embedding_dict[gloss_dir].append(embeds[0])
    #各gloss_dirのembeddingの平均を計算する
    gloss_centroid_dict={}
    for gloss_dir,embeddings in gloss_embedding_dict.items():
        gloss_centroid_dict[gloss_dir]=np.mean(embeddings,axis=0).tolist()
    #gloss_centroid_dictをnpz形式で保存する
    np.savez(f"{save_path}/gloss_centroid.npz",**gloss_centroid_dict)
if __name__=="__main__":
    #video_path="/home/caffe/work/signer0_sample431_color.mp4"
    #response=generate_marlin(video_path)
    #print(response)
    sign_language_description_T("phoenixT",save_path="/media/caffe/data_storage/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/pose_description")
    """
    Please provide a detailed description of the sign language user’s posture and movements in this video, focusing particularly on the arms, both hands, and facial expressions.
Please format your answer according to the table below.
[Time Range] | [Right Hand] | [Left Hand] | [Facial Expression]
    """