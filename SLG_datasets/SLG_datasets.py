import pandas as pd
import torch
import torchvision.io
from torch.utils.data import Dataset
from loader import *
from utils import TemporalRescale, process_text
from Parameter.Parameter_phoenix import *
from torch.utils.data import Dataset
import h5py
from Parameter.Parameter import LSA_T_KEYDIR,LSA_T_METADIR

class CSLR_datasets_onlydata(Dataset):
    """
    phoenixデータセットを読み込むためのクラス(ctc_loss用)
    loaderの出力はdata,targets,input_length,target_length
    data:入力データ(処理後)
    targets:ラベル系列
    input_length:入力データの長さ(torch.tensor, [batch_size])
    target_length:ラベル系列の長さ(torch.tensor, [batch_size])
    """

    def __init__(self, data_path,data_dict,  transforms, trainable=True, resize=256):
        super().__init__()
        self.data_path = data_path
        self.data_dict=data_dict
        # self.class2gloss= {v: k for k, v in gloss2class.items()}
        self.transforms = transforms
        self.trainable = trainable
        self.resize = resize
        self.keypoints=h5py.File(LSA_T_KEYDIR,"r")
        self.metadata=pd.read_csv(LSA_T_METADIR)

    def __len__(self):
        return len(self.data_path)
    def load_keypoints(self,video_path):
        video_name=video_path.split("/")[-1]
        video_name_no_mp4=video_name.split(".")[0]
        #self.metadataのidがvideo_name_no_mp4に一致する行を取得
        row=self.metadata[self.metadata["id"]==video_name_no_mp4]
        if len(row)==0:
            raise ValueError(f"video {video_name_no_mp4} not found in metadata")
        #row行のsigner_amount列の値を取得
        signer_amount=row["signers_amount"].values[0]
        if signer_amount>1:
            signer_id=row["infered_signer"].values[0]
            print(row["infered_signer_confidence"].values[0])
        else:
            signer_id="signer_0"
        try:
            boxes=self.keypoints[video_name][signer_id]["boxes"][:]
        except KeyError as e:
            print(video_path)
            return None
        boxes=pd.DataFrame(boxes)
        boxes=boxes.interpolate(method="linear",limit_direction="both").values
        return boxes
    def clip_signer(self,data,keypoints):
        """
        numpy配列の動画データをキーポイントに基づいてトリミングする
        画像サイズは256x256にリサイズする
        :param data:
        :param keypoints:
        :return:
        """
        T, H, W, C = data.shape
        new_data=np.zeros((T,256,256,C),dtype=data.dtype)
        max_width=np.max(keypoints[:,2]-keypoints[:,0])
        max_height=np.max(keypoints[:,3]-keypoints[:,1])
        max_size=int(max(max_width,max_height))+20
        for t in range(T):
            if keypoints[t][0]==0 and keypoints[t][1]==0 and keypoints[t][2]==0 and keypoints[t][3]==0:
                new_data[t]=data[t]
                continue
            x1 = int(max(0, keypoints[t][0] - (max_size - (keypoints[t][2] - keypoints[t][0])) // 2))
            y1 = int(max(0, keypoints[t][1] - (max_size - (keypoints[t][3] - keypoints[t][1])) // 2))
            x2 = int(min(W, x1 + max_size))
            y2 = int(min(H, y1 + max_size))
            cropped=data[t,y1:y2,x1:x2,:]
            resized=cv2.resize(cropped,(256,256),interpolation=cv2.INTER_LANCZOS4)
            new_data[t]=resized
        return new_data
    def clip_square(self,data,img_size=256):
        """
        numpy配列の動画データを中央を基準に正方形にトリミングする
        画像サイズは256x256にリサイズする
        :param data:
        :return:
        """
        T, H, W, C = data.shape
        sq_size=min(H,W)
        new_data=np.zeros((T,img_size,img_size,C),dtype=data.dtype)
        for t in range(T):
            if H>W:
                y1=(H-W)//2
                y2=y1+W
                cropped=data[t,y1:y2,:,:]
            else:
                x1=(W-H)//2
                x2=x1+H
                cropped=data[t,:,x1:x2,:]
            new_data[t]=torchvision.transforms.functional.resize(torch.from_numpy(cropped).permute(2,0,1),[img_size,img_size]).permute(1,2,0).numpy()
        return new_data

    def __getitem__(self, idx):
        # データの読み込み
        # data=image2video(self.data_path[idx],img_size=(self.resize,self.resize))
        dataset_id, data_path = self.data_path[idx]
        #print(data_path)
        if dataset_id==0 or dataset_id == 1:
            data = image2video(data_path, img_size=(self.resize, self.resize), ext="png")
        elif dataset_id==2:
            data=video2npy(data_path,img_size=None)
            data=self.clip_square(data,img_size=self.resize)
            #self.show_video(data)
        elif dataset_id == 3 or dataset_id==5:
            data = image2video(data_path, img_size=(self.resize, self.resize), ext="jpg")
        elif dataset_id==4:
            keypoints=self.load_keypoints(data_path)
            if keypoints is None:
                data = video2npy(data_path, img_size=(self.resize, self.resize))
                print(f"keypoints not found for {data_path}, using full frame")
            else:
                data = video2npy(data_path, img_size=None)
                data=self.clip_signer(data,keypoints)
        else:
            data = video2npy(data_path, img_size=(self.resize, self.resize))
        data = torch.from_numpy(data).float()
        data = data.permute(0, 3, 1, 2)
        if data.size()[0]>300:
            #300フレームを超える場合はランダムに300フレームをクリップ
            start_idx=np.random.randint(0,data.size()[0]-300)
            data=data[start_idx:start_idx+300]
        # data=npy2video(self.data_path[idx])
        data = self.transforms(data)
        data = data / 127.5 - 1.0
        # data.size()=(T,C,H,W)

        input_length = torch.from_numpy(np.array(len(data)))
        return data, input_length,  data_path, dataset_id

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, input_length,  data_path,dataset_id = list(zip(*batch))
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            # video_length = torch.LongTensor([len(vid) for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            # right_pad=0
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)

        return padded_video, video_length,  data_path, dataset_id
    @staticmethod
    def collate_fn_nopadding(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, input_length,  data_path, dataset_id = list(zip(*batch))
        # input_lengthが最も長いものに合わせてパディング
        max_len = len(video[0])
        video_length = torch.LongTensor([len(vid) for vid in video])
        video = torch.stack([torch.cat([vid, torch.zeros(max_len - len(vid), *vid.shape[1:])], dim=0) for vid in video])
        # それ以外はそのまま
        input_length = torch.LongTensor(input_length)
        return video, input_length, data_path, dataset_id
class CSLR_datasets(CSLR_datasets_onlydata):
    def __init__(self,data_path,data_dict,targets_corpus,gloss2class,  transforms, trainable=True, resize=256):
        super().__init__(data_path,data_dict, transforms, trainable, resize)
        self.targets_corpus=targets_corpus
        self.gloss2class=gloss2class

    def __getitem__(self, idx):
        # データの読み込み
        # data=image2video(self.data_path[idx],img_size=(self.resize,self.resize))
        dataset_id, data_path = self.data_path[idx]
        #print(data_path)
        if dataset_id==0 or dataset_id == 1:
            data = image2video(data_path, img_size=(self.resize, self.resize), ext="png")
            if data_path.split("/")[-1] == "1":
                id = data_path.split("/")[-2]
            else:
                id = data_path.split("/")[-1]
            sequence = self.targets_corpus[self.targets_corpus["id"] == id]["annotation"].values[0]            # ラベル系列の取得
            # ラベル系列をクラスに変換
            targets = [self.gloss2class[gloss] for gloss in sequence.split(" ") if gloss in self.gloss2class.keys()]
            # ラベル系列の長さ
            target_length = torch.tensor(len(targets))
            targets = torch.tensor(targets)
            truth_label_flag=True
        elif dataset_id==2:
            data=video2npy(data_path,img_size=None)
            data=self.clip_square(data,img_size=self.resize)
            targets=torch.tensor(-1)#nullを-1で表現
            target_length=torch.tensor(0)
            truth_label_flag=False
            #self.show_video(data)
        elif dataset_id == 3:
            data = image2video(data_path, img_size=(self.resize, self.resize), ext="jpg")
            targets=torch.tensor(-1)#nullを-1で表現
            target_length=torch.tensor(0)
            truth_label_flag=False
            #self.show_video(data)
        elif dataset_id==4:
            keypoints=self.load_keypoints(data_path)
            if keypoints is None:
                data = video2npy(data_path, img_size=(self.resize, self.resize))
                print(f"keypoints not found for {data_path}, using full frame")
            else:
                data = video2npy(data_path, img_size=None)
                data=self.clip_signer(data,keypoints)
            targets=torch.tensor(-1)#nullを-1で表現
            target_length=torch.tensor(0)
            truth_label_flag=False
            #self.show_video(data)
        else:
            data = video2npy(data_path, img_size=(self.resize, self.resize))
            targets=torch.tensor(-1)#nullを-1で表現
            target_length=torch.tensor(0)
            truth_label_flag=False
            #self.show_video(data)
        data = torch.from_numpy(data).float()
        data = data.permute(0, 3, 1, 2)
        if data.size()[0]>300:
            #300フレームを超える場合はランダムに300フレームをクリップ
            start_idx=np.random.randint(0,data.size()[0]-300)
            data=data[start_idx:start_idx+300]
        # data=npy2video(self.data_path[idx])
        data = self.transforms(data)
        data = data / 127.5 - 1.0
        # data.size()=(T,C,H,W)

        input_length = torch.from_numpy(np.array(len(data)))

        return data, input_length,  data_path,targets,target_length
    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, input_length,data_path,label,label_length = list(zip(*batch))
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            #video_length = torch.LongTensor([len(vid) for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            #right_pad=0
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor(label_length)
        padded_label = []
        for lab in label:
            if lab.dim() == 0:
                lab = [lab]
            padded_label.extend(lab)
        if type(padded_label[0]) is not str:
            padded_label = torch.LongTensor(padded_label)
        #padded_labelにある-1を除去
        #padded_label = padded_label[padded_label != -1]
        return padded_video, video_length, padded_label, label_length,data_path
