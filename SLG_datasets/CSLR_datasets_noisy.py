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
from pytorchvideo.transforms.transforms import UniformTemporalSubsample
from utils import RandomTimeMasking

class CSLR_datasets_noisy(CSLR_datasets_onlydata):
    def __init__(self,data_path,data_dict,targets_corpus,gloss2class, transforms, pseudo_transforms,truth_transforms,dataset="phoenix", trainable=True, resize=256):
        super().__init__(data_path,data_dict, transforms, trainable, resize)
        self.targets_corpus=targets_corpus
        self.gloss2class=gloss2class
        self.pseudo_transforms=pseudo_transforms
        self.truth_transforms=truth_transforms
        self.timemask=RandomTimeMasking()
        self.dataset=dataset

    def augment_length(self,clip):
        vid_len = len(clip)
        new_len = int(vid_len * (0.8 + (1.2 - 0.8) * np.random.random()))
        if new_len < 32:
            new_len = 32
        if new_len > 230:
            new_len = 230
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        return new_len
    def TemporalRescale(self, clip,new_len):
        vid_len = len(clip)
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
            clip = clip[index]
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
            # clip= clip[index]
            clip = UniformTemporalSubsample(new_len, temporal_dim=0)(clip)
            # clip=self.uniformsubsamplebilinear(clip,new_len)
        return clip
    def no_label_data_process(self,data):
        if data.size()[0] > 300:
            # 300フレームを超える場合はランダムに300フレームをクリップ
            start_idx = np.random.randint(0, data.size()[0] - 300)
            data = data[start_idx:start_idx + 300]
        data = self.pseudo_transforms(data)
        return data
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

        elif dataset_id==2:
            data=video2npy(data_path,img_size=None)
            data=self.clip_square(data,img_size=self.resize)

        elif dataset_id == 3 or dataset_id == 5:
            data = image2video(data_path, img_size=(self.resize, self.resize), ext="jpg")

        elif dataset_id==4:
            keypoints=self.load_keypoints(data_path)
            if keypoints is None:
                data = video2npy(data_path, img_size=(self.resize, self.resize))
                print(f"keypoints not found for {data_path}, using full frame")
            else:
                data = video2npy(data_path, img_size=None)
                data=self.clip_signer(data,keypoints)

            #self.show_video(data)
        else:
            data = video2npy(data_path, img_size=(self.resize, self.resize))

            #self.show_video(data)
        data = torch.from_numpy(data).float()
        data = data.permute(0, 3, 1, 2)
        truth_data=self.truth_transforms(data)


        truth_data= truth_data / 127.5 - 1.0

        if self.dataset=="phoenix":
            if dataset_id==0:
                data = self.transforms(data)
                sequence = self.targets_corpus[self.targets_corpus["id"] == id]["annotation"].values[0]  # ラベル系列の取得
                # ラベル系列をクラスに変換
                targets = [self.gloss2class[gloss] for gloss in sequence.split(" ") if gloss in self.gloss2class.keys()]
                # ラベル系列の長さ
                target_length = torch.tensor(len(targets))
                targets = torch.tensor(targets)
            else:
                data=self.no_label_data_process(data)
                targets = torch.tensor(-1)  # nullを-1で表現
                target_length = torch.tensor(0)
        elif self.dataset=="phoenixT":
            if dataset_id==1:
                data = self.transforms(data)

                sequence = self.targets_corpus[self.targets_corpus["id"] == id]["annotation"].values[0]  # ラベル系列の取得
                # ラベル系列をクラスに変換
                targets = [self.gloss2class[gloss] for gloss in sequence.split(" ") if gloss in self.gloss2class.keys()]
                # ラベル系列の長さ
                target_length = torch.tensor(len(targets))
                targets = torch.tensor(targets)
            else:
                data=self.no_label_data_process(data)
                targets = torch.tensor(-1)  # nullを-1で表現
                target_length = torch.tensor(0)

        elif self.dataset=="CSL-Daily":
            if dataset_id==5:
                data = self.transforms(data)
                id = data_path.split("/")[-1]
                sequence = self.targets_corpus[self.targets_corpus["id"] == id]["annotation"].values[0]  # ラベル系列の取得
                # ラベル系列をクラスに変換
                targets = [self.gloss2class[gloss] for gloss in sequence.split(" ") if gloss in self.gloss2class.keys()]
                # ラベル系列の長さ
                target_length = torch.tensor(len(targets))
                targets = torch.tensor(targets)
            else:
                data=self.no_label_data_process(data)
                targets = torch.tensor(-1)  # nullを-1で表現
                target_length = torch.tensor(0)
        else:
            raise ValueError("dataset must be phoenix or phoenixT")

        data = data / 127.5 - 1.0
        # data.size()=(T,C,H,W)
        if self.trainable==True:
            new_len=self.augment_length(truth_data)
            truth_data=self.TemporalRescale(truth_data,new_len)
            data=self.TemporalRescale(data,new_len)
            if self.dataset=="phoenix":
                if dataset_id!=0:
                    data=self.timemask(data)
            elif self.dataset=="phoenixT":
                if dataset_id!=1:
                    data=self.timemask(data)
            elif self.dataset=="CSL-Daily":
                if dataset_id!=5:
                    data=self.timemask(data)
        input_length = torch.from_numpy(np.array(len(data)))

        return data,truth_data, input_length,  data_path,targets,target_length
    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video,truth_video, input_length,data_path,label,label_length = list(zip(*batch))
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            max_truth_len=len(truth_video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            #video_length = torch.LongTensor([len(vid) for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            right_truth_pad = int(np.ceil(max_truth_len / 4.0)) * 4 - max_truth_len + 6
            #right_pad=0
            max_len = max_len + left_pad + right_pad
            max_truth_len=max_truth_len+left_pad+right_truth_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
            padded_truth_video= [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_truth_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in truth_video]
            padded_truth_video = torch.stack(padded_truth_video)
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
            padded_truth_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in truth_video]
            padded_truth_video = torch.stack(padded_truth_video).permute(0, 2, 1)
        label_length = torch.LongTensor(label_length)
        padded_label = []
        for lab in label:
            if lab.dim() == 0:
                lab = [lab]
            padded_label.extend(lab)
        if type(padded_label[0]) is not str:
            padded_label = torch.Tensor(padded_label).to(torch.int32)
        #padded_labelにある-1を除去
        #padded_label = padded_label[padded_label != -1]
        return padded_video,padded_truth_video, video_length, padded_label, label_length,data_path
