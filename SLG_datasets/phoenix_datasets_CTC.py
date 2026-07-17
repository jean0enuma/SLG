import torch
from torch.utils.data import Dataset
from loader import *
from utils import TemporalRescale,process_text
from Parameter.Parameter_phoenix import *
from torch.utils.data import Dataset
from utils.phoenix_cleanup import clean_phoenix_2014_trans
class Phoenix_datasets_CTC(Dataset):
    """
    phoenixデータセットを読み込むためのクラス(ctc_loss用)
    loaderの出力はdata,targets,input_length,target_length
    data:入力データ(処理後)
    targets:ラベル系列
    input_length:入力データの長さ(torch.tensor, [batch_size])
    target_length:ラベル系列の長さ(torch.tensor, [batch_size])
    """
    def  __init__(self, data_path,targets_corpus,gloss2class,transforms,trainable=True,resize=256,ext="png"):
        super().__init__()
        self.data_path = data_path
        self.targets_corpus=targets_corpus
        self.gloss2class=gloss2class
        self.transforms=transforms
        self.trainable=trainable
        self.resize=resize
        self.ext=ext
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        # データの読み込み
        data=image2video(self.data_path[idx],img_size=(self.resize,self.resize),ext=self.ext)
        #data=npy2video(self.data_path[idx])
        data=torch.from_numpy(data).permute(0,3,1,2)
        if self.data_path[idx].split("/")[-1]=="1":
            id = self.data_path[idx].split("/")[-2]
            sequence = self.targets_corpus[self.targets_corpus["id"] == id]["annotation"].values[0]
        else:
            id = self.data_path[idx].split("/")[-1]
            sequence = self.targets_corpus[self.targets_corpus["id"] == id]["annotation"].values[0]
            sequence=clean_phoenix_2014_trans(sequence)

        #data.size()=(T,C,H,W)
        data=self.transforms(data)
        data=data/127.5-1.0
        #data.size()=(T,C,H,W)
        input_length=torch.from_numpy(np.array(len(data)))
        #ラベル系列の取得
        # ラベル系列をクラスに変換
        targets = [self.gloss2class[gloss] for gloss in sequence.split(" ") if gloss in self.gloss2class.keys()]
        # ラベル系列の長さ
        target_length = torch.tensor(len(targets))
        targets = torch.tensor(targets)
        return data, targets, input_length,target_length,id
    def max_input_length(self):
        return max([len(glob.glob(f"{path}/*.png")) for path in self.data_path])
    def average_input_length(self):
        return np.mean([len(glob.glob(f"{path}/*.png")) for path in self.data_path])

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, input_length, label_length, data_path = list(zip(*batch))
        true_length = torch.LongTensor([len(vid) for vid in video])  # ★ 追加: 実動画長 L
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (vid[0][None].expand(left_pad, -1, -1, -1),
                 vid,
                 vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1)),
                dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (vid, vid[-1][None].expand(max_len - len(vid), -1)), dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], data_path, true_length
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            if type(padded_label[0]) is not str:
                padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, data_path, true_length

    @staticmethod
    def collate_fn_nopadding(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, input_length, label_length, data_path = list(zip(*batch))
        # input_lengthが最も長いものに合わせてパディング
        max_len = len(video[0])
        video_length = torch.LongTensor([len(vid) for vid in video])
        video = torch.stack([torch.cat([vid, torch.zeros(max_len - len(vid), *vid.shape[1:])], dim=0) for vid in video])
        # それ以外はそのまま
        input_length = torch.LongTensor(input_length)
        label_length = torch.LongTensor([len(lab) for lab in label])

        if max(label_length) == 0:
            return video, video_length, [], []
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            if type(padded_label[0]) is not str:
                padded_label = torch.LongTensor(padded_label)
            return video, input_length, padded_label,label_length,data_path



