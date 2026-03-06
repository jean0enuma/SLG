from torch.utils.data import Dataset
from loader import *

from loader import image2video, coordinate_preprocess
import torch.nn.functional as F
import torchvision.io

class VideoMAE_datasets(Dataset):
    """
    phoenixデータセットを読み込むためのクラス(ctc_loss用)
    loaderの出力はdata,targets,input_length,target_length
    data:入力データ(処理後)
    targets:ラベル系列
    input_length:入力データの長さ(torch.tensor, [batch_size])
    target_length:ラベル系列の長さ(torch.tensor, [batch_size])
    """
    def __init__(self, data_path,resize=256,trainable=True):
        super().__init__()
        self.data_path = data_path
        self.trainable=trainable
        self.resize=resize
        #self.bbox=load_bbox(kind_dataset)

    def __len__(self):
        return len(self.data_path)
    def renormalize_skeleton(self,data,height=1080,width=1920):
        #data:(2,T,F)
        #画像を正方形にしたとき(短い辺に合わせたとき)の座標に変換
        #dataは0~1に正規化されているとする
        #原点は左上
        x_data=data[0]*width
        y_data=data[1]*height
        if height>width:
            diff=(height-width)/2
            x_data=x_data
            y_data=y_data-diff
        else:
            diff=(width-height)/2
            x_data=x_data-diff
            y_data=y_data
        x_data=torch.tensor(x_data)
        y_data=torch.tensor(y_data)
        return torch.stack([x_data/width,y_data/height],dim=0)
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
        #self.data_pathはid,pathのタプルorリスト
        #データをロード
        id,data_path=self.data_path[idx]
        if id==0:
            data = image2video(data_path, img_size=(self.resize, self.resize), ext="png")
        elif id==2:
            data=video2npy(data_path,img_size=None)
            data=self.clip_square(data,img_size=self.resize)
            #self.show_video(data)
        elif id == 1:
            data = image2video(data_path, img_size=(self.resize, self.resize), ext="jpg")
        else:
            data = video2npy(data_path, img_size=(self.resize, self.resize))
        file_name=data_path.split("/")[-1].split(".mp4")[0]
        data = torch.from_numpy(data).float()
        data = data.permute(0, 3, 1, 2)#(T,C,H,W)
        #dataから，16フレームを連続で抽出
        start_frame=random.randint(0,max(0,data.size(0)-16))
        data=data[start_frame:start_frame+16]
        data=data/127.5-1.0
        return data,data_path
    def collate_fn(self,batch):
        cod_data_list, input_length_list, sentence_list,id_list,path_list = zip(*batch)
        # 座標データのパディング
        max_length = max([data[0].shape[1] for data in cod_data_list])
        padded_cod_data = []
        for data in cod_data_list:
            all_cod, face_cod, body_cod, hand_cod= data
            pad_size = max_length - all_cod.shape[1]
            if pad_size > 0:
                all_cod = F.pad(all_cod, (0, 0, 0, pad_size), mode='constant', value=0)
                face_cod = F.pad(face_cod, (0, 0, 0, pad_size), mode='constant', value=0)
                body_cod = F.pad(body_cod, (0, 0, 0, pad_size), mode='constant', value=0)
                hand_cod = F.pad(hand_cod, (0, 0, 0, pad_size), mode='constant', value=0)
            padded_cod_data.append((all_cod, face_cod, body_cod, hand_cod))
        padded_cod_data = tuple(torch.stack([data[i] for data in padded_cod_data]) for i in range(4))
        # 入力長のテンソル化
        input_length_tensor = torch.tensor(input_length_list)
        padded_tokens_tensor=self.tokenizer(
            list(sentence_list),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        id_list=torch.tensor(id_list)
        return padded_cod_data, input_length_tensor, padded_tokens_tensor, id_list, path_list

