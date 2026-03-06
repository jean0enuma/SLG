from torch.utils.data import Dataset
from loader import *

from loader import image2video, coordinate_preprocess,all_connections,coordinate_preprocess_3d
import torch.nn.functional as F

class SLG_t2s_datasets(Dataset):
    """
    phoenixデータセットを読み込むためのクラス(ctc_loss用)
    loaderの出力はdata,targets,input_length,target_length
    data:入力データ(処理後)
    targets:ラベル系列
    input_length:入力データの長さ(torch.tensor, [batch_size])
    target_length:ラベル系列の長さ(torch.tensor, [batch_size])
    """
    def __init__(self, data_path, cod_data_path_root, face_data_path_root, targets_corpus, tokenizer, trainable=True,
                 is_processed=False, is_sg_filter=True, is_3d=False, scale_ratio=(0.8, 1.2), min_frame=16, max_frame=300):
        super().__init__()
        self.data_path = data_path
        self.face_data_path_root=face_data_path_root
        self.cod_data_path_root=cod_data_path_root
        self.tokenizer=tokenizer#tokenizerはhuggingfaceのものを想定
        self.trainable=trainable
        #self.bbox=load_bbox(kind_dataset)
        self.targets_corpus=targets_corpus
        self.scale_ratio=scale_ratio
        self.min_frame=min_frame
        self.max_frame=max_frame
        self.is_processed=is_processed
        self.is_3d=is_3d
        self.is_sg_filter=is_sg_filter

    def __len__(self):
        return len(self.data_path)
    def show_max_length(self):
        length_list=[]
        for id,data_path in self.data_path:
            file_name=data_path.split("/")[-1].split(".mp4")[0]
            cod_data_path=f"{self.cod_data_path_root[id]}/{file_name}.csv"
            cod_data=np.loadtxt(f"{cod_data_path}",delimiter=",",dtype=np.float32)
            length_list.append(cod_data.shape[0])
        print(f"Max length: {max(length_list)}, Min length: {min(length_list)}, Average length: {sum(length_list)/len(length_list)}")
        return max(length_list)
    def hand_zero_list(self):
        delete_path_list=[]
        for id,data_path in self.data_path:
            file_name = data_path.split("/")[-1].split(".mp4")[0]
            face_data_path = f"{self.face_data_path_root[id]}/{file_name}.csv"
            cod_data_path = f"{self.cod_data_path_root[id]}/{file_name}.csv"
            face_data = np.loadtxt(f"{face_data_path}", delimiter=",", dtype=np.float32)
            cod_data = np.loadtxt(f"{cod_data_path}", delimiter=",", dtype=np.float32)

            # 座標データの前処理
            try:
                if self.is_3d:
                    cod_data, face_cod_data, hand_cod_data, body_cod_data = coordinate_preprocess_3d(cod_data, face_data,
                                                                                                    is_face_connect=True,is_sg_filter=self.is_sg_filter)
                else:
                    cod_data, face_cod_data, hand_cod_data, body_cod_data = coordinate_preprocess(cod_data, face_data,
                                                                                                  is_face_connect=True,is_sg_filter=self.is_sg_filter)
            except:
                print(f"Error in coordinate_preprocess: {cod_data_path}, {face_data_path}")
                raise ValueError("Error in coordinate_preprocess")
            if cod_data.shape[1]<8:
                delete_path_list.append(data_path)
        return delete_path_list
    def delete_data(self,delete_path_list):
        new_data_path=[]
        data_path=self.data_path
        for id,data_path in data_path:
            if data_path in delete_path_list:
                self.data_path.remove((id,data_path))
        return self.data_path
    def temporal_rescale(self,data,face_data,hand_data,body_data):
        #data:(2,T,F)
        #動画のフレーム数をランダムに変更する
        #self.min_frame,self.max_frameはフレーム数の最小値と最大値
        #self.scale_ratioはフレーム数の変更率の範囲(例:0.8~1.2)
        scale_ratio=random.uniform(self.scale_ratio[0],self.scale_ratio[1])
        new_length=int(data.shape[1]*scale_ratio)
        new_length=max(self.min_frame,min(self.max_frame,new_length))
        #data=torch.tensor(data)
        if new_length>data.shape[1]:
            data=data.permute(2,0,1)  #(F,2,T)
            face_data=face_data.permute(2,0,1)#(F,2,T)
            hand_data=hand_data.permute(2,0,1)  #(F,2,T)
            body_data=body_data.permute(2,0,1)  #(F,2,T)
            try:
                data=F.interpolate(data,size=new_length,mode='linear',align_corners=False) #(F,2,new_length)
            except:
                print(f"Error in F.interpolate: data shape {data.shape}, new_length {new_length}")
                raise ValueError("Error in F.interpolate")
            face_data=F.interpolate(face_data,size=new_length,mode='linear',align_corners=False) #(F,2,new_length).
            hand_data=F.interpolate(hand_data,size=new_length,mode='linear',align_corners=False) #(F,2,new_length)
            body_data=F.interpolate(body_data,size=new_length,mode='linear',align_corners=False) #(F,2,new_length)
            data=data.permute(1,2,0) #(2,new_length,F)
            face_data=face_data.permute(1,2,0) #(2,new_length,F)
            hand_data=hand_data.permute(1,2,0) #(2,new_length,F)
            body_data=body_data.permute(1,2,0) #(2,new_length,F)
        else:
            indexes=np.linspace(0,data.shape[1]-1,new_length).astype(int)
            data=data[:,indexes,:]
            face_data=face_data[:,indexes,:]
            hand_data=hand_data[:,indexes,:]
            body_data=body_data[:,indexes,:]
        return data,face_data,hand_data,body_data
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
        if self.is_3d:
            z_data=data[2]
            return torch.stack([x_data/width,y_data/height,z_data],dim=0)
        else:
            return torch.stack([x_data/width,y_data/height],dim=0)
    def __getitem__(self, idx):
        #self.data_pathはid,pathのタプルorリスト
        #データをロード
        id,data_path=self.data_path[idx]
        if id==3:
            file_name=data_path.split("/")[-2].split(".mp4")[0]
        else:
            file_name=data_path.split("/")[-1].split(".mp4")[0]
        face_data_path=f"{self.face_data_path_root[id]}/{file_name}.csv"
        cod_data_path=f"{self.cod_data_path_root[id]}/{file_name}.csv"
        face_data=np.loadtxt(f"{face_data_path}",delimiter=",",dtype=np.float32)
        cod_data=np.loadtxt(f"{cod_data_path}",delimiter=",",dtype=np.float32)
        if self.is_processed:
            #face_cod_data=coordinate_preprocess_face(face_data)
            T,JC=cod_data.shape
            cod_data=torch.tensor(cod_data).reshape(T,-1,3).permute(2,0,1)#(2,T,JC)
            face_data=torch.tensor(face_data).reshape(T,-1,2).permute(2,0,1)#(2,T,FC)
            hand_cod_data = cod_data[:,:, 6:]
            body_cod_data = cod_data[:, :,:6]
            cod_data=torch.tensor(cod_data).float()
            face_cod_data=torch.tensor(face_data).float()
            hand_cod_data=torch.tensor(hand_cod_data).float()
            body_cod_data=torch.tensor(body_cod_data).float()
        else:
            #座標データの前処理
            try:
                if self.is_3d:
                    cod_data, face_cod_data, hand_cod_data, body_cod_data = coordinate_preprocess_3d(cod_data, face_data,
                                                                                                    is_face_connect=True,is_sg_filter=self.is_sg_filter)
                else:
                    cod_data,face_cod_data,hand_cod_data,body_cod_data=coordinate_preprocess(cod_data,face_data,is_face_connect=True,is_sg_filter=self.is_sg_filter)
            except Exception as e:
                print(f"Exception in coordinate_preprocess: {e}")
                print(f"Error in coordinate_preprocess: {cod_data_path}, {face_data_path}")
                raise ValueError("Error in coordinate_preprocess")
            if id==2:
                cod_data=self.renormalize_skeleton(cod_data)
                face_cod_data=self.renormalize_skeleton(face_cod_data)
                hand_cod_data=self.renormalize_skeleton(hand_cod_data)
                body_cod_data=self.renormalize_skeleton(body_cod_data)
            else:
                cod_data=torch.tensor(cod_data)
                face_cod_data=torch.tensor(face_cod_data)
                hand_cod_data=torch.tensor(hand_cod_data)
                body_cod_data=torch.tensor(body_cod_data)
        #if self.trainable:
        #    cod_data,face_cod_data,hand_cod_data,body_cod_data=self.temporal_rescale(cod_data,face_cod_data,hand_cod_data,body_cod_data)

        all_cod=cod_data
        cod_data=(all_cod,face_cod_data,body_cod_data,hand_cod_data)

        # data.size()=(T,C,H,W)
        input_length = torch.tensor(cod_data[0].shape[1])  # 入力データの長さ
        # ラベル系列の取得
        target_corpus=self.targets_corpus[id]
        sequence = target_corpus[target_corpus["id"] == file_name]["annotation"].values[0]
        return cod_data,input_length,sequence,id,data_path
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
        padded_cod_data = list(torch.stack([data[i] for data in padded_cod_data]) for i in range(4))
        sort_all_connections=sort_connections(all_connections)
        parents = build_parents_from_connections(
            sort_all_connections,
            root=1
        )

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

