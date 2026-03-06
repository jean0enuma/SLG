import torch
from torch.utils.data import Dataset
from loader import *

import torch.nn.functional as F

class SLGText2UnitsDatasets(Dataset):
    """
    phoenixデータセットを読み込むためのクラス(ctc_loss用)
    loaderの出力はdata,targets,input_length,target_length
    data:入力データ(処理後)
    targets:ラベル系列
    input_length:入力データの長さ(torch.tensor, [batch_size])
    target_length:ラベル系列の長さ(torch.tensor, [batch_size])
    """
    def __init__(self, data_path, cod_data_path_root, face_data_path_root, trainable=True,
                 is_processed=False, is_sg_filter=True, is_3d=False, scale_ratio=(0.8, 1.2), min_frame=16, max_frame=300, tokenizer=None, texts_corpus=None):
        super().__init__()
        self.data_path = data_path
        self.face_data_path_root=face_data_path_root
        self.cod_data_path_root=cod_data_path_root
        self.trainable=trainable
        #self.bbox=load_bbox(kind_dataset)
        self.targets_corpus=texts_corpus
        self.scale_ratio=scale_ratio
        self.min_frame=min_frame
        self.max_frame=max_frame
        self.is_processed=is_processed
        self.is_3d=is_3d
        self.is_sg_filter=is_sg_filter
        self.tokenizer=tokenizer

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
    def temporal_rescale(self,data):
        #data:(T,F)
        #動画のフレーム数をランダムに変更する
        #self.min_frame,self.max_frameはフレーム数の最小値と最大値
        #self.scale_ratioはフレーム数の変更率の範囲(例:0.8~1.2)
        scale_ratio=random.uniform(self.scale_ratio[0],self.scale_ratio[1])
        new_length=int(data.shape[0]*scale_ratio)
        new_length=max(self.min_frame,min(self.max_frame,new_length))
        #data=torch.tensor(data)
        if new_length>data.shape[0]:
            data=data.permute(1,0).unsqueeze(1) #(F,1,T)
            try:
                data=F.interpolate(data,size=new_length,mode='linear',align_corners=False) #(F,1,new_length)
            except:
                print(f"Error in F.interpolate: data shape {data.shape}, new_length {new_length}")
                raise ValueError("Error in F.interpolate")
            data=data.squeeze(1).permute(1,0) #(new_length,F)

        else:
            indexes=np.linspace(0,data.shape[0]-1,new_length).astype(int)
            data=data[indexes,:]
        return data
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
            data_path="/".join(data_path.split("/")[:-1])
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
        left_data=hand_cod_data[:, : ,:21].permute(1,2,0).numpy()
        right_data=hand_cod_data[:, : ,21:].permute(1,2,0).numpy()
        pose_data=cod_data.permute(1,2,0).numpy() #(T,JC,2or3)
        pose_data,mask=build_coarse_from_mediapipe(pose_data,left_data,right_data,add_hand_points=None,is_heuristic_feature=True)
        pose_data=concat_coarse_with_interaction(pose_data,left_data,right_data,mask)
        pose_data=torch.tensor(pose_data).float()
        if self.trainable:
            pose_data+=torch.randn_like(pose_data)*0.01
            pose_data*=1+torch.rand(1)*0.05
            #pose_data=self.temporal_rescale(pose_data)
        #hand maskの作成(手の座標が全て0のフレームをbool値でマスクする)
        left_hand_mask=torch.tensor(mask['left_valid']) #(T,)
        right_hand_mask=torch.tensor(mask['right_valid']) #(T,)
        hand_mask=torch.stack([left_hand_mask,right_hand_mask],dim=1) #(T,2)
        #if self.trainable:
        #    cod_data,face_cod_data,hand_cod_data,body_cod_data=self.temporal_rescale(cod_data,face_cod_data,hand_cod_data,body_cod_data)

        # data.size()=(T,C,H,W)
        input_length = torch.tensor(pose_data.shape[0])  # 入力データの長さ
        # ラベル系列の取得
        if self.tokenizer is not None and self.targets_corpus is not None:
            target_corpus = self.targets_corpus[id]
            sequence = target_corpus[target_corpus["id"] == file_name]["annotation"].values[0]
            return pose_data,hand_mask,input_length,id,data_path,sequence
        else:
            return pose_data,hand_mask,input_length,id,data_path
    def collate_fn(self,batch):
        if self.tokenizer is not None and self.targets_corpus is not None:
            cod_data_list,hand_mask_list, input_length_list,id_list,path_list, sentence_list = zip(*batch)
        else:
            cod_data_list,hand_mask_list, input_length_list,id_list,path_list = zip(*batch)
        # 座標データのパディング
        max_length = max([data.shape[0] for data in cod_data_list])
        padded_cod_data = []
        padded_mask=[]
        for data,mask in zip(cod_data_list,hand_mask_list):
            pad_size = max_length - data.shape[0]
            if pad_size > 0:
                data =torch.cat([data, torch.zeros(pad_size, data.shape[1])], dim=0)  # (max_length, F)
                mask=torch.cat([mask,torch.zeros(pad_size,mask.shape[1])],dim=0).bool() #(max_length,2)
            padded_cod_data.append(data)
            padded_mask.append(mask)
        padded_cod_data = torch.stack([data for data in padded_cod_data])
        padded_mask=torch.stack([mask for mask in padded_mask])
        # 入力長のテンソル化
        input_length_tensor = torch.tensor(input_length_list)
        id_list=torch.tensor(id_list)
        if self.tokenizer is not None and self.targets_corpus is not None:
            padded_tokens_tensor = self.tokenizer(
                list(sentence_list),
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            return padded_cod_data,padded_mask, input_length_tensor, id_list, path_list, padded_tokens_tensor
        else:
            return padded_cod_data,padded_mask, input_length_tensor, id_list, path_list

