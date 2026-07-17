import torch
from torch.utils.data import Dataset
from loader import *

import torch.nn.functional as F
from transformers import BatchEncoding
from transformers import pipeline
from utils.phoenix_cleanup import clean_phoenix_2014,clean_phoenix_2014_trans

# モデルのロード (BART-baseベースの言い換えモデル)
# GPUがある場合は device=0 を指定

class SLTDatasets(Dataset):
    """
    phoenixデータセットを読み込むためのクラス(ctc_loss用)
    loaderの出力はdata,targets,input_length,target_length
    data:入力データ(処理後)
    targets:ラベル系列
    input_length:入力データの長さ(torch.tensor, [batch_size])
    target_length:ラベル系列の長さ(torch.tensor, [batch_size])
    """
    def __init__(self,
                 data_path,
                 cod_data_path_root,
                 face_data_path_root,
                 trainable=True,
                 is_processed=False,
                 is_sg_filter=True,
                 is_3d=False,
                 scale_ratio=(0.8, 1.2),
                 min_frame=16,
                 max_frame=300,
                 tokenizer=None,
                 texts_corpus=None,
                 gloss2class=None,
                 is_delete_cod=False,
                 is_norm=True,
                 is_openpose=False):
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
        self.return_length=False
        self.gloss2class=gloss2class
        self.is_delete_cod=is_delete_cod
        self.is_norm=is_norm
        self.is_openpose=is_openpose
    def __len__(self):
        return len(self.data_path)
    def set_return_length(self):
        self.return_length=True
    def show_max_length(self):
        length_list=[]
        for id,data_path in self.data_path:
            file_name=data_path.split("/")[-1].split(".mp4")[0]
            cod_data_path=f"{self.cod_data_path_root[id]}/{file_name}.csv"
            cod_data=np.loadtxt(f"{cod_data_path}",delimiter=",",dtype=np.float32)
            length_list.append(cod_data.shape[0])
        print(f"Max length: {max(length_list)}, Min length: {min(length_list)}, Average length: {sum(length_list)/len(length_list)}")
        return max(length_list)

    def  temporal_rescale(self,data):
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
        file_name=data_path.split("/")[-1].split(".mp4")[0]
        video_name=data_path.split("/")[-1]
        if id==3:
            gloss_name= data_path.split("/")[-2]
            face_data_path=f"{self.face_data_path_root[id]}/{gloss_name}/{file_name}.csv"
            cod_data_path=f"{self.cod_data_path_root[id]}/{gloss_name}/{file_name}.csv"
        else:
            face_data_path=f"{self.face_data_path_root[id]}/{file_name}.csv"
            cod_data_path=f"{self.cod_data_path_root[id]}/{file_name}.csv"
        face_data=np.loadtxt(f"{face_data_path}",delimiter=",",dtype=np.float32)
        cod_data=np.loadtxt(f"{cod_data_path}",delimiter=",",dtype=np.float32)
        if self.is_processed:
            #face_cod_data=coordinate_preprocess_face(face_data)
            T,JC=cod_data.shape
            cod_data=torch.tensor(cod_data).reshape(T,-1,3).permute(2,0,1)#(3,T,JC)
            face_data=torch.tensor(face_data).reshape(T,-1,2).permute(2,0,1)#(3,T,FC)
            center_data=cod_data[:, :, 1].clone()#(3,T)
            shoulder_length=torch.sqrt((cod_data[0,:,2]-cod_data[0,:,3])**2+(cod_data[1,:,2]-cod_data[1,:,3])**2+(cod_data[2,:,2]-cod_data[2,:,3])**2)#(T,)
            #cod_data-=center_data.unsqueeze(2)#(3,T,JC)
            #cod_data/=shoulder_length.unsqueeze(0).unsqueeze(2)#(3,T,JC)
            hand_cod_data = cod_data[:,:, 6:]
            body_cod_data = cod_data[:,:, :6]
            body_cod_data=torch.cat([body_cod_data,hand_cod_data[:,:,0:1],hand_cod_data[:,:,21:22]],dim=2)
            body_cod_data-=center_data.unsqueeze(2)
            body_cod_data/=shoulder_length.unsqueeze(0).unsqueeze(2)
            left_center_data=hand_cod_data[:,:,0].clone()
            right_center_data=hand_cod_data[:,:,21].clone()
            left_length=torch.sqrt((hand_cod_data[0,:,0]-hand_cod_data[0,:,5])**2+(hand_cod_data[1,:,0]-hand_cod_data[1,:,5])**2+(hand_cod_data[2,:,0]-hand_cod_data[2,:,5])**2)#(T,)
            right_length=torch.sqrt((hand_cod_data[0,:,21]-hand_cod_data[0,:,26])**2+(hand_cod_data[1,:,21]-hand_cod_data[1,:,26])**2+(hand_cod_data[2,:,21]-hand_cod_data[2,:,26])**2)#(T,)
            hand_cod_data[:,:, :21]-=center_data.unsqueeze(2)
            hand_cod_data[:,:, 21:]-=center_data.unsqueeze(2)
            hand_cod_data[:,:, :21]/=(shoulder_length.unsqueeze(0).unsqueeze(2)/2)
            hand_cod_data[:,:, 21:]/=(shoulder_length.unsqueeze(0).unsqueeze(2)/2)
            cod_data=torch.cat([body_cod_data,hand_cod_data],dim=2)
            #od_data=hand_cod_data
            cod_data=torch.tensor(cod_data).float()
            face_cod_data=torch.tensor(face_data).float()
            hand_cod_data=torch.tensor(hand_cod_data).float()
            body_cod_data=torch.tensor(body_cod_data).float()
        else:
            #座標データの前処理
            if self.is_3d:
                try:
                    if not self.is_openpose:
                        cod_data, face_cod_data, hand_cod_data, body_cod_data = coordinate_preprocess_3d(cod_data,
                                                                                                             face_data,
                                                                                                             is_face_connect=True,
                                                                                                             is_sg_filter=False,
                                                                                                         is_delete_cod=self.is_delete_cod,
                                                                                                            )
                    else:
                        B,JC=cod_data.shape
                        BF,JF=face_data.shape
                        cod_data=cod_data.reshape(B,-1,3).transpose(2,0,1)#(3,T,JC)
                        face_data=face_data.reshape(BF,-1,2).transpose(2,0,1)#(2,T,FC)
                        hand_cod_data=cod_data[:,:,-42:]

                except Exception as e:
                    print(f"Error in coordinate_preprocess_3d: {e}")
                    print(f"cod_data shape: {cod_data.shape}, face_data shape: {face_data.shape}")
                    print(data_path)
                    raise ValueError("Error in coordinate_preprocess_3d",data_path)
            else:
                cod_data, face_cod_data, hand_cod_data, body_cod_data=coordinate_preprocess(cod_data,
                                                                                            face_data,
                                                                                            is_face_connect=True,
                                                                                            is_sg_filter=False,
                                                                                            is_openpose=self.is_openpose)
            # face_cod_data=coordinate_preprocess_face(face_data)
            C,T,J = cod_data.shape
            cod_data = torch.tensor(cod_data)  # (3,T,JC)
            if self.is_norm:
                if self.is_sg_filter:
                    cod_data = cod_data.numpy()
                    cod_data = np.where(cod_data == 0, np.nan, cod_data)
                    cod_data = apply_savgol_filter(cod_data, window_size=7, poly_order=2)
                    cod_data = np.where(np.isnan(cod_data), 0, cod_data)
                    cod_data=torch.tensor(cod_data)
                face_cod_data = torch.tensor(face_cod_data) # (3,T,FC)
                center_data = cod_data[:, :, 1].clone()  # (3,T)
                shoulder_length = torch.sqrt(
                    (cod_data[0, :, 2] - cod_data[0, :, 3]) ** 2 + (cod_data[1, :, 2] - cod_data[1, :, 3]) ** 2 + (
                                cod_data[2, :, 2] - cod_data[2, :, 3]) ** 2)  # (T,)
                # cod_data-=center_data.unsqueeze(2)#(3,T,JC)
                # cod_data/=shoulder_length.unsqueeze(0).unsqueeze(2)#(3,T,JC)
                hand_cod_data = cod_data[:, :, 6:]
                body_cod_data = cod_data[:, :, :6]
                body_cod_data = torch.cat([body_cod_data, hand_cod_data[:, :, 0:1], hand_cod_data[:, :, 21:22]], dim=2)
                body_cod_data -= center_data.unsqueeze(2)
                body_cod_data /= (shoulder_length.unsqueeze(0).unsqueeze(2))+1e-8
                left_center_data = hand_cod_data[:, :, 0].clone()
                right_center_data = hand_cod_data[:, :, 21].clone()
                #left_center_data=center_data.clone()
                #right_center_data=center_data.clone()
                left_length = torch.sqrt((hand_cod_data[0, :, 0] - hand_cod_data[0, :, 5]) ** 2 + (
                            hand_cod_data[1, :, 0] - hand_cod_data[1, :, 5]) ** 2 + (
                                                     hand_cod_data[2, :, 0] - hand_cod_data[2, :, 5]) ** 2)  # (T,)
                right_length = torch.sqrt((hand_cod_data[0, :, 21] - hand_cod_data[0, :, 26]) ** 2 + (
                            hand_cod_data[1, :, 21] - hand_cod_data[1, :, 26]) ** 2 + (
                                                      hand_cod_data[2, :, 21] - hand_cod_data[2, :, 26]) ** 2)  # (T,)
                hand_cod_data[:, :, :21] -= left_center_data.unsqueeze(2)
                hand_cod_data[:, :, 21:] -= right_center_data.unsqueeze(2)
                hand_cod_data[:, :, :21] /= (shoulder_length.unsqueeze(0).unsqueeze(2) / 2)+1e-8
                hand_cod_data[:, :, 21:] /= (shoulder_length.unsqueeze(0).unsqueeze(2) / 2)+1e-8

                #face_cod_data=face_cod_data-center_data.unsqueeze(2)
                #face_cod_data/= (shoulder_length.unsqueeze(0).unsqueeze(2))+1e-8

                #cod_data = torch.cat([body_cod_data,hand_cod_data], dim=2)
                cod_data=hand_cod_data
                cod_data = torch.tensor(cod_data).float()
                face_cod_data = torch.tensor(face_cod_data).float()
                hand_cod_data = torch.tensor(hand_cod_data).float()
                body_cod_data = torch.tensor(body_cod_data).float()
        C,T,J=cod_data.shape
        if self.trainable:
            cod_data = self.temporal_rescale(cod_data.permute(1, 2, 0).reshape(T, -1)).reshape(-1, J, C).permute(2, 0,
                                                                                                            1)  # (3,T,JC)
        pose_data=cod_data.permute(1,2,0) #(T,JC,2or3)
        hand_cod_data=pose_data[:,8:, :].permute(2,0,1) #(3,T,JC-8)
        left_data=hand_cod_data[:, :, :21].permute(1, 2, 0)#(2,T,21)
        right_data=hand_cod_data[:, :, 21:].permute(1, 2, 0)
        left_valid=torch.sum(torch.abs(left_data[:,:,:2]),dim=(1,2))>0 #(T,)
        right_valid=torch.sum(torch.abs(right_data[:,:,:2]),dim=(1,2))>0 #(T,)
        left_valid=left_valid.bool()
        right_valid=right_valid.bool()
        #pose_data=pose_data.reshape(pose_data.shape[0],-1) #(T,JC*2or3)
        mask={"left_valid":left_valid,"right_valid":right_valid}
        #hand maskの作成(手の座標が全て0のフレームをbool値でマスクする)
        left_hand_mask=mask['left_valid'].detach() #(T,)
        right_hand_mask=mask['right_valid'].detach()#(T,)
        hand_mask=torch.stack([left_hand_mask,right_hand_mask],dim=1) #(T,2)
        # output=average_movint(output.reshape(T,J*C),window_size=7).reshape(T,J,C)

        # pose_data[:,29:]も同様
        #hand_mask = (np.max(np.abs(pose_data[:, 29:]), axis=2) < 0.01)
        try:
            # pose_data[:,8:29]で，全てが0.01以下のときは，手が検出されなかったとして，0にする
            hand_mask_l = hand_mask[:, 0]
            hand_mask_r = hand_mask[:, 1]
            pose_data[:, 8:29][~hand_mask_l] = 0.0
            pose_data[:, 29:][~hand_mask_r] = 0.0
        except Exception as e:
            print(f"Error in applying hand mask: {e}")
            print(f"pose_data shape: {pose_data.shape}, hand_mask shape: {hand_mask.shape}")
            raise ValueError("Error in applying hand mask")
            pose_data=torch.tensor(pose_data).float()
        # data.size()=(T,C,H,W)
        input_length = torch.tensor(pose_data.shape[0])  # 入力データの長さ
        # ラベル系列の取得
        if self.tokenizer is not None and self.targets_corpus is not None and self.gloss2class is not None:
            target_corpus = self.targets_corpus[id]
            gloss_seq= target_corpus[target_corpus["id"] == file_name]["annotation"].values[0]
            trans_seq=target_corpus[target_corpus["id"] == file_name]["translation"].values[0]

                #if random.random()<0.5 and self.trainable:
                #    sequence=transform_local(sequence)

            return pose_data,hand_mask,input_length,id,data_path,gloss_seq,trans_seq
        else:
           raise ValueError("tokenizer or targets_corpus is None, cannot get target sequence")
    def collate_fn(self,batch):
        if self.tokenizer is not None and self.targets_corpus is not None and self.gloss2class is not None:
            cod_data_list,hand_mask_list, input_length_list,id_list,path_list, gloss_seq_list,trans_seq_list = zip(*batch)
        else:
            cod_data_list,hand_mask_list, input_length_list,id_list,path_list = zip(*batch)
        # 座標データのパディング
        max_length = max([data.shape[0] for data in cod_data_list])
        padded_cod_data = []
        padded_mask=[]
        for data,mask in zip(cod_data_list,hand_mask_list):
            pad_size = max_length - data.shape[0]
            #data:(T,JC,2or3)
            if pad_size > 0:
                data =torch.cat([data, torch.zeros(pad_size, data.shape[1],data.shape[2])], dim=0)  # (T,JC,2or3)
                mask=torch.cat([mask,torch.zeros(pad_size,mask.shape[1])],dim=0).bool() #(max_length,2)
            padded_cod_data.append(data)
            padded_mask.append(mask)
        padded_cod_data = torch.stack([data for data in padded_cod_data])#(バッチサイズ,T,JC,2or3)
        padded_mask=torch.stack([mask for mask in padded_mask])
        # 入力長のテンソル化
        input_length_tensor = torch.tensor(input_length_list)
        id_list=torch.tensor(id_list)
        if self.tokenizer is not None and self.targets_corpus is not None and self.gloss2class is not None:
            if isinstance(self.tokenizer, dict):
                pad_tokens=self.tokenizer["pad_token"]
                eos_token=self.tokenizer["eos_token"]
                bos_token=self.tokenizer["bos_token"]
                new_sentence_list=[f"bos_token {sentence} eos_token" for sentence in trans_seq_list]
                max_tokens_length=max([len(sentence.split(" ")) for sentence in new_sentence_list])
                input_ids = torch.ones((len(new_sentence_list), max_tokens_length), dtype=torch.long) * pad_tokens
                attention_mask=torch.zeros((len(new_sentence_list), max_tokens_length), dtype=torch.long)
                for i, sentence in enumerate(new_sentence_list):
                    #sentence = clean_phoenix_2014_trans(" ".join(sentence.split(" ")))
                    tokens = sentence.split(" ")
                    if "." in tokens:
                        tokens.remove(".")
                    try:
                        input_ids[i, :len(tokens)] = torch.tensor(
                            [self.tokenizer[tokens[j]] for j in range(len(tokens))],
                            dtype=torch.long,
                        )
                    except KeyError as e:
                        print(f"KeyError: {e} in sentence: {sentence}")
                        raise ValueError(f"KeyError: {e} in sentence: {sentence}")
                    attention_mask[i, :len(tokens)] = 1
                padded_tokens_tensor=BatchEncoding(
                    {"input_ids": input_ids,
                     "attention_mask": attention_mask})
            else:
                padded_tokens_tensor = self.tokenizer(
                    list(trans_seq_list),
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
            gloss_input_ids=torch.zeros((len(gloss_seq_list),max([len(gloss_seq.split(" ")) for gloss_seq in gloss_seq_list])),dtype=torch.long)
            new_gloss_seq_list=[]
            for i,gloss_seq in enumerate(gloss_seq_list):
                tokens=gloss_seq.split(" ")
                if "." in tokens:
                    tokens.remove(".")
                new_gloss_seq_list.append(" ".join(tokens))
                try:
                    gloss_input_ids[i,:len(tokens)]=torch.tensor([self.gloss2class[tokens[j]] for j in range(len(tokens))],dtype=torch.long)
                except KeyError as e:
                    print(f"KeyError: {e} in gloss_seq: {gloss_seq}")
                    raise ValueError(f"KeyError: {e} in gloss_seq: {gloss_seq}")
            gloss_attention_mask=torch.zeros((len(new_gloss_seq_list),max([len(gloss_seq.split(" ")) for gloss_seq in new_gloss_seq_list])),dtype=torch.long)
            for i in range(len(new_gloss_seq_list)):
                gloss_attention_mask[i,:len(new_gloss_seq_list[i].split(" "))]=1
            gloss_padded_tokens_tensor=BatchEncoding(
                {"input_ids": gloss_input_ids,
                 "attention_mask": gloss_attention_mask})

            return padded_cod_data,padded_mask, input_length_tensor, id_list, path_list, padded_tokens_tensor,gloss_padded_tokens_tensor
        else:
            return padded_cod_data,padded_mask, input_length_tensor, id_list, path_list

