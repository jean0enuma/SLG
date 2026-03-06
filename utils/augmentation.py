import random
import numpy as np
from pytorchvideo.transforms.transforms import UniformTemporalSubsample
import torchvision.transforms.functional as F
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
import cv2
import torch
import pandas as pd
class RandomTimeMasking(object):
    """
    動画クリップのランダムな時間領域をマスクする
    連続する5~20フレームをランダムに選択し，その部分を0でマスクする
    """
    def __init__(self,num_mask=2, min_mask_len=5, max_mask_len=10):
        self.num_mask = num_mask
        self.min_mask_len = min_mask_len
        self.max_mask_len = max_mask_len
    def __call__(self,clip):
        vd_len = len(clip)#(T,C,H,W)
        n_mask=random.randint(1,self.num_mask)
        for _ in range(n_mask):
            mask_len = random.randint(self.min_mask_len, self.max_mask_len)
            if vd_len - mask_len <= 0:
                start_frame = 0
            else:
                start_frame = random.randint(0, vd_len - mask_len)
            end_frame = start_frame + mask_len
            clip[start_frame:end_frame] = 0
        return clip
class UniformTemporalSubsampleWithCod(object):
    def __init__(self, num_frames):
        self.num_frames = num_frames
    def __call__(self, clip,hand_cod,face_cod,body_cod,bbox=None):
        #入力をnumpy配列に変換
        clip=clip.numpy()
        #もし，クリップの長さがnum_framesより短い場合は，重複ありで等間隔にサンプリング
        if len(clip) >= self.num_frames:
            idxs = np.linspace(0, len(clip) - 1, self.num_frames).astype(int)
            return clip[idxs],hand_cod[:,idxs],face_cod[:,idxs],body_cod[:,idxs],bbox[idxs] if bbox is not None else None
        #クリップの長さがnum_framesより長い場合は，nanを入れるフレームを等間隔に選択し，それをBilinear補間で補間
        else:
            #clip,had_cod,face_codをpd.DataFrameに変換
            #新しくclipを作成
            new_clip=np.zeros((self.num_frames,clip.shape[1],clip.shape[2],clip.shape[3]))
            #新しくhand_codを作成
            new_hand_cod=np.zeros((2,self.num_frames,hand_cod.shape[2]))
            #新しくface_codを作成
            new_face_cod=np.zeros((2,self.num_frames,face_cod.shape[2]))
            #new_clipへ挿入するフレームのインデックスを計算
            insert_frame=np.linspace(0,self.num_frames-1,len(clip),dtype=np.int64)
            #new_clipへ挿入
            new_clip[insert_frame]=clip
            #new_hand_codへ挿入
            new_hand_cod[:,insert_frame]=hand_cod
            #new_face_codへ挿入
            new_face_cod[:,insert_frame]=face_cod
            #np.arrange(arrange_frame)とinsert_frameの差集合を計算(補間するフレーム)
            dif_frame=np.array(list(set(np.arange(self.num_frames))-set(insert_frame)))
            #dif_frameのフレームをnp.nanにする
            new_clip[dif_frame]=np.nan
            #new_hand_codのフレームをnp.nanにする
            new_hand_cod[:,dif_frame]=np.nan
            #new_face_codのフレームをnp.nanにする
            new_face_cod[:,dif_frame]=np.nan

            new_body_cod=np.zeros((2,self.num_frames,body_cod.shape[2]))
            new_body_cod[:,insert_frame]=body_cod
            new_body_cod[:,dif_frame]=np.nan


            #フレームごとにflatten
            new_clip=new_clip.reshape(-1,new_clip.shape[1]*new_clip.shape[2]*new_clip.shape[3])
            new_hand_cod=new_hand_cod.reshape(self.num_frames,-1)
            new_face_cod=new_face_cod.reshape(self.num_frames,-1)
            new_body_cod=new_body_cod.reshape(self.num_frames,-1)
            #new_clipをpandasのデータフレームに変換
            new_clip=pd.DataFrame(new_clip)
            new_hand_cod=pd.DataFrame(new_hand_cod)
            new_face_cod=pd.DataFrame(new_face_cod)
            new_body_cod=pd.DataFrame(new_body_cod)
            #new_clipの欠損値を線形補間
            new_clip=new_clip.interpolate(method="linear",limit_direction="both",axis=0)
            new_hand_cod=new_hand_cod.interpolate(method="linear",limit_direction="both",axis=0)
            new_face_cod=new_face_cod.interpolate(method="linear",limit_direction="both",axis=0)
            new_body_cod=new_body_cod.interpolate(method="linear",limit_direction="both",axis=0)
            #new_clipをnumpy配列に変換
            new_clip=new_clip.values
            new_hand_cod=new_hand_cod.values
            new_face_cod=new_face_cod.values
            new_body_cod=new_body_cod.values
            #new_clipをフレームごとにreshape
            new_clip=new_clip.reshape(-1,clip.shape[1],clip.shape[2],clip.shape[3])
            new_hand_cod=new_hand_cod.reshape(2,self.num_frames,hand_cod.shape[2])
            new_face_cod=new_face_cod.reshape(2,self.num_frames,face_cod.shape[2])
            new_body_cod=new_body_cod.reshape(2,self.num_frames,body_cod.shape[2])
            new_clip=new_clip.astype(np.uint8)

            if bbox is not None:
                new_bbox=np.zeros((self.num_frames,bbox.shape[1],bbox.shape[2]))
                new_bbox[insert_frame]=bbox
                new_bbox[dif_frame]=np.nan
                new_bbox=pd.DataFrame(new_bbox)
                new_bbox=new_bbox.interpolate(method="linear",limit_direction="both",axis=0)
                new_bbox=new_bbox.values
            else:
                new_bbox=None
            return torch.tensor(new_clip),new_hand_cod,new_face_cod,new_body_cod,new_bbox




class ClipFrame(object):
    def __init__(self, num_frames):
        self.num_frames = num_frames
    def __call__(self, clip):
        vd_len = len(clip)
        if len(clip)<self.num_frames:
            return clip
        start_frame= random.randint(0, vd_len - self.num_frames)
        end_frame = start_frame + self.num_frames
        return clip[start_frame:end_frame]
class ArrangeFrame(object):
    def __init__(self, num_frames):
        self.num_frames = num_frames
    def __call__(self, clip):
        vd_len = len(clip)
        frame_idx= np.linspace(0, vd_len - 1, self.num_frames).astype(int)
        return clip[frame_idx]
class StrideFrame(object):
    def __init__(self,stride=3):
        self.stride = stride
    def __call__(self, clip):
        vd_len = len(clip)
        frame_idx= np.arange(0, vd_len, self.stride).astype(int)
        return clip[frame_idx]
class StrideSplitFrame(object):
    def __init__(self,stride=3,num_seq=8,seq_len=4):
        self.stride = stride
        self.num_seq=num_seq
        self.seq_len=seq_len
    def __call__(self,clip):
        vd_len = len(clip)
        if vd_len<=self.num_seq*self.seq_len*self.stride:
            frame_idx= np.linspace(0, vd_len - 1, self.num_seq*self.seq_len).astype(int)
            return clip[frame_idx]
        else:
            start_idx=np.random.choice(range(vd_len-self.num_seq*self.seq_len*self.stride),1)
            seq_idx=np.arange(self.num_seq*self.seq_len)*self.stride+start_idx
            return clip[seq_idx]
class MinFrameNearestSubsample(object):
    def __init__(self, min_frame):
        self.min_frame = min_frame
    def __call__(self, clip):
        vd_len= len(clip)
        if vd_len<=self.min_frame:
            clip=UniformTemporalSubsample(self.min_frame,temporal_dim=0)(clip)
        return clip
class MinMaxFrameNearestSubsample(object):
    def __init__(self, min_frame,max_frame):
        self.min_frame = min_frame
        self.max_frame = max_frame
    def __call__(self, clip):
        vd_len= len(clip)
        if vd_len<=self.min_frame:
            clip=UniformTemporalSubsample(self.min_frame,temporal_dim=0)(clip)
        elif vd_len>self.max_frame:
            clip=UniformTemporalSubsample(self.max_frame,temporal_dim=0)(clip)
        return clip
class StepFrame(object):
    def __init__(self, step=4):
        self.step = step
    def __call__(self, clip):
        vd_len = len(clip)
        if vd_len <= self.step:
            return clip
        frame_idx = np.arange(0, vd_len, self.step).astype(int)
        return clip[frame_idx]

class TemporalRescale(object):
    def __init__(self, max_len=230,min_len=32,temp_scaling=0.2, frame_interval=1):
        self.min_len = min_len
        self.max_len = int(np.ceil(max_len/frame_interval))
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling
    def uniformsubsamplebilinear(self,clip,new_len):
        new_clip = np.zeros((new_len, clip.shape[1], clip.shape[2], clip.shape[3]))
        insert_frame = np.linspace(0, new_len - 1, len(clip), dtype=np.int64)
        new_clip[insert_frame] = clip
        dif_frame = np.array(list(set(np.arange(new_len)) - set(insert_frame)))
        new_clip[dif_frame] = np.nan
        new_clip = new_clip.reshape(-1, new_clip.shape[1] * new_clip.shape[2] * new_clip.shape[3])
        new_clip = pd.DataFrame(new_clip)
        new_clip = new_clip.interpolate(method="linear", limit_direction="both", axis=0)
        new_clip = new_clip.values
        new_clip = new_clip.reshape(-1, clip.shape[1], clip.shape[2], clip.shape[3])
        new_clip = new_clip.astype(np.uint8)
        new_clip = torch.tensor(new_clip)
        return new_clip

    def __call__(self, clip):
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
            clip = clip[index]
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
            #clip= clip[index]
            clip=UniformTemporalSubsample(new_len,temporal_dim=0)(clip)
            #clip=self.uniformsubsamplebilinear(clip,new_len)
        return clip


class TemporalRescaleWithCod(object):
    def __init__(self, max_len=230,min_len=32,temp_scaling=0.2):
        self.min_len = min_len
        self.max_len = max_len
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling
    def __call__(self, clip,hand_cod,face_cod,body_cod,bbox=None):
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return clip[index],hand_cod[:,index],face_cod[:,index],body_cod[:,index],bbox[index] if bbox is not None else None

class ResizeWithCod(object):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True):
        self.size = size
        self.max_size = max_size
        self.antialias = antialias
        self.interpolation = interpolation
    def __call__(self,clip,hand_cod,face_cod,body_cod):
        return clip,hand_cod,face_cod,body_cod

class RandomCropWithCod(object):
    """
    pytorchのtransformに適用できるランダムクロップ
    骨格座標をクロッピングに対応させる
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, clip,hand_cod,face_cod,body_cod,bbox=None):
        #clip: torch.tensor(*,C,H,W)
        #cod:torch.tensor(2,T,F)
        h, w = clip.size(-2), clip.size(-1)
        hand_cod[0]=hand_cod[0]*w
        hand_cod[1]=hand_cod[1]*h
        face_cod[0]=face_cod[0]*w
        face_cod[1]=face_cod[1]*h
        body_cod[0]=body_cod[0]*w
        body_cod[1]=body_cod[1]*h
        if bbox is not None:
            bbox[:,:,0:2]*=w
            bbox[:,:,2:4]*=h
        th, tw = self.size
        if w == tw and h == th:
            return clip,hand_cod,face_cod,body_cod,bbox

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        #座標データのクロッピング(座標は0-1の値)
        hand_cod[0]=hand_cod[0]-x1
        hand_cod[1]=hand_cod[1]-y1
        face_cod[0]=face_cod[0]-x1
        face_cod[1]=face_cod[1]-y1
        body_cod[0]=body_cod[0]-x1
        body_cod[1]=body_cod[1]-y1
        hand_cod[0]=hand_cod[0]/tw
        hand_cod[1]=hand_cod[1]/th
        face_cod[0]=face_cod[0]/tw
        face_cod[1]=face_cod[1]/th
        body_cod[0]=body_cod[0]/tw
        body_cod[1]=body_cod[1]/th
        if bbox is not None:
            bbox[:,:,0:2]-=x1
            bbox[:,:,2:4]-=y1
            bbox[:,:,0:2]/=tw
            bbox[:,:,2:4]/=th
        return clip[...,y1:y1 + th, x1:x1 + tw] ,hand_cod,face_cod,body_cod,bbox
class CenterCropWithCod(object):
    """
    pytorchのtransformに適用できるセンタークロップ
    骨格座標をクロッピングに対応させる
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, clip,hand_cod,face_cod,body_cod,bbox=None):
        #clip: torch.tensor(*,C,H,W)
        #cod:torch.tensor(2,T,F)
        h, w = clip.size(-2), clip.size(-1)
        hand_cod[0] = hand_cod[0] * w
        hand_cod[1] = hand_cod[1] * h
        face_cod[0] = face_cod[0] * w
        face_cod[1] = face_cod[1] * h
        body_cod[0] = body_cod[0] * w
        body_cod[1] = body_cod[1] * h
        if bbox is not None:
            bbox[:, :, 0:2] *= w
            bbox[:, :, 2:4] *= h

        th, tw = self.size
        if w == tw and h == th:
            return clip,hand_cod,face_cod,body_cod,bbox

        x1 = (w - tw) // 2
        y1 = (h - th) // 2
        #座標データのクロッピング(座標は0-1の値)
        hand_cod[0] = hand_cod[0] - x1
        hand_cod[1] = hand_cod[1] - y1
        face_cod[0] = face_cod[0] - x1
        face_cod[1] = face_cod[1] - y1
        body_cod[0] = body_cod[0] - x1
        body_cod[1] = body_cod[1] - y1
        hand_cod[0] = hand_cod[0] / tw
        hand_cod[1] = hand_cod[1] / th
        face_cod[0] = face_cod[0] / tw
        face_cod[1] = face_cod[1] / th
        body_cod[0] = body_cod[0] / tw
        body_cod[1] = body_cod[1] / th
        if bbox is not None:
            bbox[:,:,0:2]-=x1
            bbox[:,:,2:4]-=y1
            bbox[:,:,0:2]/=tw
            bbox[:,:,2:4]/=th

        return clip[...,y1:y1 + th, x1:x1 + tw],hand_cod,face_cod,body_cod,bbox

class RandomHorizontalFlipWithCod(object):
    def __init__(self,flip_prob=0.5):
        self.flip_prob=flip_prob
    def __call__(self, data,hand_data,face_data,body_cod,bbox=None):
        if random.random() < self.flip_prob:
            data=torch.flip(data,dims=[3])
            hand_data[0] = (hand_data[0] - 0.5) * (-1) + 0.5
            face_data[0] = (face_data[0] - 0.5) * (-1) + 0.5
            #右手と左手が反転しているので，座標データでもそれを反映させる
            tmp=hand_data[:,:,21:].copy()
            hand_data[:,:,21:]=hand_data[:,:,:21]
            hand_data[:,:,:21]=tmp

            tmp=body_cod[:,:,[2,4,6]]
            body_cod[:,:,[2,4,6]]=body_cod[:,:,[3,5,7]]
            body_cod[:,:,[3,5,7]]=tmp
            if bbox is not None:
                tmp = bbox[:, 1].copy()
                bbox[:, 1] = bbox[:, 2]
                bbox[:, 2] = tmp

        return data,hand_data,face_data,body_cod,bbox

class RandomCropWithOptical(object):
    """
    pytorchのtransformに適用できるランダムクロップ
    骨格座標をクロッピングに対応させる
    """

    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, clip,o_clip):
        # clip: torch.tensor(*,C,H,W)
        # cod:torch.tensor(2,T,F)
        h, w = clip.size(-2), clip.size(-1)
        oh, ow = o_clip.size(-2), o_clip.size(-1)
        th, tw = (self.size[0],self.size[0])if len(self.size) > 1 else (self.size, self.size)
        oth, otw = (self.size[1],self.size[1]) if len(self.size) > 1 else (th, tw)
        if w == tw and h == th:
            return clip, o_clip

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        ox1= random.randint(0, ow - otw)
        oy1= random.randint(0, oh - oth)
        # 座標データのクロッピング(座標は0-1の値)

        return clip[..., y1:y1 + th, x1:x1 + tw], o_clip[..., oy1:oy1 + oth, ox1:ox1 + otw]
class CenterCropWithOptical(object):
    """
    pytorchのtransformに適用できるセンタークロップ
    骨格座標をクロッピングに対応させる
    """

    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, clip, o_clip):
        # clip: torch.tensor(*,C,H,W)
        # cod:torch.tensor(2,T,F)
        h, w = clip.size(-2), clip.size(-1)
        oh, ow = o_clip.size(-2), o_clip.size(-1)
        th, tw = (self.size[0], self.size[0]) if len(self.size) > 1 else (self.size, self.size)
        oth, otw = (self.size[1], self.size[1]) if len(self.size) > 1 else (th, tw)
        if w == tw and h == th:
            return clip, o_clip

        x1 = (w - tw) // 2
        y1 = (h - th) // 2
        ox1 = (ow - otw) // 2
        oy1 = (oh - oth) // 2
        # 座標データのクロッピング(座標は0-1の値)

        return clip[..., y1:y1 + th, x1:x1 + tw], o_clip[..., oy1:oy1 + oth, ox1:ox1 + otw]
class ResizeWithOptical(object):
    def __init__(self, size,interpolation=InterpolationMode.BILINEAR,max_size=None,antialias=True):
        self.resize=Resize(size,interpolation,max_size,antialias)
    def __call__(self,clip,o_clip,mask_clip=None):
        return self.resize(clip),self.resize(o_clip),self.resize(mask_clip) if mask_clip is not None else None
class RandomHorizontalFlipWithOptical(object):
    def __init__(self,flip_prob=0.5):
        self.flip_prob=flip_prob
    def __call__(self, clip, o_clip):
        if random.random() < self.flip_prob:
            clip=torch.flip(clip, dims=[3])
            o_clip=torch.flip(o_clip, dims=[3])
        return clip,o_clip

class TemporalRescaleWithOptical(object):
    def __init__(self, max_len=230,min_len=32,temp_scaling=0.2, frame_interval=1):
        self.min_len = min_len
        self.max_len = int(np.ceil(max_len/frame_interval))
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling
    def __call__(self, clip,o_clip):
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return clip[index], o_clip[index]


