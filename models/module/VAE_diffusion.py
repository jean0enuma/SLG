from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from Parameter.Parameter import *
import numpy as np
import math
from transformers import AutoModel
from transformers import CLIPTextModel
def create_mask(target_length, max_len):
    # target_length: (batch_size,)
    batch_size = target_length.size(0)
    mask = torch.ones((batch_size, max_len), dtype=torch.float32, device=target_length.device)
    for i in range(batch_size):
        mask[i, :target_length[i]] = 0.0
    return mask  # (batch_size, max_len)
def create_slide_window_mask(length,window_size,device):
    mask=torch.ones((length, length), dtype=torch.bool, device=device)
    for i in range(length):
        start=max(0,i-window_size//2)
        end=min(length,i+window_size//2+1)
        mask[i,start:end]=False
    return mask  # (length, length)
class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder, self).__init__()
        self.config=config
        d_model=config['encoder']['d_model']
        nhead=config['encoder']['nhead']
        num_layers=config['encoder']['num_layers']
        dim_feedforward=d_model*config['encoder']['ffn_mult']
        activation=config['encoder'].get('activation','relu')
        enc_layer=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout=0.1,activation=activation,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(enc_layer,num_layers=num_layers)
    def position_embedding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))  # (d_model/2,)
        pe = torch.zeros(seq_len, d_model, device=device)  # (seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数次元にsin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数次元にcos
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    def forward(self,q,src_mask=None,q_padding_mask=None):
        q=q+self.position_embedding(q.size(1), q.size(2), q.device)
        q=self.encoder(q, mask=src_mask, src_key_padding_mask=q_padding_mask)
        return q
class Decoder(nn.Module):
    def __init__(self,config):
        super(Decoder, self).__init__()
        #VAEのDecoderはTransformerのDecoderと同じ構造を持つ
        #ただし、出力は潜在空間の次元数になるようにする
        d_model=config['decoder']['d_model']
        nhead=config['decoder']['nhead']
        num_layers=config['decoder']['num_layers']
        dim_feedforward=d_model*config['decoder']['ffn_mult']
        activation=config['decoder'].get('activation','relu')
        dropout=config['decoder'].get('dropout',0.1)
        decoder=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout,activation,batch_first=True,norm_first=True)
        self.decoder=nn.TransformerEncoder(decoder,num_layers=num_layers)
    def position_embedding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))  # (d_model/2,)
        pe = torch.zeros(seq_len, d_model, device=device)  # (seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数次元にsin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数次元にcos
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    def forward(self, z, src_mask=None, padding_mask=None):
        # z: (batch_size, seq_len, d_model)
        z=z+self.position_embedding(z.size(1), z.size(2), z.device)
        output = self.decoder(z, mask=src_mask, src_key_padding_mask=padding_mask)
        return output

class DiffusionModel(nn.Module):
    #TODO: conditionを使った拡散モデルの実装(CFG付き)
    def __init__(self,config):
        super(DiffusionModel, self).__init__()
        self.config=config
        #拡散モデルの定義
    def forward(self,z,text_inputs,video_inputs):
        #z: (B, seq_len, z_dim)
        #text_inputs: CLIPのテキストエンコーダーの入力形式
        #video_inputs: 動画特徴量の入力形式
        #拡散モデルの順伝播
        return z
    @torch.no_grad()
    def ddim_sample(self,condition,num_steps):
        #conditionを使ったDDIMサンプリングの実装
        return condition
    @torch.no_grad()
    def ddpm_sample(self,condition,num_steps):
        #conditionを使ったDDPMサンプリングの実装
        return condition
class VAETransformerDiffusion(nn.Module):
    def __init__(self,config):
        super(VAETransformerDiffusion, self).__init__()
        self.config=config
        q_input_dim=config['encoder']['q_input_dim']
        kv_input_dim=config['encoder']['kv_input_dim']
        d_model=config['encoder']['d_model']
        z_dim=config['encoder']['z_dim']
        dec_dmodel=config['decoder']['d_model']
        self.is_diffusion=config.get('diffusion',False)
        self.input_predictor=nn.Linear(q_input_dim, 1)
        self.encoder=Encoder(config)
        self.decoder=Decoder(config)
        #TODO: Diffusionモデルの定義
        #self.diffusion=DiffusionModel(config)
        self.mean_proj=nn.Linear(d_model,z_dim)
        self.logvar_proj=nn.Linear(d_model,z_dim)
        self.z_proj=nn.Linear(z_dim,dec_dmodel)
        self.input_proj=nn.Linear(q_input_dim, d_model)
        self.output_fc=nn.Linear(dec_dmodel,kv_input_dim)
        self.body_weight=config.get('body_weight',0.3)
        self.hand_weight=config.get('hand_weight',0.7)
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    def encode(self,q,kv,src_mask=None,q_padding_mask=None,kv_padding_mask=None):
        encoded=self.encoder(q,kv,src_mask=src_mask,q_padding_mask=q_padding_mask,kv_padding_mask=kv_padding_mask)
        mean=self.mean_proj(encoded)
        logvar=self.logvar_proj(encoded)
        return mean,logvar
    def decode(self,z,src_mask=None,padding_mask=None):
        decoded=self.decoder(z,src_mask=src_mask,padding_mask=padding_mask)
        return decoded
    def forward(self,pose_input,pose_length,text_inputs,video_inputs):
        #kv_embed: (B, seq_len_kv, d_model)
        #kv_padding_mask: (B, seq_len_kv)
        #pose_input: (B, seq_len_q, joint_num, coord_dim)
        #pose_length: (B,)
        B, T, J, C = pose_input.size()
        body_coord = pose_input[:, :, :8]
        hand_coord = pose_input[:, :, 8:]

        # pose_input=pose_input.view(B,T,-1)  # (B, T, joint_num*coord_dim)
        q = pose_input.reshape(B, T, -1)  # (B, T, joint_num*coord_dim)

        src_mask = create_slide_window_mask(T, window_size=self.config['encoder']['window_size'],
                                            device=q.device)  # (T, T)
        q_padding_mask = create_mask(pose_length, T)  # (B, T)
        q=self.input_proj(q)  # (B, T, d_model)
        encoded=self.encoder.forward(q,src_mask=src_mask,q_padding_mask=q_padding_mask)

        mean=self.mean_proj(encoded)
        logvar=self.logvar_proj(encoded)
        z=self.reparameterize(mean,logvar)
        if self.is_diffusion:
            #TODO: conditionを使った拡散モデルの実装(CFG付き)
            diff_out=self.diffusion(z,text_inputs,video_inputs)
            #TODO: 拡散モデルの損失関数の実装
            #diffusion_loss=#
            #return diffusion_loss #DIffusionモデルの損失
        z=self.z_proj(z)

        decoded=self.decoder(z,src_mask=src_mask,padding_mask=q_padding_mask)
        output=self.output_fc(decoded)
        output=output.view(B,T,J,C)  # (B, T, joint_num, coord_dim)

        body_output=output[:,:,:8]
        hand_output=output[:,:,8:]
        body_recon_loss=F.smooth_l1_loss(body_output, body_coord, reduction='none')  # (B, T, 8, coord_dim)
        hand_recon_loss=F.smooth_l1_loss(hand_output, hand_coord, reduction='none')  # (B, T, joint_num-8, coord_dim)
        q_padding_mask=(~q_padding_mask.bool()).long()
        body_recon_loss = (body_recon_loss.mean(dim=[2,3])*q_padding_mask).sum()/q_padding_mask.sum()
        hand_recon_loss = (hand_recon_loss.mean(dim=[2,3])*q_padding_mask).sum()/q_padding_mask.sum()
        recon_loss=self.body_weight*body_recon_loss+self.hand_weight*hand_recon_loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())


        loss = self.config.get('recon_weight',1.0)*recon_loss + self.config['kl_weight'] * kl_loss

        return {
            'loss_total': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'output': output,
        }
