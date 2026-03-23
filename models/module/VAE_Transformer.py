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
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, activation):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn=nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3=nn.Dropout(0.1)

    def forward(self, q,kv, src_mask=None,q_padding_mask=None,kv_padding_mask=None):
        # q: (batch_size, seq_len_q, d_model)
        # kv: (batch_size, seq_len_kv, d_model)
        # src_mask: (seq_len_q, seq_len_kv) or (batch_size, seq_len_q, seq_len_kv)
        # q_padding_mask: (batch_size, seq_len_q)
        # kv_padding_mask: (batch_size, seq_len_kv)
        # 1. 自己注意
        q=self.norm1(q)
        attn_output, _ = self.self_attn(q, q, q, attn_mask=src_mask, key_padding_mask=q_padding_mask)
        q = q + self.dropout1(attn_output)
        q = self.norm2(q)
        # 2. クロス注意
        attn_output, _ = self.cross_attn(q, kv, kv, attn_mask=None, key_padding_mask=kv_padding_mask)
        q = q + self.dropout2(attn_output)
        q = self.norm3(q)
        # 3. FFN
        ffn_output = self.ffn(q)
        q = q + self.dropout3(ffn_output)

        return q
class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder, self).__init__()
        self.config=config
        d_model=config['encoder']['d_model']
        nhead=config['encoder']['nhead']
        num_layers=config['encoder']['num_layers']
        dim_feedforward=d_model*config['encoder']['ffn_mult']
        activation=config['encoder'].get('activation','relu')
        self.encoder=nn.ModuleList([EncoderLayer(d_model,nhead,dim_feedforward,activation) for _ in range(num_layers)])
    def position_embedding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))  # (d_model/2,)
        pe = torch.zeros(seq_len, d_model, device=device)  # (seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数次元にsin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数次元にcos
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    def forward(self,q,kv,src_mask=None,q_padding_mask=None,kv_padding_mask=None):
        q=q+self.position_embedding(q.size(1), q.size(2), q.device)
        for layer in self.encoder:
            q=layer(q,kv,src_mask=src_mask,q_padding_mask=q_padding_mask,kv_padding_mask=kv_padding_mask)
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
class PoseEncoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        input_dim=config['encoder']['kv_input_dim']
        d_model=config['encoder']['d_model']
        self.input_proj=nn.Linear(input_dim,d_model)
        encoder_layer=nn.TransformerEncoderLayer(d_model,config['encoder']['pose_nhead'],d_model*config['encoder']['ffn_mult'],batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(encoder_layer,num_layers=config['encoder']['pose_num_layers'])
    def position_embedding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))  # (d_model/2,)
        pe = torch.zeros(seq_len, d_model, device=device)  # (seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数次元にsin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数次元にcos
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    def forward(self,pose_inputs,src_mask=None,padding_mask=None):
        # pose_inputs: (B, seq_len, joint_num,coord_dim)
        if len(pose_inputs.size())==4:
            B,T,J,C=pose_inputs.size()
            pose_inputs=pose_inputs.view(B,T,-1)  # (B, seq_len, joint_num*coord_dim)
        kv=self.input_proj(pose_inputs)
        kv=kv+self.position_embedding(kv.size(1), kv.size(2), kv.device)
        encoded=self.encoder(kv,mask=src_mask,src_key_padding_mask=padding_mask)
        return encoded
class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_name='openai/clip-vit-base-patch32', d_model=256):
        super(TextEncoder, self).__init__()
        if pretrained_model_name=="openai/clip-vit-base-patch32":
            self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name)
        else:
            self.text_encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.linear = nn.Linear(self.text_encoder.config.hidden_size, d_model)

    def forward(self, input_ids, attention_mask):
        # CLIPのテキストエンコーダーで特徴抽出
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # 線形変換で次元を合わせる
        text_features = self.linear(text_features)
        return text_features
class VAE_Transformer(nn.Module):
    def __init__(self,config):
        super(VAE_Transformer, self).__init__()
        self.config=config
        q_input_dim=config['encoder']['q_input_dim']
        kv_input_dim=config['encoder']['kv_input_dim']
        d_model=config['encoder']['d_model']
        z_dim=config['z_dim']
        dec_dmodel=config['decoder']['d_model']
        self.is_text_cond=config.get('text_cond',False)
        self.encoder=Encoder(config)
        self.decoder=Decoder(config)
        self.mean_proj=nn.Linear(d_model,z_dim)
        self.logvar_proj=nn.Linear(d_model,z_dim)
        self.z_proj=nn.Linear(z_dim,dec_dmodel)
        self.output_fc=nn.Linear(dec_dmodel,kv_input_dim)
        self.anchor_frame=nn.Parameter(torch.randn(1,1,d_model),requires_grad=True)  # (1,1,d_model)
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
    def forward(self,kv_embed,kv_padding_mask,pose_input,pose_length):
        #kv_embed: (B, seq_len_kv, d_model)
        #kv_padding_mask: (B, seq_len_kv)
        #pose_input: (B, seq_len_q, joint_num, coord_dim)
        #pose_length: (B,)
        B,T,J,C=pose_input.size()
        pose_input=pose_input.view(B,T,-1)  # (B, T, joint_num*coord_dim)
        q=self.anchor_frame.expand(B,T,-1)

        src_mask=create_slide_window_mask(T,window_size=self.config['encoder']['window_size'],device=q.device)  # (T, T)
        q_padding_mask=create_mask(pose_length, T)  # (B, T)

        encoded=self.encoder(q,kv_embed,src_mask=src_mask,q_padding_mask=q_padding_mask,kv_padding_mask=kv_padding_mask)

        mean=self.mean_proj(encoded)
        logvar=self.logvar_proj(encoded)
        z=self.reparameterize(mean,logvar)
        z=self.z_proj(z)

        decoded=self.decoder(z,src_mask=src_mask,padding_mask=q_padding_mask)
        output=self.output_fc(decoded)

        recon_loss = F.mse_loss(output, pose_input, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + self.config['kl_weight'] * kl_loss

        return {
            'loss_total': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'output': output

        }
class VAETransformerCond(nn.Module):
    def __init__(self,config):
        super(VAETransformerCond, self).__init__()
        self.config=config
        self.vae_transformer=VAE_Transformer(config)
        self.text_encoder=TextEncoder(pretrained_model_name=config['text_encoder_name'],d_model=config['encoder']['d_model'])
        self.pose_encoder=PoseEncoder(config)
        self.is_text_cond=config.get('text_cond',False)
        if self.is_text_cond:
            for param in self.pose_encoder.parameters():
                param.requires_grad=False
        else:
            for param in self.text_encoder.parameters():
                param.requires_grad=False
    def forward(self,pose_inputs,pose_length,text_inputs=None):
        if self.is_text_cond:
            kv_embed=self.text_encoder(**text_inputs).last_hidden_state  # (B, seq_len, text_d_model)
            text_ids=text_inputs['input_ids']  # (B, seq_len)
            kv_padding_mask=~text_inputs['attention_mask'].bool()  # (B, seq_len)
        else:
            kv_padding_mask=create_mask(pose_length, pose_inputs.size(1))  # (B, seq_len)
            kv_embed=self.pose_encoder(pose_inputs, src_mask=None, padding_mask=kv_padding_mask)  # (B, seq_len, d_model)
        return self.vae_transformer(kv_embed,kv_padding_mask,pose_inputs,pose_length)
