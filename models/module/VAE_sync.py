import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.module.CLIP_Skeleton import SkeletonTextCLIP
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
    def __init__(self, d_model, nhead, dim_feedforward, activation,is_cross_attention=False):
        super(EncoderLayer, self).__init__()
        self.is_cross_attn=is_cross_attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        if is_cross_attention:
            self.cross_attn=nn.MultiheadAttention(d_model, nhead, batch_first=True)
            self.dropout2 = nn.Dropout(0.1)
            self.norm2 = nn.LayerNorm(d_model)

        self.ffn=nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout3=nn.Dropout(0.1)

    def forward(self, q,kv,src_mask=None,q_padding_mask=None,kv_padding_mask=None):
        # q: (batch_size, seq_len_q, d_model)
        # kv: (batch_size, seq_len_kv, d_model)
        # src_mask: (seq_len_q, seq_len_kv) or (batch_size, seq_len_q, seq_len_kv)
        # q_padding_mask: (batch_size, seq_len_q)
        # kv_padding_mask: (batch_size, seq_len_kv)
        # 1. 自己注意
        res=q
        q=self.norm1(q)
        attn_output, _ = self.self_attn(q, q, q, attn_mask=src_mask, key_padding_mask=q_padding_mask)
        q = res + self.dropout1(attn_output)
        if self.is_cross_attn:
            res=q
            q = self.norm2(q)
            q=self.cross_attn(q, kv, kv, attn_mask=src_mask, key_padding_mask=kv_padding_mask)[0]
            q = res + self.dropout2(q)
        res=q
        q = self.norm3(q)
        # 3. FFN
        ffn_output = self.ffn(q)
        q = res + self.dropout3(ffn_output)

        return q
class Encoder(nn.Module):
    def __init__(self,config,is_cross_attention=False):
        super().__init__()
        self.config=config
        d_model=config['d_model']
        nhead=config['nhead']
        num_layers=config['num_layers']
        dim_feedforward=d_model*config['ffn_mult']
        activation=config.get('activation','relu')
        self.query=nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)
        self.encoder=nn.ModuleList([EncoderLayer(d_model,nhead,dim_feedforward,activation,is_cross_attention=True) for _ in range(num_layers)])
    def position_embedding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))  # (d_model/2,)
        pe = torch.zeros(seq_len, d_model, device=device)  # (seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数次元にsin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数次元にcos
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    def forward(self,kv,lgt,src_mask=None,q_padding_mask=None,kv_padding_mask=None):
        #kv: (batch_size, seq_len_kv, d_model)
        query=self.query.expand(kv.size(0), lgt.max().item(), -1)  # (batch_size, max_seq_len, d_model)
        query=query+self.position_embedding(lgt.max().item(), self.config['d_model'], kv.device)  # 位置エンベディングを加算
        for layer in self.encoder:
            query=layer(query,kv,src_mask=src_mask,q_padding_mask=q_padding_mask,kv_padding_mask=kv_padding_mask)
        return query  # (batch_size, max_seq_len, d_model)
class Decoder(nn.Module):
    def __init__(self,config):
        super(Encoder, self).__init__()
        self.config=config
        d_model=config['d_model']
        nhead=config['nhead']
        num_layers=config['num_layers']
        dim_feedforward=d_model*config['ffn_mult']
        activation=config.get('activation','relu')
        self.encoder=nn.ModuleList([EncoderLayer(d_model,nhead,dim_feedforward,activation,is_cross_attention=False) for _ in range(num_layers)])
    def position_embedding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))  # (d_model/2,)
        pe = torch.zeros(seq_len, d_model, device=device)  # (seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数次元にsin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数次元にcos
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    def forward(self,query,kv,lgt,src_mask=None,q_padding_mask=None,kv_padding_mask=None):
        #kv: (batch_size, seq_len_kv, d_model)
        query=query+self.position_embedding(lgt.max().item(), self.config['d_model'], kv.device)  # 位置エンベディングを加算
        for layer in self.encoder:
            query=layer(query,kv,src_mask=src_mask,q_padding_mask=q_padding_mask,kv_padding_mask=kv_padding_mask)
        return query  # (batch_size, max_seq_len, d_model)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        t = t.float()
        emb_scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class VAE_sync(nn.Module):
    def __init__(self,config):
        super(VAE_sync, self).__init__()
        d_model=config['model']['encoder']['d_model']
        z_dim=config['model']['encoder']['z_dim']
        dec_dmodel=config['model']['decoder']['d_model']
        self.config=config
        self.encoder=Encoder(config['model']['encoder'])
        clip=SkeletonTextCLIP(config['model']['skeleton_clip'])
        self.skeleton_embed=clip.skeleton_encoder
        self.text_embed=clip.text_encoder
        for p in self.skeleton_embed.parameters():
            p.requires_grad=False
        for p in self.text_embed.parameters():
            p.requires_grad=False
        self.skeleton_adapter=nn.Sequential(
            nn.Linear(config['model']['skeleton_clip']['proj_dim'], d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.text_adapter=nn.Sequential(
            nn.Linear(config['model']['skeleton_clip']['proj_dim'], d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.encoder=Encoder(config['model']['encoder'])
        self.decoder=Encoder(config['model']['decoder'])
        self.mean_proj = nn.Linear(d_model, z_dim)
        self.logvar_proj = nn.Linear(d_model, z_dim)
        self.z_proj = nn.Linear(z_dim, dec_dmodel)
        self.out_proj=nn.Linear(dec_dmodel, config['model']['pose_dim'])
        self.config=config

    def forward(self,skeleton,skeleton_length,text_inputs):
        skeleton_mask=create_mask(skeleton_length,skeleton.size(1)).to(skeleton.device)  # (batch_size, seq_len_skeleton)
        text_mask=text_inputs['attention_mask'].to(skeleton.device) # (batch_size, seq_len_text)
        text_tokens=text_inputs['input_ids'].to(skeleton.device)  # (batch_size, seq_len_text)
        skeleton_emb=self.skeleton_embed(skeleton,skeleton_mask)["last_hidden_state"]  # (batch_size, seq_len_skeleton, d_model)

        text_emb=self.text_embed(text_tokens,text_mask)["last_hidden_state"]  # (batch_size, seq_len_text, d_model)

        #VAE(skeleton)
        encoded=self.enoder(skeleton_emb,skeleton_length,src_mask=None,q_padding_mask=skeleton_mask,kv_padding_mask=skeleton_mask)  # (batch_size, max_seq_len_skeleton, d_model)
        mean=self.mean_proj(encoded)  # (batch_size, max_seq_len_skeleton, z_dim)
        logvar_s=self.logvar_proj(encoded)  # (batch_size, max_seq_len_skeleton, z_dim)
        std_s=torch.exp(0.5*logvar_s)
        eps_s=torch.randn_like(std_s)
        z=mean+eps_s*std_s  # (batch_size, max_seq_len_skeleton, z_dim)
        z_dec=self.z_proj(z)  # (batch_size, max_seq)
        #Decoder
        text_mask=~(text_mask.bool())  # (batch_size, max_seq_len_text) -> (batch_size, max_seq_len_text)
        decoded=self.decoder(z_dec,text_emb,skeleton_length,src_mask=None,q_padding_mask=skeleton_mask,kv_padding_mask=text_mask)  # (batch_size, max_seq_len_text, dec_dmodel)
        s_out=self.out_proj(decoded)  # (batch_size, max_seq_len_text, pose_dim)
        #VAE(text)
        encoded_text=self.encoder(text_emb,skeleton_length,src_mask=None,q_padding_mask=skeleton_mask,kv_padding_mask=text_mask)  # (batch_size, max_seq_len_text, d_model)
        mean_t=self.mean_proj(encoded_text)  # (batch_size, max_seq_len_text, z_dim)
        logvar_t=self.logvar_proj(encoded_text)  # (batch_size, max_seq_len_text, z_dim)
        std_t=torch.exp(0.5*logvar_t)
        eps_t=torch.randn_like(std_t)
        z_t=mean_t+eps_t*std_t  # (batch_size, max_seq_len_text, z_dim)
        z_t_dec=self.z_proj(z_t)  # (batch_size, max_seq_len_text, dec_dmodel)
        decoded_text=self.decoder(z_t_dec,text_emb,skeleton_length,src_mask=None,q_padding_mask=skeleton_mask,kv_padding_mask=text_mask)  # (batch_size, max_seq_len_text, dec_dmodel)
        t_out=self.out_proj(decoded_text)  # (batch_size, max_seq_len_text, pose_dim)

        #--loss--#
        skeleton_mask=~(skeleton_mask.bool())  # (batch_size, max_seq_len_skeleton) -> (batch_size, max_seq_len_skeleton)
        recon_loss=F.mse_loss(s_out,skeleton, reduction='none').mean(dim=-1)  # (batch_size, max_seq_len_text)
        recon_loss=(recon_loss*skeleton_mask).sum()/skeleton_mask.sum()
        recon_loss_text=F.mse_loss(t_out,skeleton, reduction='none').mean(dim=-1)  # (batch_size, max_seq_len_text)
        recon_loss_text=(recon_loss_text*text_mask).sum()/text_mask.sum()
        kl_loss_s=-0.5*torch.sum(1+logvar_s-mean.pow(2)-logvar_s.exp())/skeleton_mask.sum()
        kl_loss_t=-0.5*torch.sum(1+logvar_t-mean_t.pow(2)-logvar_t.exp())/text_mask.sum()
        #skeletonとテキストの潜在空間を近づけるためのKLダイバージェンス
        kl_loss_s_aux=-0.5*torch.mean(1+logvar_s-logvar_t.detach()-((mean-mean_t.detach()).pow(2)+logvar_s.exp())/logvar_t.exp().detach(),dim=-1)
        kl_loss_s_aux=(kl_loss_s_aux*skeleton_mask).sum()/skeleton_mask.sum()
        kl_loss_t_aux=-0.5*torch.mean(1+logvar_t-logvar_s-((mean_t-mean).pow(2)+logvar_t.exp())/logvar_s.exp(),dim=-1)
        kl_loss_t_aux=(kl_loss_t_aux*text_mask).sum()/text_mask.sum()

        loss=self.config['loss_parameters']['recon_skeleton']*recon_loss+self.config['loss_parameters']['recon_text']*recon_loss_text+self.config['loss_parameters']['kl_skeleton']*kl_loss_s+self.config['loss_parameters']['kl_text']*kl_loss_t+self.config['loss_parameters']['kl_skeleton_aux']*kl_loss_s_aux+self.config['loss_parameters']['kl_text_aux']*kl_loss_t_aux


        return {
            "loss": loss,
            "recon_loss_skeleton": recon_loss,
            "recon_loss_text": recon_loss_text,
            "kl_loss_skeleton": kl_loss_s,
            "kl_loss_text": kl_loss_t,
            "kl_loss_skeleton_aux": kl_loss_s_aux,
            "kl_loss_text_aux": kl_loss_t_aux
        }

