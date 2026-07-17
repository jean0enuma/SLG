import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.module.CLIP_Skeleton import SkeletonTextCLIP
from tqdm import tqdm
import numpy as np
def create_pading_mask(target_length, max_len):
    # target_length: (batch_size,)
    batch_size = target_length.size(0)
    mask = torch.ones((batch_size, max_len), dtype=torch.float32, device=target_length.device)
    for i in range(batch_size):
        mask[i, :target_length[i]] = 0.0
    return mask  # (batch_size, max_len)
def create_attn_mask(query_mask, key_mask):
    # query_mask: (batch_size, seq_len_q) with True for valid positions
    # key_mask: (batch_size, seq_len_k) with True for valid positions
    #return mask: (batch_size, seq_len_q, seq_len_k) with True for positions to mask
    batch_size, seq_len_q = query_mask.size()
    seq_len_k = key_mask.size(1)
    attn_mask = ~(query_mask.unsqueeze(2) & key_mask.unsqueeze(1))
    return attn_mask  # (batch_size, seq_len_q, seq_len_k)
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
        res=q
        q=self.norm1(q)
        attn_output, _ = self.self_attn(q, q, q, attn_mask=src_mask, key_padding_mask=q_padding_mask)
        q = res + self.dropout1(attn_output)
        res=q
        q = self.norm2(q)
        # 2. クロス注意
        attn_output, _ = self.cross_attn(q, kv, kv, attn_mask=None, key_padding_mask=kv_padding_mask)
        q = res + self.dropout2(attn_output)
        res=q
        q = self.norm3(q)
        # 3. FFN
        ffn_output = self.ffn(q)
        q = res + self.dropout3(ffn_output)

        return q
class MHAPooling(nn.Module):
    def __init__(self,d_model):
        super(MHAPooling, self).__init__()
        self.mha=nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
    def forward(self,query,x,L,mask=None):
        # x: (batch_size, seq_len, d_model)
        B,T,D=x.size()
        query=query.expand(B, L, -1)  # (batch_size, L, d_model)
        pooled,_=self.mha(query,x,x,key_padding_mask=mask) #(B,L,D)
        return pooled
class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder, self).__init__()
        self.config=config
        d_model=config['encoder']['d_model']
        nhead=config['encoder']['nhead']
        num_layers=config['encoder']['num_layers']
        dim_feedforward=d_model*config['encoder']['ffn_mult']
        activation=config['encoder'].get('activation','relu')
        dropout=config['encoder'].get('dropout',0.1)
        decoder=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout,activation,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(decoder,num_layers=num_layers)
    def position_embedding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))  # (d_model/2,)
        pe = torch.zeros(seq_len, d_model, device=device)  # (seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数次元にsin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数次元にcos
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    def forward(self,q,src_mask=None,q_padding_mask=None):
        q=q+self.position_embedding(q.size(1), q.size(2), q.device)
        output = self.encoder(q, mask=src_mask, src_key_padding_mask=q_padding_mask)
        return output
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
        #decoder=nn.TransformerDecoderLayer(d_model,nhead,dim_feedforward,dropout,activation,batch_first=True,norm_first=True)
        #self.decoder=nn.TransformerDecoder(decoder,num_layers=num_layers)
    def position_embedding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))  # (d_model/2,)
        pe = torch.zeros(seq_len, d_model, device=device)  # (seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数次元にsin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数次元にcos
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    def forward(self, z,kv=None, src_mask=None, q_padding_mask=None,kv_padding_mask=None):
        # z: (batch_size, seq_len, d_model)
        z=z+self.position_embedding(z.size(1), z.size(2), z.device)
        output = self.decoder(z, mask=src_mask, src_key_padding_mask=q_padding_mask)
        #output=self.decoder(z,kv,kv, tgt_mask=src_mask, memory_mask=None, tgt_key_padding_mask=q_padding_mask, memory_key_padding_mask=kv_padding_mask)
        return output

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

class VAE_class_embed(nn.Module):
    def __init__(self,config):
        super().__init__()
        d_model=config['model']['encoder']['d_model']
        z_dim=config['model']['encoder']['z_dim']
        dec_dmodel=config['model']['decoder']['d_model']
        self.config=config
        clip=SkeletonTextCLIP(config['model']['clip_text'])
        self.skeleton_embed=clip.skeleton_encoder
        self.text_embed=clip.text_encoder
        self.mhapooling=MHAPooling(d_model)
        for p in self.skeleton_embed.parameters():
            p.requires_grad=False
        for p in self.text_embed.embed_model.parameters():
            p.requires_grad=False
        self.skeleton_adapter = nn.Sequential(
            nn.Linear(config['model']['clip_text']['proj_dim'], d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.encoder=Encoder(config['model'])
        self.decoder=Decoder(config['model'])
        self.mean_proj = nn.Linear(d_model, z_dim)
        self.logvar_proj = nn.Linear(d_model, z_dim)
        self.z_proj = nn.Linear(z_dim, dec_dmodel)
        self.out_proj=nn.Linear(dec_dmodel, config['model']['pose_dim'])
        self.config=config
        self.class_embed=None
    def load_class_embed(self, path, device):
        class_embed_vector = np.load(path)  # (num_classes, d_model)
        self.class_embed = torch.from_numpy(class_embed_vector).to(device)  # (num_classes, d_model)
    def create_class_embed(self, ds_train, tokenizer, device, save_path=None):

        with torch.no_grad():
            class_dict={}
            class_embed=[]
            device=device
            i=0
            #各単語ごとのskeleton平均埋め込みベクトルを計算してclass_embedに保存する
            for pose_data,hand_mask,input_length,id,data_path,sequence in tqdm(ds_train, total=len(ds_train)):
                tokens=tokenizer([sequence], padding='max_length', truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt').to(device)
                pose_data=pose_data.to(device)  # (T, J, C)
                input_length.unsqueeze(0).to(device)  # (1,)
                #pose_mask=create_pading_mask(input_length, pose_data.size(0)).bool().to(device)  # (T,)
                text_mask=tokens['attention_mask'].to(device)  # (1, seq_len_text)
                text_tokens=tokens['input_ids'].to(device)  # (1, seq_len_text)
                skeleton_emb=self.skeleton_embed(pose_data.unsqueeze(0),return_sequence=True)["embeddings"]  # (1, d_model)
                if sequence not in class_dict:
                    class_dict[i]=sequence
                    i+=1
                    class_embed.append([skeleton_emb.squeeze(0).cpu()])
                else:
                    class_embed[i].append(skeleton_emb.squeeze(0).cpu())
            class_embed_vector=torch.zeros(len(class_dict), skeleton_emb.size(-1))  # (num_classes, d_model)
            for i in range(len(class_dict)):
                class_embed_vector[i]=torch.stack(class_embed[i], dim=0).mean(dim=0)  # (d_model,)
        class_embed_vector=F.normalize(class_embed_vector, dim=-1)  # (num_classes, d_model)を正規化
        self.class_embed=class_embed_vector.to(device)  # (num_classes, d_model)
        if save_path is not None:
            np.save(f"{save_path}/class_embed.npy", class_embed_vector.cpu().numpy())
    @staticmethod
    def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p, mask=None):
        """
        KL( q || p ) for diagonal Gaussians
        mu/logvar: (B, T, D)
        mask: (B, T) with True for valid positions
        """
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)

        kl = 0.5 * (
                logvar_p - logvar_q +
                (var_q + (mu_q - mu_p).pow(2)) / (var_p + 1e-8) - 1.0
        )  # (B, T, D)

        kl = kl.mean(dim=-1)  # (B, T)

        if mask is not None:
            kl = kl * mask.float()
            kl = kl.sum() / (mask.float().sum() + 1e-8)
        else:
            kl = kl.mean()

        return kl
    def clip_load(self,clip_state_dict):
        #SkeletonTextCLIPの重みから，self.skeleton_embedとself.text_embedにskeleton_encoderとtext_encoderの重みをそれぞれ抽出してロードする
        skeleton_state_dict={}
        text_state_dict={}
        for k,v in clip_state_dict.items():
            if k.startswith("skeleton_encoder."):
                new_k=k.replace("skeleton_encoder.","")
                skeleton_state_dict[new_k]=v
            elif k.startswith("text_encoder."):
                new_k=k.replace("text_encoder.","")
                text_state_dict[new_k]=v
        self.skeleton_embed.load_state_dict(skeleton_state_dict, strict=False)
        self.text_embed.load_state_dict(text_state_dict, strict=False)
    def extract_skeleton_embed(self,text_emb_norm):
        #self.class_embedを用いてtext_embからskeleton_embを抽出する
        #text_emb: (batch_size,t_len,  d_model)(正規化済み)
        #class_embed: (num_classes, d_model)(正規化済み)
        #return: (batch_size, seq_len_text, d_model)
        #text_embとclass_embedのコサイン類似度を計算して，
        #類似度が最も高いクラスのclass_embedをtext_embの各位置に割り当てる
        batch_size, d_model = text_emb_norm.size()
        # コサイン類似度の計算
        similarity = torch.matmul(text_emb_norm, self.class_embed.t())  # (batch_size, num_classes)
        # 最も類似度の高いクラスのインデックスを取得
        _, indices = similarity.max(dim=-1)  # (batch_size,)
        # class_embedから対応するクラスの埋め込みを取得
        skeleton_emb = self.class_embed[indices]  # (batch_size, d_model)
        return skeleton_emb.unsqueeze(1)  # (batch_size, 1, d_model




    def forward(self,skeleton,skeleton_length,text_inputs):
        B,T,J,C=skeleton.size()
        skeleton_mask= create_pading_mask(skeleton_length, skeleton.size(1)).bool().to(skeleton.device)
        text_mask=text_inputs['attention_mask'].to(skeleton.device) # (batch_size, seq_len_text)
        text_tokens=text_inputs['input_ids'].to(skeleton.device)  # (batch_size, seq_len_text)
        with torch.no_grad():
            text_emb=self.text_embed(text_tokens,text_mask)["embeddings"]  # (batch_size, seq_len_text, d_model)
            query = self.skeleton_embed(skeleton, skeleton_mask, return_sequence=True)["last_hidden_state"][:, :1]  # (batch_size, 1, d_model)
        query=query.expand(-1, T, -1)  # (batch_size, seq_len_text, d_model)

        skeleton_emb=self.extract_skeleton_embed(text_emb)
        #VAE(text2skeleton embedding)
        src_mask=None
        skeleton_emb=self.skeleton_adapter(skeleton_emb)  # (batch_size, seq_len_skeleton, d_model)
        skeleton_emb=self.mhapooling(query,skeleton_emb,L=T,mask=None)  # (batch_size, T, d_model)
        encoded=self.encoder(skeleton_emb,src_mask=src_mask,q_padding_mask=skeleton_mask)  # (batch_size, max_seq_len_skeleton, d_model)
        mean=self.mean_proj(encoded)  # (batch_size, max_seq_len_skeleton, z_dim)
        logvar_s=self.logvar_proj(encoded).clamp(-30.0, 20.0)  # (batch_size, max_seq_len_skeleton, z_dim)
        std_s=torch.exp(0.5*logvar_s)
        eps_s=torch.randn_like(std_s)
        z=mean+eps_s*std_s  # (batch_size, max_seq_len_skeleton, z_dim)
        z_dec=self.z_proj(z)  # (batch_size, max_seq)
        #Decoder
        text_mask=~(text_mask.bool())  # (batch_size, max_seq_len_text) -> (batch_size, max_seq_len_text)
        decoded=self.decoder(z_dec,src_mask=src_mask,q_padding_mask=skeleton_mask)  # (batch_size, max_seq_len_text, dec_dmodel)
        s_out=self.out_proj(decoded)  # (batch_size, max_seq_len_text, pose_dim)
        #--loss--#
        skeleton_mask=~(skeleton_mask.bool())  # (batch_size, max_seq_len_skeleton) -> (batch_size, max_seq_len_skeleton)
        skeleton=skeleton.view(B,T,-1)  # (batch_size, seq_len_skeleton, J*C)
        recon_loss=F.smooth_l1_loss(s_out,skeleton.detach(), reduction='none').mean(dim=-1)  # (batch_size, max_seq_len_text)
        recon_loss=(recon_loss*skeleton_mask).sum()/skeleton_mask.sum()
        kl_loss_s=(skeleton_mask*(-0.5*torch.mean(1+logvar_s-mean.pow(2)-logvar_s.exp(),dim=-1))).sum()/skeleton_mask.sum()

        loss=self.config['loss_parameters']['recon_skeleton']*recon_loss+self.config['loss_parameters']['kl_skeleton']*kl_loss_s

        return {
            "loss_total": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss_s,
            "output": s_out,
            "mean": mean,
            "logvar": logvar_s,
            "z": z,
            "length_loss": torch.tensor(0.0, device=skeleton.device)  # 長さの損失はここでは計算しない
        }

