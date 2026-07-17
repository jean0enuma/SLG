import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.module.CLIP_Skeleton import SkeletonTextCLIP
from models.module.AdaLN import AdaLN
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
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, activation):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn=nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm1_1= nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        self.norm3_1=nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3=nn.Dropout(0.1)

    def forward(self, q,kv, src_mask=None,q_padding_mask=None,kv_padding_mask=None,scale=None,shift=None,gate=None):
        # q: (batch_size, seq_len_q, d_model)
        # kv: (batch_size, seq_len_kv, d_model)
        # src_mask: (seq_len_q, seq_len_kv) or (batch_size, seq_len_q, seq_len_kv)
        # q_padding_mask: (batch_size, seq_len_q)
        # kv_padding_mask: (batch_size, seq_len_kv)
        # 1. 自己注意
        res=q
        q=self.norm1(q)
        if scale is not None and shift is not None and gate is not None:
            q=q*scale[:,:1]+shift[:,:1]
            attn_output, _ = self.self_attn(q, q, q, attn_mask=src_mask, key_padding_mask=q_padding_mask)
            attn_output = torch.nan_to_num(attn_output, nan=0.0)
            q = res + self.norm1_1(self.dropout1(attn_output))*gate[:,:1]
        else:
            attn_output, _ = self.self_attn(q, q, q, attn_mask=src_mask, key_padding_mask=q_padding_mask)
            attn_output = torch.nan_to_num(attn_output, nan=0.0)
            q = res + self.norm1_1(self.dropout1(attn_output))
        res=q
        q = self.norm2(q)
        # 2. クロス注意
        attn_output, _ = self.cross_attn(q, kv, kv, attn_mask=None, key_padding_mask=kv_padding_mask)
        attn_output = torch.nan_to_num(attn_output, nan=0.0)
        q = res + self.dropout2(attn_output)
        res=q
        q = self.norm3(q)
        if scale is not None and shift is not None and gate is not None:
            q=torch.nan_to_num(q*scale[:,1:]+shift[:,1:], nan=0.0)
            # 3. FFN
            ffn_output = self.ffn(q)
            q = res + self.norm3_1(self.dropout3(ffn_output))*gate[:,1:]
        else:
            ffn_output = self.ffn(q)
            q = res + self.norm3_1(self.dropout3(ffn_output))

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
        #SELF.pe=nn.Parameter(self.position_embedding(config['model']['max_seq_len'], d_model), requires_grad=True)
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
        cond_dim=config['clip_text']['proj_dim']

        self.shared_adaln=AdaLN(d_model, time_dim=cond_dim)
        self.shared_adaln2=AdaLN(d_model, time_dim=cond_dim)
        self.decoder=nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward, activation) for _ in range(num_layers)])
    def position_embedding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))  # (d_model/2,)
        pe = torch.zeros(seq_len, d_model, device=device)  # (seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数次元にsin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数次元にcos
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    def forward(self, z, kv,src_mask=None, q_padding_mask=None,cond=None,kv_padding_mask=None):
        # z: (batch_size, seq_len, d_model)
        z=z+self.position_embedding(z.size(1), z.size(2), z.device)
        if cond is not None:
            scale1, shift1,gate1 = self.shared_adaln(cond)
            scale2, shift2,gate2=self.shared_adaln2(cond)
            scale=torch.cat([scale1,scale2],dim=1)
            shift=torch.cat([shift1,shift2],dim=1)
            gate=torch.cat([gate1,gate2],dim=1)
        else:
            scale, shift,gate=None,None,None
        for layer in self.decoder:
            z =layer(z,kv, src_mask=src_mask, q_padding_mask=q_padding_mask,kv_padding_mask=kv_padding_mask,scale=scale,shift=shift,gate=gate)
        return z

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
        clip=SkeletonTextCLIP(config['model']['clip_text'])

        #self.skeleton_embed=clip.skeleton_encoder
        self.skeleton_embed=nn.Sequential(
            nn.Linear(config['model']['pose_dim'],d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.text_embed=clip.text_encoder
        hidden_size=self.text_embed.embed_model.config.hidden_size if config['model']['clip_text']['text_encoder_name']!="scratch" else d_model
        config['model']['clip_text']['proj_dim']=hidden_size
        self.mhapooling=MHAPooling(d_model)
        #for p in self.skeleton_embed.parameters():
        #    p.requires_grad=False
        if config['model']['clip_text']['text_encoder_name']!="scratch":
            for p in self.text_embed.embed_model.parameters():
                p.requires_grad=False
        #self.skeleton_adapter=nn.Sequential(
        #    nn.Linear(config['model']['clip_text']['proj_dim'], d_model),
        #    nn.ReLU(),
        #    nn.Linear(d_model, d_model)
        #)
        self.text_adapter=nn.Sequential(
            nn.Linear(hidden_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.text_dec_adapter=nn.Sequential(
            nn.Linear(hidden_size, dec_dmodel),
            nn.ReLU(),
            nn.Linear(dec_dmodel, dec_dmodel)
        )
        self.encoder=Encoder(config['model'])
        self.encoder_text=Encoder(config['model'])
        self.decoder=Decoder(config['model'])
        self.mean_proj = nn.Linear(d_model, z_dim)
        self.logvar_proj = nn.Linear(d_model, z_dim)
        self.z_proj = nn.Linear(z_dim, dec_dmodel)
        self.mean_proj_t = nn.Linear(d_model, z_dim)
        self.logvar_proj_t = nn.Linear(d_model, z_dim)
        self.z_proj_t = nn.Linear(z_dim, dec_dmodel)
        self.out_proj=nn.Linear(dec_dmodel, config['model']['pose_dim'])
        self.config=config
        self.anchor_frame=None

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
    def mean_anchor_frame(self,dl_loader):
        #dl_loaderから全ての骨格データを取得し，平均的なフレームを計算する
        total_frames=0
        mean_frame=None
        for batch in dl_loader:
            padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, sequence = batch
            skeleton=padded_cod_data.to(next(self.parameters()).device)  # (batch_size, seq_len, J, C)
            B,T,J,C=skeleton.size()
            skeleton=skeleton.reshape(B,T,-1)  # (batch_size, seq_len, J*C)
            if mean_frame is None:
                mean_frame=torch.zeros(skeleton.size(2), device=skeleton.device)  # (J*C,)
            mean_frame+=skeleton.sum(dim=(0,1))  # (J*C,)
            total_frames+=B*T
        mean_frame/=total_frames
        self.anchor_frame=mean_frame  # (J*C,)
        self.anchor_frame.requires_grad=False
    def forward(self,skeleton,skeleton_length,text_inputs):
        B,T,J,C=skeleton.size()
        skeleton_mask= create_pading_mask(skeleton_length, skeleton.size(1)).bool().to(skeleton.device)
        text_mask=text_inputs['attention_mask'].to(skeleton.device) # (batch_size, seq_len_text)
        text_tokens=text_inputs['input_ids'].to(skeleton.device)  # (batch_size, seq_len_text)
        skeleton_emb=self.skeleton_embed(skeleton.reshape(B,T,-1))
        # with torch.no_grad():
            #skeleton_emb=self.skeleton_embed(skeleton,skeleton_mask,return_sequence=True)["last_hidden_state"]  # (batch_size, seq_len_skeleton, d_model)
        if len(text_tokens.shape)==1:
            text_tokens=text_tokens.unsqueeze(1)
            text_mask=text_mask.unsqueeze(1)
        text_out=self.text_embed(text_tokens,text_mask)
        text_emb=text_out["last_hidden_state"]  # (batch_size, seq_len_text, d_model)
        text_proj=text_out['pooled_features']

        #VAE(skeleton)
        src_mask=None
        text_kv = self.text_adapter(text_emb)
        text_dec_input=self.text_dec_adapter(text_emb)  # (batch_size, seq_len_text, dec_dmodel)
        # (batch_size, seq_len_text, d_model)
        #skeleton_emb=self.skeleton_adapter(skeleton_emb)  # (batch_size, seq_len_skeleton, d_model)
        encoded=self.encoder(skeleton_emb,src_mask=src_mask,q_padding_mask=skeleton_mask)  # (batch_size, max_seq_len_skeleton, d_model)
        mean=self.mean_proj(encoded)  # (batch_size, max_seq_len_skeleton, z_dim)
        logvar_s=self.logvar_proj(encoded).clamp(-30.0, 20.0)  # (batch_size, max_seq_len_skeleton, z_dim)
        std_s=torch.exp(0.5*logvar_s)
        eps_s=torch.randn_like(std_s)
        z=mean+eps_s*std_s  # (batch_size, max_seq_len_skeleton, z_dim)
        z_dec=self.z_proj(z)  # (batch_size, max_seq)
        #print("[NaN DEBUG] z_dec nan:", torch.isnan(z_dec).sum().item())
        #print("[NaN DEBUG] text_emb nan:", torch.isnan(text_emb).sum().item())
        #print("[NaN DEBUG] text_proj nan:", torch.isnan(text_proj).sum().item())
        #print("[NaN DEBUG] text_dec_input nan:", torch.isnan(text_dec_input).sum().item())
        #Decoder
        text_mask=~(text_mask.bool())  # (batch_size, max_seq_len_text) -> (batch_size, max_seq_len_text)
        #print("[NaN DEBUG] text_mask all-masked samples:", text_mask.all(dim=1).sum().item(), "/ skeleton_mask all-masked:", skeleton_mask.all(dim=1).sum().item())
        decoded=self.decoder(z_dec,text_dec_input,src_mask=src_mask,q_padding_mask=skeleton_mask,cond=text_proj,kv_padding_mask=text_mask)  # (batch_size, max_seq_len_text, dec_dmodel)
        #print("[NaN DEBUG] decoded nan:", torch.isnan(decoded).sum().item())
        s_out=self.out_proj(decoded)  # (batch_size, max_seq_len_text, pose_dim)
        #print("[NaN DEBUG] s_out nan:", torch.isnan(s_out).sum().item())
        #VAE(text)

        text_dec_input=self.text_dec_adapter(text_emb)  # (batch_size, seq_len_text, dec_dmodel)
        anchor_frame=self.skeleton_embed(self.anchor_frame.unsqueeze(0).unsqueeze(0).expand(B,1,-1)) if self.anchor_frame is not None else skeleton_emb[:,:1]
        text_enc_input=self.mhapooling(anchor_frame,text_kv,L=T,mask=text_mask)  # (batch_size, T, d_model)
        encoded_text=self.encoder_text(text_enc_input,src_mask=None,q_padding_mask=skeleton_mask)  # (batch_size, max_seq_len_text, d_model)
        mean_t=self.mean_proj_t(encoded_text)  # (batch_size, max_seq_len_text, z_dim)
        logvar_t=self.logvar_proj_t(encoded_text).clamp(-30.0, 20.0)  # (batch_size, max_seq_len_text, z_dim)
        std_t=torch.exp(0.5*logvar_t)
        eps_t=torch.randn_like(std_t)
        z_t=mean_t+eps_t*std_t  # (batch_size, max_seq_len_text, z_dim)
        z_t_dec=self.z_proj_t(z_t)  # (batch_size, max_seq_len_text, dec_dmodel)
        decoded_text=self.decoder(z_t_dec,text_dec_input,src_mask=None,q_padding_mask=skeleton_mask,cond=text_proj,kv_padding_mask=text_mask)  # (batch_size, max_seq_len_text, dec_dmodel)
        t_out=self.out_proj(decoded_text)  # (batch_size, max_seq_len_text, pose_dim)

        #--loss--#
        skeleton_mask=~(skeleton_mask.bool())  # (batch_size, max_seq_len_skeleton) -> (batch_size, max_seq_len_skeleton)
        skeleton=skeleton.reshape(B,T,-1)  # (batch_size, seq_len_skeleton, J*C)
        body_skeleton=skeleton[:,:,:8*C]
        hand_skeleton=skeleton[:,:,8*C:50*C]
        #face_skeleton=skeleton[:,:,50*C:]
        s_body=s_out[:,:,:8*C]
        s_hand=s_out[:,:,8*C:50*C]
        #s_face=s_out[:,:,50*C:]
        t_body=t_out[:,:,:8*C]
        t_hand=t_out[:,:,8*C:50*C]
        #t_face=t_out[:,:,50*C:]
        body_weights=self.config['loss_parameters']['body_weight']
        hand_weights=self.config['loss_parameters']['hand_weight']
        #face_weights=self.config['loss_parameters']['face_weight']
        recon_loss=(body_weights*F.smooth_l1_loss(s_body,body_skeleton.detach(), reduction='none').mean(dim=-1)
                    +hand_weights*F.smooth_l1_loss(s_hand,hand_skeleton.detach(), reduction='none').mean(dim=-1)
                    #+face_weights*F.smooth_l1_loss(s_face,face_skeleton.detach(), reduction='none').mean(dim=-1)
                    )  # (batch_size, max_seq_len_skeleton)
        recon_loss=(recon_loss*skeleton_mask).sum()/skeleton_mask.sum()

        recon_loss_text=(body_weights*F.smooth_l1_loss(t_body,body_skeleton.detach(), reduction='none').mean(dim=-1)
                         +hand_weights*F.smooth_l1_loss(t_hand,hand_skeleton.detach(), reduction='none').mean(dim=-1)
                         #+face_weights*F.smooth_l1_loss(t_face,face_skeleton.detach(), reduction='none').mean(dim=-1)
                         )  # (batch_size, max_seq_len_skeleton)
        recon_loss_text=(recon_loss_text*skeleton_mask).sum()/skeleton_mask.sum()
        kl_loss_s=(skeleton_mask*(-0.5*torch.mean(1+logvar_s-mean.pow(2)-logvar_s.exp(),dim=-1))).sum()/skeleton_mask.sum()
        kl_loss_t=(skeleton_mask*(-0.5*torch.mean(1+logvar_t-mean_t.pow(2)-logvar_t.exp(),dim=-1))).sum()/skeleton_mask.sum()
        #skeletonとテキストの潜在空間を近づけるためのKLダイバージェンス
        def _masked_mean(t, mask):
            sel = t.masked_select(mask)
            return sel.mean() if sel.numel() > 0 else t.new_tensor(0.0)
        kl_loss_s_aux=self.gaussian_kl(mean, logvar_s, mean_t.detach(), logvar_t.detach(), mask=skeleton_mask)+_masked_mean(F.mse_loss(mean, mean_t.detach(), reduction='none').mean(dim=-1), skeleton_mask)+_masked_mean(F.mse_loss(logvar_s, logvar_t.detach(), reduction='none').mean(dim=-1), skeleton_mask)
        kl_loss_t_aux=self.gaussian_kl(mean_t, logvar_t, mean.detach(), logvar_s.detach(), mask=skeleton_mask)+_masked_mean(F.mse_loss(mean_t, mean.detach(), reduction='none').mean(dim=-1), skeleton_mask)+_masked_mean(F.mse_loss(logvar_t, logvar_s.detach(), reduction='none').mean(dim=-1), skeleton_mask)

        loss=self.config['loss_parameters']['recon_skeleton']*recon_loss+self.config['loss_parameters']['recon_text']*recon_loss_text+self.config['loss_parameters']['kl_skeleton']*kl_loss_s+self.config['loss_parameters']['kl_text']*kl_loss_t+self.config['loss_parameters']['kl_skeleton_aux']*kl_loss_s_aux+self.config['loss_parameters']['kl_text_aux']*kl_loss_t_aux
        #loss=self.config['loss_parameters']['recon_skeleton']*recon_loss+self.config['loss_parameters']['kl_skeleton']*kl_loss_s

        return {
            "loss": loss,
            "recon_loss_skeleton": recon_loss,
            "recon_loss_text": recon_loss_text,
            "kl_loss_skeleton": kl_loss_s,
            "kl_loss_text": kl_loss_t,
            "kl_loss_skeleton_aux": kl_loss_s_aux,
            "kl_loss_text_aux": kl_loss_t_aux,
            "text_output":t_out,
        }

