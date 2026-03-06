import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPTextModel
import math
class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-multilingual-cased'):
        super(TextEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs  # returns the last hidden state and other outputs

class PoseDecoder(nn.Module):
    # 簡単なTransformerデコーダーの例
    def __init__(self, pose_dim, hidden_dim,ffn_dim, num_layers, num_heads,dropout=0.1,activation="gelu",num_lang=3,m_lang=False):
        super(PoseDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim, dropout=dropout,norm_first=True,activation=activation)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, pose_dim)
        self.input_fc = nn.Linear(pose_dim, hidden_dim)

        #言語情報のための埋め込み層
        #言語情報は(B)の形状
        self.m_lang=m_lang
        if m_lang:
            self.lang_embedding = nn.Embedding(num_lang, hidden_dim)  # 仮に10言語対応とする3
    def sinusoidal_position_encoding(self, seq_len, dim):
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (seq_len, dim)
    def forward(self, encoded_text, pose_input,text_attn_mask,pose_attn_mask,text_lang):
        # encoded_text: (batch_size, seq_len, hidden_size)
        #text_langからの埋め込みを開始トークンとする
        batch_size, seq_len, _ = encoded_text.size()
        pose_len=pose_input.size(1)
        decoder_input = self.input_fc(pose_input)  # (batch_size, seq_len, hidden_size)
        if self.m_lang:
            lang_embeds = self.lang_embedding(text_lang)  # (batch_size, hidden_size)
            lang_embeds = lang_embeds.unsqueeze(1)  # (batch_size, 1, hidden_size)
            decoder_input = torch.cat([lang_embeds, decoder_input[:, :-1, :]], dim=1)  # (batch_size, seq_len, hidden_size)
            pose_attn_mask=torch.cat([torch.ones((batch_size,1),dtype=pose_attn_mask.dtype,device=pose_attn_mask.device),pose_attn_mask[:,:-1]],dim=1)

        decoder_input= decoder_input + self.sinusoidal_position_encoding(pose_len, decoder_input.size(-1)).to(decoder_input.device).unsqueeze(0)  # 位置エンコーディングの追加
        decoder_input = decoder_input.permute(1, 0, 2) # (seq_len, batch_size, hidden_size)
        encoded_text = encoded_text.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        # デコーダーのマスクを追加
        # causal_maskの追加
        causal_mask=nn.Transformer.generate_square_subsequent_mask(pose_len).to(decoder_input.device)  # (seq_len, seq_len)
        #causal_mask = torch.triu(torch.ones((pose_len, pose_len), device=decoder_input.device), diagonal=1).bool()  # (seq_len, seq_len)
        decoded_output = self.transformer_decoder(decoder_input, encoded_text,tgt_mask=causal_mask,tgt_key_padding_mask=~pose_attn_mask.bool(),memory_key_padding_mask=~text_attn_mask.bool())  # (seq_len, batch_size, hidden_size)
        decoded_output = decoded_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)
        pose_outputs = self.fc_out(decoded_output)  # (batch_size, seq_len, pose_dim)
        #停止判定のロジットも出力
        return pose_outputs
    def generate(self, encoded_text, pose_input,text_attn_mask,pose_attn_mask,max_len,text_lang,t=0):
        # 生成モードの実装（ビームサーチなども可能）
        # ここでは単純に順次生成する例を示す
        # lang_injectinがaddでなければ，開始トークンとしてtext_langの埋め込みを使用，それ以外の場合は，最初のフレームのpose_inputを使用
        #stop_logitsも出力する
        batch_size = encoded_text.size(0)
        decoder_input = self.input_fc(pose_input)[:,:t+1]  # (batch_size, seq_len, hidden_size)
        pe=self.sinusoidal_position_encoding(int(max_len), decoder_input.size(-1)).to(decoder_input.device).unsqueeze(0)  # (1, max_len, hidden_size)
        causal_mask=nn.Transformer.generate_square_subsequent_mask(int(max_len)).to(decoder_input.device)  # (max_len, max_len)
        decoder_input = decoder_input + pe[:,:t+1]  # 位置エンコーディングの追加
        decoder_input = decoder_input.permute(1, 0, 2) # (t+1, batch_size, hidden_size)
        encoded_text_perm = encoded_text.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        tgt_mask=causal_mask[:t+1, :t+1]  # (t+1, t+1)
        decoded_output = self.transformer_decoder(decoder_input, encoded_text_perm,tgt_mask=tgt_mask,tgt_key_padding_mask=~pose_attn_mask.bool(),memory_key_padding_mask=~text_attn_mask.bool())  # (t+1, batch_size, hidden_size)
        decoded_output = decoded_output.permute(1, 0, 2)  # (batch_size, t+1, hidden_size)
        pose_output = self.fc_out(decoded_output)  # (batch_size, t, pose_dim)

        return pose_output




class Text2Pose(nn.Module):
    def __init__(self, config):
        super(Text2Pose, self).__init__()
        #self.config = config
        if config['text_encoder_name'] == "openai/clip-vit-base-patch32":
            self.text_encoder=CLIPTextModel.from_pretrained(config['text_encoder_name'])
        else:
            self.text_encoder = AutoModel.from_pretrained(config['text_encoder_name'])
        if config['text_encoder_requires_grad'] is False:
            self.text_encoder.eval()  # テキストエンコーダーを評価モードに設定
        for param in self.text_encoder.parameters():
            param.requires_grad = config['text_encoder_requires_grad']  # テキストエンコーダーのパラメータを固定
        self.pose_decoder = PoseDecoder(
            pose_dim=config['pose_dim'],
            hidden_dim=config['decoder_hidden_dim'] if config["text_adapter"]==True else self.text_encoder.config.hidden_size,
            ffn_dim=config['decoder_ffn_dim'],
            num_layers=config['decoder_num_layers'],
            num_heads=config['decoder_num_heads'],
            dropout=config.get('decoder_dropout', 0.1),
            num_lang=config.get('decoder_num_lang', 1)
        )
        #encoderから，decoderへの特徴量の次元変換層(一旦次元を下げ，activationを挟んでから上げる)
        text_dim=self.text_encoder.config.hidden_size
        if config["text_adapter"]==False and text_dim==config['decoder_hidden_dim']:
            self.tp_mapper=nn.Identity()
        else:
            self.tp_mapper=nn.Sequential(
                nn.LayerNorm(config['decoder_hidden_dim']),
                nn.Linear(text_dim, config['decoder_hidden_dim']),
            )
    def create_attn_mask(self,seq_length):
        #huggingfaceのattention_maskに合わせた形状を作成(1:有効部分,0:パディング部分)
        #seq_length:(batch_size,)
        batch_size = seq_length.size(0)
        max_len = torch.max(seq_length)
        attn_mask=torch.zeros((batch_size, max_len), dtype=torch.long, device=seq_length.device)
        for i in range(batch_size):
            attn_mask[i, :seq_length[i]] = 1
        return attn_mask  #(batch_size, max_len)

    def sinusoidal_embedding(self,x, dim):
        device = x.device
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device).float() / half
        )
        args = x.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return emb
    def forward(self, text_inputs,text_lang,pose_input,pose_length):
        #text_inputsはhuggingfaceのtokenizerでエンコードされた形式{input_ids:(batch_size, seq_len),atention_mask:(batch_size, seq_len)}を想定
        #text_langはテキストの言語情報など、必要に応じて使用
        #text_encoderは学習済みモデルとして想定(mbertなど)
        #pose_decoderはテキストエンコードされた特徴量からポーズ系列を生成するモデル(Transformer decoderを想定)
        # Encode the text inputs
        encoded_text = self.text_encoder(**text_inputs).last_hidden_state  # (batch_size, seq_len, hidden_size)
        encoded_text= self.tp_mapper(encoded_text)  #(batch_size, seq_len, decoder_hidden_dim)
        #pose_lengthを条件として埋め込む
        #encoded_text=encoded_text+length_embed  #(batch_size, seq_len, decoder_hidden_dim)
        # Decode to pose outputs
        #デコーダーへの入力は適宜調整する必要あり
        pose_mask=self.create_attn_mask(pose_length)  #(batch_size, pose_seq_len)
        text_mask=text_inputs['attention_mask']  #(batch_size, text_seq_len)
        pose_input=pose_input.permute(0,2,3,1)
        pose_input=pose_input.reshape(pose_input.size(0),pose_input.size(1),-1)
        pose_outputs= self.pose_decoder(encoded_text,pose_input, text_mask,pose_mask,text_lang)  # (batch_size, pose_seq_len, pose_dim)

        return {
            "predicted_poses": pose_outputs,
        }
    def encoder_text_forward(self,text_inputs,pose_length):
        encoded_text = self.text_encoder(**text_inputs).last_hidden_state  # (batch_size, seq_len, hidden_size)
        length_embed = self.sinusoidal_embedding(pose_length, encoded_text.size(-1)).unsqueeze(
            1)  # (batch_size, 1, hidden_size)
        encoded_text = torch.concat([length_embed, encoded_text], dim=1)  # (batch_size, seq_len+1, hidden_size)
        encoded_text = self.tp_mapper(encoded_text)  # (batch_size, seq_len, decoder_hidden_dim)
        text_mask=text_inputs['attention_mask']  #(batch_size, text_seq_len)
        text_mask=torch.cat([torch.ones((text_mask.size(0),1),dtype=text_mask.dtype,device=text_mask.device),text_mask],dim=1)  #(batch_size, seq_len+1)
        return encoded_text,text_mask
    @torch.no_grad()
    def generate(self,encoded_text,text_lang,pose_input,pose_length):
        # Generate pose sequences given text inputs
        encoded_text,text_mask=self.encoder_text_forward(encoded_text,pose_length)
        pose_input=pose_input.permute(0,2,3,1)
        pose_mask = self.create_attn_mask(pose_length)
        pose_input = pose_input.reshape(pose_input.size(0), pose_input.size(1), -1)
        max_len = pose_input.size(1)
        for t in range(max_len-1):
            generated_poses= self.pose_decoder(encoded_text,pose_input, text_mask,pose_mask,text_lang)  # (batch_size, pose_seq_len, pose_dim)
            pose_input[:,t+1]=generated_poses[:,t]
        return {
            "predicted_poses": pose_input,
        }
