import torch
from transformers import CLIPTokenizer, CLIPTextModel
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from models.module.VQ_VAE_Transformer import VQVAETransformer1D, VQLossWeights, VQVAETransformer1DSeparated


class UnitsDecoder(nn.Module):
    # 簡単なTransformerデコーダーの例
    def __init__(self, config):
        super().__init__()
        num_codes = {"pose": config['vqvae']['separated_vae']['n_pose_codes'],
                     "left": config['vqvae']['separated_vae']['n_hand_codes'],
                     "right": config['vqvae']['separated_vae']['n_hand_codes'],
                     "extra": config['vqvae']['separated_vae']['n_extra_codes']}
        rvq_staves = config['vqvae']['rvq_stages']
        pose_hidden_dim = config['model']['pose_hidden_dim']
        hand_hidden_dim = config['model']['hand_hidden_dim']
        extra_hidden_dim = config['model']['extra_hidden_dim']
        hidden_dim = pose_hidden_dim + hand_hidden_dim * 2 + extra_hidden_dim
        ffn_dim = config['model']['ffn_mult'] * hidden_dim
        num_layers = config['model']['num_layers']
        num_heads = config['model']['num_heads']
        dropout = config.get("dropout", 0.1)
        activation = config['model'].get("activation", "gelu")
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim,
                                                   dropout=dropout, norm_first=True, activation=activation)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.pose_fc_out = nn.Linear(hidden_dim, config['vqvae']['separated_vae']['n_pose_codes'])
        self.left_hand_fc_out = nn.Linear(hidden_dim, config['vqvae']['separated_vae']['n_hand_codes'])
        self.right_hand_fc_out = nn.Linear(hidden_dim, config['vqvae']['separated_vae']['n_hand_codes'])
        self.extra_fc_out = nn.Linear(hidden_dim, config['vqvae']['separated_vae']['n_extra_codes'])
        self.input_fc = nn.ModuleDict({
            "pose": nn.Embedding(config['vqvae']['separated_vae']['n_pose_codes'], pose_hidden_dim),
            "left": nn.Embedding(config['vqvae']['separated_vae']['n_hand_codes'], hand_hidden_dim),
            "right": nn.Embedding(config['vqvae']['separated_vae']['n_hand_codes'], hand_hidden_dim),
            "extra": nn.Embedding(config['vqvae']['separated_vae']['n_extra_codes'], extra_hidden_dim),
        })
        self.bos_token = nn.Parameter(torch.randn(1, hidden_dim))

    def sinusoidal_position_encoding(self, seq_len, dim):
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (seq_len, dim)

    def forward(self, encoded_text, pose_input, text_attn_mask, pose_attn_mask, text_lang):
        # encoded_text: (batch_size, seq_len, hidden_size)
        # text_langからの埋め込みを開始トークンとする
        batch_size, seq_len, _ = encoded_text.size()
        decoder_input = torch.cat(
            [self.input_fc["pose"](pose_input[0][:, 0]), self.input_fc["left"](pose_input[1][:, 0],),
             self.input_fc["right"](pose_input[2][:,0]), self.input_fc["extra"](pose_input[3][:, 0])],
            dim=-1)  # (batch_size, seq_len, hidden_dim)
        decoder_input = torch.cat([self.bos_token.expand(batch_size, -1, -1), decoder_input], dim=1)  # BOSトークンの追加
        pose_len = decoder_input.size(1)
        decoder_input = decoder_input + self.sinusoidal_position_encoding(pose_len, decoder_input.size(-1)).to(
            decoder_input.device).unsqueeze(0)  # 位置エンコーディングの追加
        decoder_input = decoder_input.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        encoded_text = encoded_text.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        # デコーダーのマスクを追加
        # causal_maskの追加
        causal_mask = nn.Transformer.generate_square_subsequent_mask(pose_len).to(
            decoder_input.device)  # (seq_len, seq_len)
        # pose_attn_mask=torch.cat([torch.ones((batch_size, 1), device=pose_attn_mask.device).bool(), pose_attn_mask.bool()], dim=1)  # BOSトークンのマスクを追加
        # causal_mask = torch.triu(torch.ones((pose_len, pose_len), device=decoder_input.device), diagonal=1).bool()  # (seq_len, seq_len)
        decoded_output = self.transformer_decoder(decoder_input, encoded_text, tgt_mask=causal_mask,
                                                  tgt_key_padding_mask=~pose_attn_mask.bool(),
                                                  memory_key_padding_mask=~text_attn_mask.bool())  # (seq_len, batch_size, hidden_size)
        decoded_output = decoded_output.permute(1, 0, 2)  # (batch_size, seq_len+1, hidden_size)
        pose_output = self.pose_fc_out(decoded_output)  # (batch_size, t+1, n_pose_codes)
        left_output = self.left_hand_fc_out(decoded_output)  # (batch_size, t+1, n_hand_codes)
        right_output = self.right_hand_fc_out(decoded_output)  # (batch_size, t+1, n_hand_codes)
        extra_output = self.extra_fc_out(decoded_output)  # (batch_size, t+1, n_extra_codes)
        # 停止判定のロジットも出力
        return pose_output[:, :-1], left_output[:, :-1], right_output[:, :-1], extra_output[:, :-1]

    def generate(self, encoded_text, pose_input, text_attn_mask, pose_attn_mask, max_len, text_lang, t=0):
        # 生成モードの実装（ビームサーチなども可能）
        # ここでは単純に順次生成する例を示す
        # lang_injectinがaddでなければ，開始トークンとしてtext_langの埋め込みを使用，それ以外の場合は，最初のフレームのpose_inputを使用
        # stop_logitsも出力する
        batch_size = encoded_text.size(0)
        decoder_input = self.input_fc(pose_input)[:, :t + 1]  # (batch_size, seq_len, hidden_size)
        pe = self.sinusoidal_position_encoding(int(max_len), decoder_input.size(-1)).to(decoder_input.device).unsqueeze(
            0)  # (1, max_len, hidden_size)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(int(max_len)).to(
            decoder_input.device)  # (max_len, max_len)
        decoder_input = decoder_input + pe[:, :t + 1]  # 位置エンコーディングの追加
        decoder_input = decoder_input.permute(1, 0, 2)  # (t+1, batch_size, hidden_size)
        encoded_text_perm = encoded_text.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        tgt_mask = causal_mask[:t + 1, :t + 1]  # (t+1, t+1)
        decoded_output = self.transformer_decoder(decoder_input, encoded_text_perm, tgt_mask=tgt_mask,
                                                  tgt_key_padding_mask=~pose_attn_mask.bool(),
                                                  memory_key_padding_mask=~text_attn_mask.bool())  # (t+1, batch_size, hidden_size)
        decoded_output = decoded_output.permute(1, 0, 2)  # (batch_size, t+1, hidden_size)
        pose_output = self.pose_fc_out(decoded_output)  # (batch_size, t+1, n_pose_codes)
        left_output = self.left_hand_fc_out(decoded_output)  # (batch_size, t+1, n_hand_codes)
        right_output = self.right_hand_fc_out(decoded_output)  # (batch_size, t+1, n_hand_codes)
        extra_output = self.extra_fc_out(decoded_output)  # (batch_size, t+1, n_extra_codes)

        return pose_output[:, :-1], left_output[:, :-1], right_output[:, :-1], extra_output[:, :-1]


class Text2Units(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stride=config["vqvae"]["stride"]
        if config['model']['text_encoder_name'] == "openai/clip-vit-base-patch32":
            self.text_encoder = CLIPTextModel.from_pretrained(config['model']['text_encoder_name'])
            if config['model']['text_encoder_requires_grad'] is False:
                self.text_encoder.eval()  # テキストエンコーダーを評価モードに設定
            else:
                self.text_encoder.train()  # テキストエンコーダーを訓練モードに設定
                # text_model.encoderの最後の2層のみ微調整する
                for name, param in self.text_encoder.named_parameters():
                    if 'encoder.layers.10' in name or 'encoder.layers.11' in name or 'encoder.final_layer_norm' in name:
                        param.requires_grad = True  # 最後の2層のみ微調整
                    else:
                        param.requires_grad = False  # その他の層は固定
        else:
            self.text_encoder = AutoModel.from_pretrained(config['model']['text_encoder_name'])
            if config['model']['text_encoder_requires_grad'] is False:
                self.text_encoder.eval()  # テキストエンコーダーを評価モードに設定
            else:
                self.text_encoder.train()

        for param in self.text_encoder.parameters():
            param.requires_grad = config['model']['text_encoder_requires_grad']  # テキストエンコーダーのパラメータを固定
        loss_w = VQLossWeights()
        loss_w.recon_pos = config['vqvae']['recon_pos_weight']
        loss_w.recon_dir = config['vqvae']['recon_dir_weight']
        loss_w.vq = config['vqvae']['vq_weight']
        self.units_model = VQVAETransformer1DSeparated(
            pose_d_model=config["vqvae"]['separated_vae']['pose_d_model'],
            hand_d_model=config["vqvae"]['separated_vae']['hand_d_model'],
            extra_d_model=config["vqvae"]['separated_vae']['extra_d_model'],
            n_pose_layers_enc=config["vqvae"]['separated_vae']['n_pose_layers_enc'],
            n_hand_layers_enc=config["vqvae"]['separated_vae']['n_hand_layers_enc'],
            n_extra_layers_enc=config["vqvae"]['separated_vae']['n_extra_layers_enc'],
            n_pose_layers_dec=config["vqvae"]['separated_vae']['n_pose_layers_dec'],
            n_hand_layers_dec=config["vqvae"]['separated_vae']['n_hand_layers_dec'],
            n_extra_layers_dec=config["vqvae"]['separated_vae']['n_extra_layers_dec'],
            n_pose_heads=config["vqvae"]['separated_vae']['n_pose_heads'],
            n_hand_heads=config["vqvae"]['separated_vae']['n_hand_heads'],
            n_extra_heads=config["vqvae"]['separated_vae']['n_extra_heads'],
            pose_code_dim=config["vqvae"]['separated_vae']['pose_code_dim'],
            hand_code_dim=config["vqvae"]['separated_vae']['hand_code_dim'],
            extra_code_dim=config["vqvae"]['separated_vae']['extra_code_dim'],
            n_pose_codes=config["vqvae"]['separated_vae']['n_pose_codes'],
            n_hand_codes=config["vqvae"]['separated_vae']['n_hand_codes'],
            n_extra_codes=config["vqvae"]['separated_vae']['n_extra_codes'],
            stride=config["vqvae"]["stride"],
            ff_mult=config["vqvae"]["ff_mult"],
            dropout=config["vqvae"]["dropout"],
            rvq_stages=config["vqvae"]["rvq_stages"],
            vq_beta=config["vqvae"]["vq_beta"],
            loss_w=loss_w
        )
        self.units_model.eval()
        for p in self.units_model.parameters():
            p.requires_grad = False
        self.decoder = UnitsDecoder(config)
        text_dim = self.text_encoder.config.hidden_size
        hidden_dim=config['model']['pose_hidden_dim']+config['model']['hand_hidden_dim']*2+config['model']['extra_hidden_dim']
        if config['model']["text_adapter"] == False and text_dim == hidden_dim:
            self.tp_mapper = nn.Identity()
        else:
            self.tp_mapper = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

    def create_attn_mask(self, seq_length):
        # huggingfaceのattention_maskに合わせた形状を作成(1:有効部分,0:パディング部分)
        # seq_length:(batch_size,)
        batch_size = seq_length.size(0)
        max_len = torch.max(seq_length)
        attn_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=seq_length.device)
        for i in range(batch_size):
            attn_mask[i, :seq_length[i]] = 1
        return attn_mask  # (batch_size, max_len)

    def forward(self, text_inputs, pose_input, pose_length, hand_valid_mask):
        encoded_text = self.text_encoder(**text_inputs).last_hidden_state  # (batch_size, seq_len, hidden_size)
        encoded_text = self.tp_mapper(encoded_text)  # (batch_size, seq_len, decoder_hidden_dim)
        # pose_lengthを条件として埋め込む
        # encoded_text=encoded_text+length_embed  #(batch_size, seq_len, decoder_hidden_dim)
        # Decode to pose outputs
        # デコーダーへの入力は適宜調整する必要あり
        pose_mask = self.create_attn_mask(pose_length // self.stride + 1)  # (batch_size, pose_seq_len//4+1)
        text_mask = text_inputs['attention_mask']  # (batch_size, text_seq_len)

        unit_output = self.units_model(pose_input, hand_valid_mask=hand_valid_mask,
                                       input_length=pose_length)  # (batch_size, pose_seq_len, code_dim)
        pose_unit_tokens = unit_output['pose_codes']  # (batch_size,stages, pose_seq_len)
        left_unit_tokens = unit_output['left_codes']  # (batch_size,stages, pose_seq_len)
        right_unit_tokens = unit_output['right_codes']  # (batch_size,stages, pose_seq_len)
        extra_unit_tokens = unit_output['extra_codes']  # (batch_size
        if len(pose_unit_tokens.size()) == 2:
            pose_unit_tokens = pose_unit_tokens.unsqueeze(1)  # (batch_size,1, pose_seq_len)
        if len(left_unit_tokens.size()) == 2:
            left_unit_tokens = left_unit_tokens.unsqueeze(1)  # (batch_size,1, pose_seq_len)
        if len(right_unit_tokens.size()) == 2:
            right_unit_tokens = right_unit_tokens.unsqueeze(1)  # (batch_size,1, pose_seq_len)
        if len(extra_unit_tokens.size()) == 2:
            extra_unit_tokens = extra_unit_tokens.unsqueeze(1)  # (batch_size,1, pose_seq_len)
        unit_tokens = torch.stack([pose_unit_tokens, left_unit_tokens, right_unit_tokens, extra_unit_tokens],
                                  dim=0)  # (4, batch_size, stages, pose_seq_len)
        pose_outputs, left_outputs, right_outputs, extra_outputs = self.decoder(encoded_text, unit_tokens, text_mask,
                                                                                pose_mask,
                                                                                None)  # (batch_size, pose_seq_len,code_dim)
        pose_mask = self.create_attn_mask(pose_length // self.stride)  # (batch_size, pose_seq_len//4+1)

        pose_ce_loss = F.cross_entropy(pose_outputs.reshape(-1, pose_outputs.size(-1)),
                                       pose_unit_tokens[:,0].reshape(-1), ignore_index=-100, reduction='none')#(batch_size*pose_seq_len,)
        pose_ce_loss=(pose_ce_loss.view(pose_unit_tokens.size(0), pose_unit_tokens.size(2), -1)*pose_mask.float().unsqueeze(1)).sum()/pose_mask.float().sum()#マスクされた部分を除いて平均

        left_outputs, right_outputs, extra_outputs = left_outputs.reshape(-1,
                                                                          left_outputs.size(-1)), right_outputs.reshape(
            -1, right_outputs.size(-1)), extra_outputs.reshape(-1, extra_outputs.size(-1))
        left_ce_loss = F.cross_entropy(left_outputs, left_unit_tokens[:,0].reshape(-1), ignore_index=-100, reduction='none')
        left_ce_loss=(left_ce_loss.view(left_unit_tokens.size(0), left_unit_tokens.size(2), -1)*pose_mask.float().unsqueeze(1)).sum()/pose_mask.float().sum()#マスクされた部分を除いて平均
        right_ce_loss = F.cross_entropy(right_outputs, right_unit_tokens[:,0].reshape(-1), ignore_index=-100, reduction='none')
        right_ce_loss=(right_ce_loss.view(right_unit_tokens.size(0), right_unit_tokens.size(2), -1)*pose_mask.float().unsqueeze(1)).sum()/pose_mask
        extra_ce_loss = F.cross_entropy(extra_outputs, extra_unit_tokens[:,0].reshape(-1), ignore_index=-100, reduction='none')
        extra_ce_loss=(extra_ce_loss.view(extra_unit_tokens.size(0), extra_unit_tokens.size(2), -1)*pose_mask.float().unsqueeze(1)).sum()/pose_mask.float().sum()#マスクされた部分を除いて平均
        ce_loss = pose_ce_loss + left_ce_loss + right_ce_loss + extra_ce_loss
        loss_total = ce_loss
        return {
            "pose_outputs": pose_outputs,
            "unit_tokens": unit_tokens,
            "loss_total": loss_total,
            "ce_loss": ce_loss,

        }
