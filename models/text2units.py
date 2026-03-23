import torch
from transformers import CLIPTokenizer, CLIPTextModel
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from models.module.VQ_VAE_Transformer import VQVAETransformer1D, VQLossWeights, VQVAETransformer1DSeparated, \
    VQVAETransformer1DAggregated, VQVAETransformer1DAggregatedCategorical
from models.module.text_word_encoder import TextTransformerEncoder


class UnitsDecoder(nn.Module):
    # 簡単なTransformerデコーダーの例
    def __init__(self, config):
        super().__init__()
        num_codes = config['vqvae']['n_codes']
        #rvq_staves = config['vqvae']['rvq_stages']
        hidden_dim = config['model']['hidden_dim']
        ffn_dim = config['model']['ffn_mult'] * hidden_dim
        num_layers = config['model']['num_layers']
        num_heads = config['model']['num_heads']
        dropout = config.get("dropout", 0.1)
        activation = config['model'].get("activation", "gelu")
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim,
                                                   dropout=dropout, norm_first=True, activation=activation)
        self.embedding_weight=nn.Parameter(torch.randn(num_codes, hidden_dim))  # コードブックのサイズ+特殊トークン分の埋め込み行列
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, num_codes)  # 出力はコードブックのサイズ+特殊トークン分

    def sinusoidal_position_encoding(self, seq_len, dim):
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (seq_len, dim)

    def forward(self, encoded_text, pose_input, text_attn_mask, pose_attn_mask, pose_length):
        # encoded_text: (batch_size, seq_len, hidden_size)
        # pose_input: (batch_size, seq_len, n_codes)  # コードブックの確率分布
        # text_attn_mask: (batch_size, text_seq_len)  # 1:有効, 0:パディング
        # pose_attn_mask: (batch_size, pose_seq_len)  # 1:有効, 0:パディング
        # text_langからの埋め込みを開始トークンとする
        batch_size, seq_len, _ = encoded_text.size()
        pose_len = pose_input.size(1)
        #確率分布から埋め込みベクトルに変換
        decoder_input = pose_input @ self.embedding_weight  # (batch_size, seq_len, hidden_size)
        pose_length = pose_length + 1  # BOSトークン分だけ長くする
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
        output = self.fc_out(decoded_output)  # (batch_size, seq_len+1, n_codes)
        return output  # 最後のトークンは次のフレームの予測に使用するため、出力から除外
    @torch.no_grad()
    def beam_search(self, encoded_text, text_attn_mask, beam_width=5, max_len=100,alpha=0.7):
        """
        Beam search decoding for UnitsDecoder
        Args:
            encoded_text: (batch_size, text_len, hidden_dim)
            text_attn_mask: (batch_size, text_len)
            beam_width: ビーム幅 (k)
            max_len: 最大生成フレーム数
        """
        device = encoded_text.device
        batch_size = encoded_text.size(0)

        # 特殊トークンの定義 (既存コードに合わせる)
        BOS_TOKEN = self.n_codes  # <BOS>
        EOS_TOKEN = self.n_codes + 1  # <EOS>
        PAD_TOKEN = self.n_codes + 2  # <PAD>

        # 各バッチに対して独立にビームサーチを実行（簡易化のためバッチサイズ1を想定、またはループ処理）
        # ここでは実装の分かりやすさのため1サンプルずつの処理を記述します
        all_best_sequences = []

        for b in range(batch_size):
            # (1, text_len, hidden_dim)
            curr_encoded_text = encoded_text[b:b + 1].permute(1, 0, 2)
            curr_text_mask = ~text_attn_mask[b:b + 1].bool()

            # 初期状態: [(スコア, トークンリスト)]
            # スコアは対数尤度の和。初期は BOS トークンのみ
            beams = [(0.0, [BOS_TOKEN])]
            completed_sequences = []

            for _ in range(max_len):
                new_beams = []
                for score, seq in beams:
                    if seq[-1] == EOS_TOKEN:
                        completed_sequences.append((score, seq))
                        continue

                    # 現在の系列をデコーダーに入力
                    tgt_input = torch.tensor([seq], device=device).transpose(0, 1)  # (seq_len, 1)
                    tgt_emb = self.input_fc(tgt_input)
                    tgt_emb = tgt_emb + self.sinusoidal_position_encoding(len(seq), tgt_emb.size(-1)).to(
                        device).unsqueeze(1)

                    causal_mask = nn.Transformer.generate_square_subsequent_mask(len(seq)).to(device)

                    # デコード実行
                    decoded = self.transformer_decoder(
                        tgt_emb, curr_encoded_text,
                        tgt_mask=causal_mask,
                        memory_key_padding_mask=curr_text_mask
                    )

                    # 最後の位置のログプロパビリティを取得
                    logits = self.fc_out(decoded[-1:])  # (1, 1, n_codes+3)
                    log_probs = F.log_softmax(logits, dim=-1).squeeze()  # (n_codes+3)

                    # 上位 k 個の候補を取得
                    topk_probs, topk_ids = torch.topk(log_probs, beam_width)

                    for i in range(beam_width):
                        new_beams.append((score + topk_probs[i].item(), seq + [topk_ids[i].item()]))

                # --- 長さペナルティを適用してソート ---
                def get_norm_score(s, sequence):
                    L = len(sequence)
                    lp = ((5 + L) ** alpha) / ((5 + 1) ** alpha)
                    return s / lp
                # 正規化後のスコアで上位 k 個を選択
                beams = sorted(new_beams, key=lambda x: get_norm_score(x[0], x[1]), reverse=True)[:beam_width]

                # すべてのビームが終了トークンに達したら早期終了
                if all(s[-1] == EOS_TOKEN for _, s in beams):
                    break

            completed_sequences.extend(beams)
            # 最もスコアの高い系列を選択（長さペナルティを考慮する場合はここでスコア/len(seq)とする）
            best_seq = max(completed_sequences, key=lambda x: x[0])[1]
            all_best_sequences.append(best_seq)

        return all_best_sequences


class Text2Units(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stride = config["vqvae"]["stride"]
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
            text_dim=self.text_encoder.config.hidden_size
        elif config['model']['text_encoder_name'] == "word_embedding":
            self.text_encoder =TextTransformerEncoder(vocab_size=config['model']['vocab_size'], d_model=config['model']['hidden_dim'],
                                        nhead=config['model']['num_heads'], num_layers=2,
                                        dim_feedforward=config['model']['ffn_mult'] * config['model']['hidden_dim'],
                                        padding_token_id=config['model']['pad_token_id'])

            text_dim=config['model']['hidden_dim']
        else:
            self.text_encoder = AutoModel.from_pretrained(config['model']['text_encoder_name'])
            text_dim=self.text_encoder.config.hidden_size
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
        self.units_model = VQVAETransformer1DAggregatedCategorical(
            n_codes=config['vqvae']['n_codes'],
            code_dim=config['vqvae']['code_dim'],
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
            tau=config["vqvae"]["tau"],
            loss_w=loss_w
        )
        self.n_codes = config['vqvae']['n_codes']

        self.units_model.eval()
        for p in self.units_model.parameters():
            p.requires_grad = False
        self.decoder = UnitsDecoder(config)
        hidden_dim = config['model']['hidden_dim']
        if config['model']["text_adapter"] == False and text_dim == hidden_dim:
            self.tp_mapper = nn.Identity()
        else:
            encoder = nn.TransformerEncoderLayer(d_model=text_dim, nhead=config['model']['num_heads'],
                                                 dim_feedforward=config['model']['ffn_mult'] * hidden_dim,
                                                 dropout=config['model'].get("dropout", 0.1),
                                                 activation=config['model'].get("activation", "gelu"), norm_first=True)
            self.tp_mapper = nn.Sequential(
                nn.TransformerEncoder(encoder, num_layers=2),
                nn.Linear(text_dim, hidden_dim)
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
        B,T,J,C=pose_input.shape
        encoded_text = self.text_encoder(**text_inputs).last_hidden_state  # (batch_size, seq_len, hidden_size)
        encoded_text = self.tp_mapper(encoded_text)  # (batch_size, seq_len, decoder_hidden_dim)

        text_mask = text_inputs['attention_mask']  # (batch_size, text_seq_len)

        unit_output = self.units_model(pose_input, hand_valid_mask=hand_valid_mask,
                                       input_length=pose_length)  # (batch_size, pose_seq_len, code_dim)
        pose_mask = self.create_attn_mask(pose_length // self.stride)  # BOSトークンを追加したため、マスクも1フレーム分長くする

        unit_tokens = F.softmax(unit_output['z_e'],dim=-1)#(Batch_size, pose_seq_len, n_codes)
        n_codes=unit_tokens.size(-1)
        pose_length = pose_length // self.stride
        output = self.decoder(encoded_text, unit_tokens, text_mask, pose_mask,
                              pose_length)  # (batch_size, pose_seq_len,n_codes)
        #クロスエントロピーの計算
        ce_loss = F.cross_entropy(output[:,1:].reshape(-1, n_codes), unit_tokens[:,:-1].reshape(-1,n_codes).detach(),reduction="none").reshape(B,T-1)#(B,T-1)
        pose_mask=self.create_attn_mask(pose_length-1)  # (B,T-1)
        ce_loss=(ce_loss*pose_mask).sum()/pose_mask.sum()

        # accuracyの計算
        acc = torch.zeros(1, device=output.device)
        return {
            "loss_total": ce_loss,
            "ce_loss": ce_loss,
            "acc": acc,
            "unit_tokens": unit_tokens,
        }

    @torch.no_grad()
    def generate(self, text_inputs, beam_width=5, max_len=100, alpha=0.7):
        """
        テキスト入力からトークン列を生成する
        Args:
            text_inputs: tokenizerの出力（input_ids, attention_maskなど）
            beam_width: ビーム幅
            max_len: 最大生成長
            alpha: 長さペナルティの係数
        Returns:
            list[list[int]]: 生成されたトークン列（バッチごと）
        """
        self.eval()

        # 1. テキストをエンコード
        # CLIPなどのモデルはdict入力を受け取る想定
        encoded_text = self.text_encoder(**text_inputs).last_hidden_state
        encoded_text = self.tp_mapper(encoded_text)

        text_mask = text_inputs['attention_mask']

        # 2. デコーダーのビームサーチを実行
        # 生成されたトークン列のリストを返す
        generated_tokens = self.decoder.beam_search(
            encoded_text,
            text_mask,
            beam_width=beam_width,
            max_len=max_len,
            alpha=alpha
        )

        return generated_tokens
class Text2UnitsTimeEmbedding(Text2Units):
    def __init__(self, config):
        super().__init__(config)
        if config['model']['text_encoder_name'] == "openai/clip-vit-base-patch32":
            self.text_encoder = CLIPTextModel.from_pretrained(config['model']['text_encoder_name'])
            if config['model']['text_encoder_requires_grad'] is False:
                self.text_encoder.eval()  # テキストエンコーダーを評価モードに設定
                for p in self.text_encoder.parameters():
                    p.requires_grad = False  # テキストエンコーダーのパラメータを固定
            text_dim = self.text_encoder.config.hidden_size
        elif config['model']['text_encoder_name'] == "word_embedding":
            self.text_encoder = TextTransformerEncoder(vocab_size=config['model']['vocab_size'],
                                                       d_model=config['model']['hidden_dim'],
                                                       nhead=config['model']['num_heads'], num_layers=2,
                                                       dim_feedforward=config['model']['ffn_mult'] * config['model'][
                                                           'hidden_dim'],
                                                       padding_token_id=config['model']['pad_token_id'])

            text_dim = config['model']['hidden_dim']
        else:
            self.text_encoder = AutoModel.from_pretrained(config['model']['text_encoder_name'])
            text_dim = self.text_encoder.config.hidden_size
            for p in self.text_encoder.parameters():
                p.requires_grad = False  # テキストエンコーダーのパラメータを固定
        if config['model']['text_encoder_requires_grad'] is False:
            self.text_encoder.eval()  # テキストエンコーダーを評価モードに設定
        else:
            self.text_encoder.train()
        sub_encoder=nn.TransformerEncoderLayer(d_model=text_dim, nhead=config['model']['num_heads'],
                                                 dim_feedforward=config['model']['ffn_mult'] * config['model']['hidden_dim'],
                                                 dropout=config['model'].get("dropout", 0.1),
                                                 activation=config['model'].get("activation", "gelu"), norm_first=True)
        self.tp_mapper=nn.TransformerEncoder(sub_encoder, num_layers=2)
        sub_decoder=nn.TransformerDecoderLayer(d_model=text_dim, nhead=config['model']['num_heads'],
                                                 dim_feedforward=config['model']['ffn_mult'] * config['model']['hidden_dim'],
                                                 dropout=config['model'].get("dropout", 0.1),
                                                 activation=config['model'].get("activation", "gelu"), norm_first=True)

        self.decoder=nn.TransformerDecoder(sub_decoder, num_layers=6)
        self.fc=nn.Linear(text_dim, config['vqvae']['n_codes'])
        self.predict_time_fc=nn.Linear(text_dim, 1)
        # pose_idxからpose_dimを計算
        pose_dim = -2
        for s in self.units_model.pose_idx:
            pose_dim += (s.stop - s.start) * s.step if s.step is not None else (s.stop - s.start)
        pose_dim *= 3
        hand_dim = 0
        for s in self.units_model.dir_l_idx:
            hand_dim += (s.stop - s.start) * s.step if s.step is not None else (s.stop - s.start)
        for s in self.units_model.dir_r_idx:
            hand_dim += (s.stop - s.start) * s.step if s.step is not None else (s.stop - s.start)
        hand_dim *= 3
        all_dim=pose_dim+hand_dim
        self.pose_query_fc=nn.Linear(all_dim, text_dim)
    def forward(self, text_inputs, pose_input, pose_length, hand_valid_mask):
        encoded_text = self.text_encoder(**text_inputs).last_hidden_state  # (batch_size, seq_len, hidden_size)
        B, T, J, C = pose_input.shape
        seq_len=torch.zeros(B, device=pose_input.device, dtype=torch.long)
        encoded_text = self.text_encoder(**text_inputs).last_hidden_state  # (batch_size, seq_len, hidden_size)

        text_mask = text_inputs['attention_mask']  # (batch_size, text_seq_len)
        for i in range(B):
            seq_len[i]=text_mask[i].sum()


        unit_output = self.units_model(pose_input, hand_valid_mask=hand_valid_mask,
                                       input_length=pose_length)  # (batch_size, pose_seq_len, code_dim)

        pose_mask = self.create_attn_mask(pose_length // self.stride)  # BOSトークンを追加したため、マスクも1フレーム分長くする(batch_size, pose_seq_len)

        unit_probs = F.softmax(-unit_output['dist'], dim=-1)  # (Batch_size, pose_seq_len, n_codes)
        n_codes = unit_probs.size(-1)
        pose_length = pose_length // self.stride
        query_input=pose_input.reshape(B,T,-1)[:,:1,:]#(B,1,J*C)
        query=self.pose_query_fc(query_input).expand(-1, T//self.stride, -1)  # (B, pose_seq_len, hidden_dim)

        text_output=self.tp_mapper(encoded_text.permute(1,0,2), src_key_padding_mask=~text_mask.bool()).permute(1,0,2)  # (B, text_seq_len+ hidden_dim)
        pd_time=self.predict_time_fc(text_output.mean(1))  # (B,1)
        tg_ratio=pose_length/seq_len
        time_loss=F.mse_loss(pd_time.squeeze(-1), tg_ratio)
        decoded_output=self.decoder(query.permute(1,0,2), text_output.permute(1,0,2), tgt_key_padding_mask=~pose_mask.bool(), memory_key_padding_mask=~text_mask.bool()).permute(1,0,2)  # (B, pose_seq_len, hidden_dim)
        decoded_output=self.fc(decoded_output)  # (B, pose_seq_len, n_codes)
        ce_loss = F.cross_entropy(decoded_output.reshape(B*T,-1), unit_probs.reshape(B*T,-1).detach(), reduction="none").reshape(B,T)#( B, pose_seq_len)
        ce_loss = (ce_loss * pose_mask).sum() / pose_mask.sum()
        acc = torch.zeros(1, device=decoded_output.device)
        loss_total = ce_loss + time_loss
        return {
            "loss_total": loss_total,
            "time_loss": time_loss,
            "ce_loss": ce_loss,
            "acc": acc,
            "unit_tokens": unit_probs,
        }
