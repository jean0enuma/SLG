import torch
import torch.nn as nn
from transformers import AutoModel,AutoTokenizer
from models.module.VQ_VAE_Transformer import VQVAETransformer1D,VQLossWeights,create_mask
from torch.nn import functional as F
class FrozenEncoderDecoder(nn.Module):
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        vqvae_config=config['vqvae_config']
        encoder_name = config.get('model_name', "google-bert/bert-base-multilingual-uncased")
        d_model= config.get('d_model', 512)
        nhead= config.get('nhead', 8)
        dim_feedforward= config.get('dim_feedforward', 2048)
        dropout= config.get('dropout', 0.1)
        num_decoder_layers= config.get('num_decoder_layers', 6)
        max_len= config.get('max_len', 512)
        loss_w = VQLossWeights()
        loss_w.recon_pos = vqvae_config['loss_parameters']['recon_pos_weight']
        loss_w.recon_hand = vqvae_config['loss_parameters']['recon_hand_weight']
        loss_w.recon_face = vqvae_config['loss_parameters']['recon_face_weight']
        loss_w.vq = vqvae_config['loss_parameters']['vq_weight']
        self.pose_tokenizer = VQVAETransformer1D(in_dim=vqvae_config["model"]["in_dim"],
                                                 d_model=vqvae_config["model"]["hidden_dim"],
                                                 n_heads=vqvae_config["model"]["n_heads"],
                                                 code_dim=vqvae_config["model"]["code_dim"],
                                                 n_codes=vqvae_config["model"]["n_codes"],
                                                 stride=vqvae_config["model"]["stride"],
                                                 n_layers_enc=vqvae_config["model"]["n_layers_enc"],
                                                 n_layers_dec=vqvae_config["model"]["n_layers_dec"],
                                                 ff_mult=vqvae_config["model"]["ff_mult"],
                                                 dropout=vqvae_config["model"]["dropout"],
                                                 rvq_stages=vqvae_config["model"]["rvq_stages"],
                                                 vq_beta=vqvae_config["model"]["vq_beta"],
                                                 levels=vqvae_config["model"]["levels"],
                                                 loss_w=loss_w)
        # --- Encoder: HuggingFace 学習済みモデル(凍結) ---
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()  # Dropout/LayerNorm の挙動を固定
        self.pose_pad_token_id = self.pose_tokenizer.n_codes  # VQコードの次がPAD
        self.pose_bos_token_id = self.pose_tokenizer.n_codes + 1  # VQコードの次の次がBOS
        self.pose_eos_token_id = self.pose_tokenizer.n_codes + 2  # VQコードの次の次の次が

        enc_dim = self.encoder.config.hidden_size
        # Encoder と Decoder の次元が違う場合に揃える
        self.enc_proj = nn.Linear(enc_dim, d_model) if enc_dim != d_model else nn.Identity()

        # --- Decoder: スクラッチ ---
        decoder_vocab_size=self.pose_tokenizer.n_codes
        self.tok_emb = nn.Embedding(decoder_vocab_size + 3, d_model)  # VQコード + BOS/EOS/PAD
        self.pos_emb = nn.Embedding(max_len, d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,  # Pre-LN の方が安定
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_decoder_layers)
        self.lm_head = nn.Linear(d_model, decoder_vocab_size + 3,bias=False)  # 出力も VQコード + BOS/EOS/PAD
    @torch.no_grad()
    def pose_tokenize(self,poses,src_mask=None):
        self.pose_tokenizer.eval()  # モデルを評価モードに設定
        # poses: (Batch, Seq_Len, Pose_Dim)
        B,T,J,C=poses.shape
        poses_flat=poses.view(B,T,-1)  # (Batch, Seq_Len, Pose_Dim)
        tokens=self.pose_tokenizer.tokenize(poses_flat, src_mask=src_mask)
        return tokens
    def train(self, mode: bool = True):
        # train() 呼び出し時に Encoder が train モードに戻らないようにする
        super().train(mode)
        self.encoder.eval()
        return self

    def encode(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.enc_proj(out.last_hidden_state)
    def token_augmentation(self,tokens, threshold=0.1, n_codes=1024):
        mask = torch.rand(tokens.shape, device=tokens.device) < threshold
        random_tokens = torch.randint(0, n_codes, tokens.shape, device=tokens.device)
        tokens[mask] = random_tokens[mask]
        return tokens

    def forward(self,input_texts, target_poses, target_length):
        device = target_poses.device
        B, T, J, C = target_poses.shape
        # --- VQ tokenize ---
        raw_tokens = self.pose_tokenize(target_poses)   # (B, K_max)
        K_max = raw_tokens.size(1)
        if self.training:
            raw_tokens = self.token_augmentation(raw_tokens, n_codes=self.pose_tokenizer.n_codes)
        # --- 各サンプルの有効コード長を pose時間スケールから比例変換 ---
        # 実際のVQVAE出力サイズ K_max を信頼することで、内部stride/padding詳細を意識しない
        ratio = K_max / T  # 例: T=200, K_max=50 → ratio=0.25
        valid_K = torch.ceil(target_length.float() * ratio).long().clamp(min=1, max=K_max)  # (B,)

        # --- ラベル組み立て: [BOS] + valid_codes + [EOS] + [-100...] ---
        L_max = K_max + 2  # BOS + max_codes + EOS
        target_tokens = torch.full((B, L_max), self.pose_pad_token_id,  # = -100
                                   dtype=torch.long, device=device)
        for b in range(B):
            L = valid_K[b].item()
            target_tokens[b, 0] = self.pose_bos_token_id
            target_tokens[b, 1:1 + L] = raw_tokens[b, :L]
            target_tokens[b, 1 + L] = self.pose_eos_token_id
            # 1+L+1 以降は -100 のまま

        # --- text encoding ---
        inputs = self.tokenizer(input_texts, return_tensors="pt",
                                padding=True, truncation=True).to(device)
        enc_input_ids = inputs.input_ids
        enc_attention_mask = inputs.attention_mask
        memory = self.encode(enc_input_ids, enc_attention_mask)

        T = target_tokens.size(1)
        pos = torch.arange(T, device=target_poses.device).unsqueeze(0)
        tgt = self.tok_emb(target_tokens) + self.pos_emb(pos)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)
        memory_kpm = (enc_attention_mask == 0)
        tgt_kpm=create_mask(valid_K+2, T)  # valid_K+2 までが有効トークン

        h = self.decoder(
            tgt=tgt, memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=memory_kpm,
        )
        logits=self.lm_head(h)
        loss=F.cross_entropy(logits[..., :-1, :].reshape(-1, logits.size(-1)),
                            target_tokens[..., 1:].reshape(-1),
                            ignore_index=self.pose_pad_token_id,label_smoothing=0.1)
        return {
            "output": logits,
            "loss": loss,
        }

    @torch.no_grad()
    def sample(
            self, input_texts, target_length,
            num_beams=5, do_sample=False,
            temperature=1.0, top_p=1.0, top_k=0,
            length_penalty: float = 1.0,
            repetition_penalty: float = 1.2,
    ):
        """
        Args:
            input_texts: str または List[str]。生の入力テキスト。
            target_length: 生成する最大トークン数。
            num_beams: 1 なら greedy / sample、>1 なら beam search。
            do_sample: True で確率的サンプリング(num_beams=1 のときのみ有効)。
            temperature, top_p, top_k: サンプリング時のフィルタリング設定。
        Returns:
            List[str]: 生成テキスト(バッチ分)。
        """
        self.eval()
        self.pose_tokenizer.eval()
        device = next(self.parameters()).device

        # --- target_length を Tensor 化 ---
        if not torch.is_tensor(target_length):
            target_length = torch.tensor(target_length, device=device, dtype=torch.long)
        else:
            target_length = target_length.to(device).long()

        B = target_length.size(0)
        T = int(target_length.max().item())

        # --- 1. テキストをトークン化 ---
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        enc = self.tokenizer(
            input_texts, return_tensors="pt", padding=True, truncation=True,
        ).to(device)
        enc_input_ids = enc["input_ids"]
        enc_attention_mask = enc["attention_mask"]
        # --- 生成するコード長を VQ-VAE の stride から見積もり ---
        # forward と同じく K ≈ T / stride。マージンとして +少しを取る。
        stride = getattr(self.pose_tokenizer, "stride", 4)
        K_max = (T + stride - 1) // stride  # ceil(T / stride)
        max_new_tokens = K_max + 4  # BOS + codes + EOS + 余裕

        n_codes = self.pose_tokenizer.n_codes
        pose_start =0
        pose_end = pose_start + n_codes
        # --- 2. Encoder は1回だけ ---
        memory = self.encode(enc_input_ids, enc_attention_mask)
        memory_kpm = (enc_attention_mask == 0)

        bos = self.pose_bos_token_id
        eos = self.pose_eos_token_id
        pad = self.pose_pad_token_id

        # --- 3. 戦略の振り分け ---
        if num_beams > 1:
            generated = self._beam_search(
                memory, memory_kpm,
                bos, eos, pad,
                max_new_tokens=target_length,
                beam_size=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
            )
        else:
            mode = "sample" if do_sample else "greedy"
            generated = self._sample_or_greedy(
                memory, memory_kpm,
                bos, eos, pad,
                max_new_tokens=target_length,
                mode=mode,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        # --- 生成系列から [POSE_i] のみを取り出してコードIDへ変換 ---
        codes_list = []
        for b in range(B):
            seq = generated[b]
            mask = (seq >= pose_start) & (seq < pose_end)
            codes = (seq[mask] - pose_start).long()  # [0, n_codes) のインデックス

            # 何も出なかった場合のフォールバック
            if codes.numel() == 0:
                codes = torch.zeros(1, dtype=torch.long, device=device)

            # サンプルごとの上限長で切り詰め
            T_b = int(target_length[b].item())
            K_b_max = max(1, (T_b + stride - 1) // stride)
            if codes.numel() > K_b_max:
                codes = codes[:K_b_max]

            codes_list.append(codes)

        # --- バッチ内で長さを揃えてパディング ---
        K_actual = max(c.numel() for c in codes_list)
        codes_padded = torch.zeros((B, K_actual), dtype=torch.long, device=device)
        for b, c in enumerate(codes_list):
            codes_padded[b, :c.numel()] = c

        # --- コードブック参照 → デコード ---
        # codebook = self.pose_tokenizer.quant.codebook.weight  # (n_codes, code_dim)
        # z_q = codebook[codes_padded]  # (B, K_actual, code_dim)
        z_q = self.pose_tokenizer.quant.indices_to_codes(codes_padded)  # (B, K_actual, code_dim)

        pd_pose = self.pose_tokenizer.decode(z_q, T)  # (B, T, Pose_Dim_flat)

        return pd_pose.view(B, T, -1)


    def _decode_step(self, memory, memory_kpm, dec_input_ids):
        """Decoder を1ステップ走らせて最終位置のロジットを返す。"""
        T = dec_input_ids.size(1)
        pos = torch.arange(T, device=dec_input_ids.device).unsqueeze(0)
        tgt = self.tok_emb(dec_input_ids) + self.pos_emb(pos)

        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)
        h = self.decoder(
            tgt=tgt, memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_kpm,
        )
        return self.lm_head(h[:, -1, :])  # (B, V)

    def _apply_repetition_penalty(self, logits, generated, penalty):
        if penalty == 1.0:
            return logits
        for b in range(generated.size(0)):
            for tok in set(generated[b].tolist()):
                if logits[b, tok] > 0:
                    logits[b, tok] /= penalty
                else:
                    logits[b, tok] *= penalty
        return logits

    def _top_k_top_p_filter(self, logits, top_k=0, top_p=1.0, filter_value=-float("inf")):
        if top_k > 0:
            kth = torch.topk(logits, top_k)[0][..., -1, None]
            logits = torch.where(logits < kth, torch.full_like(logits, filter_value), logits)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            idx_remove = remove.scatter(1, sorted_idx, remove)
            logits = logits.masked_fill(idx_remove, filter_value)
        return logits

    def _sample_or_greedy(
            self, memory, memory_kpm,
            bos, eos, pad, max_new_tokens, mode,
            temperature, top_k, top_p, repetition_penalty,
    ):
        B = memory.size(0)
        device = memory.device
        ys = torch.full((B, 1), bos, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self._decode_step(memory, memory_kpm, ys)  # (B, V)
            logits = self._apply_repetition_penalty(logits, ys, repetition_penalty)

            if mode == "greedy":
                next_tok = logits.argmax(dim=-1)
            else:  # sample
                logits = logits / max(temperature, 1e-8)
                logits = self._top_k_top_p_filter(logits, top_k, top_p)
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # 既に終了したシーケンスは pad を伸ばす
            next_tok = torch.where(finished, torch.full_like(next_tok, pad), next_tok)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)

            finished = finished | (next_tok == eos)
            if finished.all():
                break

        return ys

    def _beam_search(
            self, memory, memory_kpm,
            bos, eos, pad, max_new_tokens, beam_size, length_penalty, repetition_penalty,
    ):
        B = memory.size(0)
        device = memory.device
        V = self.lm_head.out_features

        # ビーム分だけ memory を複製: (B, ...) -> (B*beam, ...)
        def expand(x, dim=0):
            return x.repeat_interleave(beam_size, dim=dim)

        memory_b = expand(memory)
        memory_kpm_b = expand(memory_kpm)

        ys = torch.full((B * beam_size, 1), bos, dtype=torch.long, device=device)
        # 各バッチで最初の1ビームだけ生かす(同一ビーム重複展開を防ぐ)
        beam_scores = torch.full((B, beam_size), -1e9, device=device)
        beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.view(-1)  # (B*beam,)
        finished = torch.zeros(B * beam_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self._decode_step(memory_b, memory_kpm_b, ys)  # (B*beam, V)
            logits = self._apply_repetition_penalty(logits, ys, repetition_penalty)
            logprobs = F.log_softmax(logits, dim=-1)

            # 終了済みビームは EOS を確率1で維持
            logprobs[finished] = -1e9
            logprobs[finished, eos] = 0.0

            # 累積スコア
            next_scores = beam_scores.unsqueeze(-1) + logprobs  # (B*beam, V)
            next_scores = next_scores.view(B, beam_size * V)

            topk_scores, topk_idx = next_scores.topk(beam_size, dim=-1)  # (B, beam)
            beam_idx = topk_idx // V  # どのビームから
            token_idx = topk_idx % V  # どのトークン

            # 行インデックスをグローバルに直す
            flat_beam = (torch.arange(B, device=device).unsqueeze(-1) * beam_size + beam_idx).view(-1)
            ys = torch.cat([ys[flat_beam], token_idx.view(-1, 1)], dim=1)
            beam_scores = topk_scores.view(-1)
            finished = finished[flat_beam] | (token_idx.view(-1) == eos)

            if finished.all():
                break

        # 長さペナルティをかけて各バッチからベストを選ぶ
        lengths = (ys != pad).sum(dim=1).clamp(min=1).float()
        final_scores = beam_scores / (lengths ** length_penalty)
        final_scores = final_scores.view(B, beam_size)
        best = final_scores.argmax(dim=-1)  # (B,)
        best_global = torch.arange(B, device=device) * beam_size + best
        return ys[best_global]
