import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer,AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType
from models.module.VQ_VAE_Transformer import VQVAETransformer1D,VQLossWeights
from models.module.VAE_diffusion import VAETransformerDiffusion
def create_pading_mask(target_length, max_len):
    # target_length: (batch_size,)
    batch_size = target_length.size(0)
    mask = torch.ones((batch_size, max_len), dtype=torch.float32, device=target_length.device)
    for i in range(batch_size):
        mask[i, :target_length[i]] = 0.0
    return mask  # (batch_size, max_len)

class MBartText2Pose(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert 'vqvae_config' in config, "vqvae_config is required in the config dictionary"
        vqvae_config=config['vqvae_config']
        model_name=config.get('model_name', "facebook/mbart-large-50")
        pose_dim=config.get('pose_dim', 378)
        use_peft=config.get('use_peft', True)
        body_weights=config.get('body_weights', 1.0)
        hand_weights=config.get('hand_weights', 1.0)
        face_weights=config.get('face_weights', 1.0)
        self.body_weights=body_weights
        self.hand_weights=hand_weights
        self.face_weights=face_weights
        self.recon_weights=config.get('recon_weights',1.0)
        self.ce_weights=config.get('ce_weights',1.0)
        # 1. Tokenizerとモデルのロード
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        loss_w = VQLossWeights()
        loss_w.recon_pos = vqvae_config['loss_parameters']['recon_pos_weight']
        loss_w.recon_hand = vqvae_config['loss_parameters']['recon_hand_weight']
        loss_w.recon_face = vqvae_config['loss_parameters']['recon_face_weight']
        loss_w.vq = vqvae_config['loss_parameters']['vq_weight']
        self.pose_tokenizer=VQVAETransformer1D(in_dim=vqvae_config["model"]["in_dim"],
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
        n_codes=self.pose_tokenizer.n_codes
        #new_tokens = ["[Pose_BOS]"] + [f"[POSE_{i}]" for i in range(n_codes)]
        self.pose_emb=nn.Embedding(n_codes+3, 1024)
        self.pose_lm_head=nn.Linear(1024, n_codes+3, bias=False)
        self.pose_bos_token_id=n_codes
        self.pose_eos_token_id=n_codes+1
        self.pose_pad_token_id=n_codes+2
        self.pose_tokenizer.eval()
        for p in self.model.parameters():
            p.requires_grad = False  # モデルの全パラメータを凍結

        def no_causal_mask(*args, **kwargs):
            return None  # causal maskを無効

        # 3. PEFT (LoRA) の設定
        if use_peft:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                lora_dropout=0.3,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        # ---------------------------------------------------------
        # 追加・変更部分：ポーズを入出力するための2つのLinear層
        # ---------------------------------------------------------
        d_model = self.model.config.d_model

        self.mse_loss = nn.MSELoss(reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean',label_smoothing=0.1,ignore_index=self.pose_pad_token_id)
        #embeddingおよびlm_headの重みを追加トークンの学習に限定するためのフック関数を登録


    def token_augmentation(self,tokens, threshold=0.1, n_codes=1024):
        mask = torch.rand(tokens.shape, device=tokens.device) < threshold
        random_tokens = torch.randint(0, n_codes, tokens.shape, device=tokens.device)
        tokens[mask] = random_tokens[mask]
        return tokens

    def train(self, mode=True):
        super().train(mode)
        self.pose_tokenizer.eval()  # 常にeval固定
        # pose_tokenizer内部のパラメータもrequires_grad=Falseに
        for p in self.pose_tokenizer.parameters():
            p.requires_grad = False
        return self
    @torch.no_grad()
    def pose_tokenize(self,poses,src_mask=None):
        self.pose_tokenizer.eval()  # モデルを評価モードに設定
        # poses: (Batch, Seq_Len, Pose_Dim)
        B,T,J,C=poses.shape
        poses_flat=poses.view(B,T,-1)  # (Batch, Seq_Len, Pose_Dim)
        tokens=self.pose_tokenizer.tokenize(poses_flat, src_mask=src_mask)
        return tokens

    def forward(self, input_texts, target_poses, target_length):
        device = self.model.device
        B, T, J, C = target_poses.shape

        # --- VQ tokenize ---
        raw_tokens = self.pose_tokenize(target_poses)  # (B, K_max) ただし K_max はサンプルごとに異なる可能性がある
        K_max = raw_tokens.size(1)

        #if self.training:
        #    raw_tokens = self.token_augmentation(raw_tokens, n_codes=self.pose_tokenizer.n_codes)

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
        target_emb=self.pose_emb(target_tokens) #(B, L_max, d_model)
        # --- forward ---
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            decoder_inputs_embeds=target_emb,  # 埋め込みを直接渡す
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_state = outputs.decoder_hidden_states[-1]  # (B, L_max, d_model)
        logits = self.pose_lm_head(hidden_state)  # (B, L_max, n_codes+3)
        ce_loss = self.cross_entropy_loss(
            logits[:,:-1].reshape(-1, logits.size(-1)),
            target_tokens[:,1:].reshape(-1)
        )
        #g_soft=F.gumbel_softmax(logits[:,:,self.pose_start_id:], tau=1.0, hard=True, dim=-1)#(B, L_max, n_codes+pose_start_id)
        #z_q=torch.matmul(g_soft,self.pose_tokenizer.quant.codebook.weight) #(B, L_max, code_dim)
        #pd_pose=self.pose_tokenizer.decode(z_q,T) #(B, L_max, Pose_Dim)
        #target_poses=target_poses.view(B,T,-1)
        #padding_mask=1-create_pading_mask(target_length, T)
        #recon_loss=self.mse_loss(pd_pose, target_poses).mean(dim=-1) #(B, L_max)
        #recon_loss=(recon_loss*padding_mask).sum()/padding_mask.sum() #(B,)
        loss= self.ce_weights * ce_loss


        return {
            "loss": loss,
            #"recon_loss": recon_loss.detach(),
            "ce_loss": ce_loss.detach(),
        }

    @torch.no_grad()
    def sample(self, input_texts, target_length,
               num_beams=5, do_sample=False,
               temperature=1.0, top_p=1.0, top_k=0):
        """
        テキストからポーズ系列を生成する。

        Args:
            input_texts: 入力テキストのリスト (長さ B)
            target_length: (B,) Tensor または list。各サンプルの目標フレーム長。
            num_beams, do_sample, temperature, top_p, top_k:
                HuggingFace generate() に渡す生成パラメータ。

        Returns:
            (B, T_max, Pose_Dim_flat) の Tensor。
        """
        self.model.eval()
        self.pose_tokenizer.eval()
        device = self.model.device

        # --- target_length を Tensor 化 ---
        if not torch.is_tensor(target_length):
            target_length = torch.tensor(target_length, device=device, dtype=torch.long)
        else:
            target_length = target_length.to(device).long()

        B = target_length.size(0)
        T = int(target_length.max().item())

        # --- テキストエンコード ---
        inputs = self.tokenizer(
            input_texts, return_tensors="pt",
            padding=True, truncation=True,
        ).to(device)

        # --- 生成するコード長を VQ-VAE の stride から見積もり ---
        # forward と同じく K ≈ T / stride。マージンとして +少しを取る。
        stride = getattr(self.pose_tokenizer, "stride", 4)
        K_max = (T + stride - 1) // stride  # ceil(T / stride)
        max_new_tokens = K_max + 4  # BOS + codes + EOS + 余裕

        n_codes = self.pose_tokenizer.n_codes
        pose_start =0
        pose_end = pose_start + n_codes

        # --- MBart で系列生成 ---
        # 学習時のラベルは [Pose_BOS] + codes + [EOS] なので、
        # forced_bos_token_id で最初に [Pose_BOS] を出させる。
        generated = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=2,
            forced_bos_token_id=self.pose_bos_token_id,
            eos_token_id=self.pose_eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.2,
        )  # (B, L)

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
        #codebook = self.pose_tokenizer.quant.codebook.weight  # (n_codes, code_dim)
        #z_q = codebook[codes_padded]  # (B, K_actual, code_dim)
        z_q=self.pose_tokenizer.quant.indices_to_codes(codes_padded) #(B, K_actual, code_dim)

        pd_pose = self.pose_tokenizer.decode(z_q, T)  # (B, T, Pose_Dim_flat)

        return pd_pose.view(B, T, -1)