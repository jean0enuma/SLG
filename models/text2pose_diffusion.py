import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPTextModel
from models.module.DiffusionTransformer import DiffusionTransformer
import math
# -------------------------
# Utilities: sinusoidal time embedding
# -------------------------
def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (B,) int64/float
    return: (B, dim)
    """
    half = dim // 2
    device = t.device
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device).float() / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def cosine_beta_schedule(timesteps, s=0.008, beta_end=0.02):
    """[8] で提案されたcosine schedule"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999) * beta_end
class DiffusionSchedule(nn.Module):
    def __init__(self, T: int, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.T = T
        #betas=linear_beta_schedule(T, beta_start, beta_end)  # (T,)+
        #betas=sigmoid_beta_schedule(T, beta_start, beta_end)  # (T,)
        betas = cosine_beta_schedule(T,beta_end=beta_end)  # (T,)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = nn.Parameter(betas, requires_grad=False)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.alpha_bars = nn.Parameter(alpha_bars, requires_grad=False)
    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape):
        """
        a: (T,)
        t: (B,) in [0, T-1]
        x_shape: the shape of the target tensor we want to extract to (B,...)
        return: (B,1,1,...) same number of dims as x_shape
        """
        batch_size = t.size(0)
        out = a.gather(-1, t).view(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        x0: (B,T,D)
        t: (B,) in [0, T-1]
        noise: (B,T,D)
        return xt: (B,T,D)
        """
        # gather alpha_bar_t per batch
        ab = self.alpha_bars[t].view(-1, 1, 1)  # (B,1,1)
        xt = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
        return xt

    # ---- DDIM helpers (x0-pred parameterization) ----
    @torch.no_grad()
    def ddim_timesteps(self, ddim_steps: int, device=None) -> torch.Tensor:
        """
        Returns (ddim_steps,) timesteps in descending order (T-1 ... 0) but strided.
        """
        if ddim_steps <= 0:
            raise ValueError("ddim_steps must be >= 1")
        if ddim_steps > self.T:
            # allow, but clamp to T
            ddim_steps = self.T
        # uniform stride
        c = self.T // ddim_steps
        # e.g. T=1000, steps=50 -> take 999, 979, ...
        ts = torch.arange(0, self.T, c, device=device)[:ddim_steps]
        ts = torch.flip(ts, dims=[0]).long()
        # ensure last is 0
        ts[-1] = 0
        return ts

    @torch.no_grad()
    def predict_eps_from_x0(self, xt: torch.Tensor, t: torch.Tensor, x0_pred: torch.Tensor) -> torch.Tensor:
        """
        xt: (B,L,D), t:(B,)
        x0_pred: (B,L,D)
        return eps_pred: (B,L,D)
        """
        ab_t = self.extract(self.alpha_bars, t, xt.shape)
        eps = (xt - torch.sqrt(ab_t) * x0_pred) / torch.sqrt(1.0 - ab_t).clamp_min(1e-12)
        return eps
    @torch.no_grad()
    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        """
        xt: (B,L,D), t:(B,)
        eps_pred: (B,L,D)
        return x0_pred: (B,L,D)
        """
        ab_t = self.extract(self.alpha_bars, t, xt.shape)
        x0_pred = (xt - torch.sqrt(1.0 - ab_t) * eps_pred) / torch.sqrt(ab_t).clamp_min(1e-12)
        return x0_pred
    @torch.no_grad()
    def ddpm_step(self, xt: torch.Tensor, t: torch.Tensor, x0_pred: torch.Tensor):
        """
        One DDPM update: x_t -> x_{t-1}.
        xt: (B,L,D)
        t: (B,) int64
        x0_pred: (B,L,D)
        """
        ab_t = self.extract(self.alpha_bars, t, xt.shape)  # (B,1,1)
        ab_prev = self.extract(self.alpha_bars, t - 1, xt.shape)  # (B,1,1)

        eps = self.predict_eps_from_x0(xt, t, x0_pred)

        x_prev = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1.0 - ab_prev) * eps
        return x_prev
    @torch.no_grad()
    def ddim_step(self, xt: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, x0_pred: torch.Tensor,
                  eta: float = 0.0,eps_pred=None):
        """
        One DDIM update: x_t -> x_{t_prev} (usually t_prev < t).
        xt: (B,L,D)
        t, t_prev: (B,) int64
        x0_pred: (B,L,D)
        eta: 0 => deterministic DDIM, >0 => adds stochasticity
        """
        ab_t = self.extract(self.alpha_bars, t, xt.shape)  # (B,1,1)
        ab_prev = self.extract(self.alpha_bars, t_prev, xt.shape)  # (B,1,1)

        eps = self.predict_eps_from_x0(xt, t, x0_pred) if eps_pred is None else eps_pred

        # sigma_t (DDIM)
        # sigma = eta * sqrt((1-ab_prev)/(1-ab_t)) * sqrt(1 - ab_t/ab_prev)
        one = torch.ones_like(ab_t)
        sigma = (
                eta
                * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t).clamp_min(1e-12))
                * torch.sqrt((1.0 - ab_t / ab_prev.clamp_min(1e-12)).clamp_min(0.0))
        )

        # direction term
        dir_coeff = torch.sqrt((1.0 - ab_prev - sigma ** 2).clamp_min(0.0))
        x_prev = torch.sqrt(ab_prev) * x0_pred + dir_coeff * eps

        if eta > 0.0:
            z = torch.randn_like(xt)
            x_prev = x_prev + sigma * z

        return x_prev
    def v_target(self,x0,t,noise):
        """
        For v-pred parameterization: v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x0
        """
        ab = self.extract(self.alpha_bars, t, x0.shape)
        v = torch.sqrt(ab) * noise - torch.sqrt(1.0 - ab) * x0
        return v
    def v_to_x0_eps(self,xt,t,v):
        """
        Convert v-pred parameterization to x0 and eps:
        x0 = sqrt(alpha_bar)*xt - sqrt(1-alpha_bar)*v
        eps = sqrt(alpha_bar)*v + sqrt(1-alpha_bar)*xt
        """
        ab = self.extract(self.alpha_bars, t, xt.shape)
        x0 = torch.sqrt(ab) * xt - torch.sqrt(1.0 - ab) * v
        eps = torch.sqrt(ab) * v + torch.sqrt(1.0 - ab) * xt
        return x0, eps
    def eps_to_x0(self, xt: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        """
        xt: (B,L,D), t:(B,)
        eps_pred: (B,L,D)
        return x0_pred: (B,L,D)
        """
        ab_t = self.extract(self.alpha_bars, t, xt.shape)
        x0_pred = (xt - torch.sqrt(1.0 - ab_t) * eps_pred) / torch.sqrt(ab_t).clamp_min(1e-12)
        return x0_pred
    @torch.no_grad
    def ddim_step_vprediction(self, xt: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, x0_pred: torch.Tensor, eps_pred: torch.Tensor,
                  eta: float = 0.0):
        """
                One DDIM update: x_t -> x_{t_prev} (usually t_prev < t).
                xt: (B,L,D)
                t, t_prev: (B,) int64
                x0_pred: (B,L,D)
                eta: 0 => deterministic DDIM, >0 => adds stochasticity
                """
        ab_t = self.extract(self.alpha_bars, t, xt.shape)  # (B,1,1)
        ab_prev = self.extract(self.alpha_bars, t_prev, xt.shape)  # (B,1,1)

        eps = eps_pred

        # sigma_t (DDIM)
        # sigma = eta * sqrt((1-ab_prev)/(1-ab_t)) * sqrt(1 - ab_t/ab_prev)
        one = torch.ones_like(ab_t)
        sigma = (
                eta
                * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t).clamp_min(1e-12))
                * torch.sqrt((1.0 - ab_t / ab_prev.clamp_min(1e-12)).clamp_min(0.0))
        )

        # direction term
        dir_coeff = torch.sqrt((1.0 - ab_prev - sigma ** 2).clamp_min(0.0))
        x_prev = torch.sqrt(ab_prev) * x0_pred + dir_coeff * eps

        if eta > 0.0:
            z = torch.randn_like(xt)
            x_prev = x_prev + sigma * z

        return x_prev



class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-multilingual-cased'):
        super(TextEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs  # returns the last hidden state and other outputs

class PoseDecoderDiffusion(nn.Module):
    # 簡単なTransformerデコーダーの例
    def __init__(self, pose_dim, time_dim:int,lang_dim,hidden_dim,cond_dim,ffn_dim, num_layers, num_heads,dropout=0.1,activation="gelu",num_lang=3,m_lang=False,objective="x0"):
        super().__init__()
        #decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim, dropout=dropout,norm_first=True,activation=activation)
        #decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim, dropout=dropout,norm_first=True,activation=activation)
        #self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.transformer_decoder = DiffusionTransformer(
            data_dim=pose_dim,
            hidden_dim=hidden_dim,
            depth=num_layers,
            n_heads=num_heads,
            time_embed_dim=time_dim,
            cond_in_dim=cond_dim + (lang_dim if m_lang else 0),
            dropout=dropout,
            use_cross_attn=True,
            objective=objective
        )
        self.objective=objective
        self.fc_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, pose_dim))
        self.input_fc = nn.Linear(pose_dim, hidden_dim)
        self.time_dim=time_dim
        self.m_lang=m_lang
        #言語情報のための埋め込み層
        #言語情報は(B)の形状
        if m_lang==True:
            self.lang_embedding = nn.Embedding(num_lang, lang_dim)
            self.cond_mlp=nn.Sequential(
                nn.Linear(hidden_dim+lang_dim+time_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.cond_mlp = nn.Sequential(
                nn.Linear(hidden_dim+ time_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

    def sinusoidal_position_encoding(self, seq_len, dim):
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (seq_len, dim)
    def forward(self, encoded_text, pose_input,t,text_attn_mask,pose_attn_mask,text_lang):
        # encoded_text: (batch_size, seq_len, hidden_size)
        #text_langからの埋め込みを開始トークンとする
        batch_size, seq_len, _ = encoded_text.size()
        pose_len=pose_input.size(1)
        x_key_padding_mask=~pose_attn_mask.bool()  #(batch_size, pose_seq_len)
        cond_padding_mask=~text_attn_mask.bool() #(batch_size, text_seq_len)
        pose_output=self.transformer_decoder(pose_input,t,encoded_text, x_key_padding_mask=x_key_padding_mask, cond_key_padding_mask=cond_padding_mask)  # (batch_size, seq_len, hidden_size)
        return pose_output
    @torch.no_grad()
    def ddpm_sample(
            self,
            diffusion_schedule: DiffusionSchedule,
            encoded_text: torch.Tensor,
            text_attn_mask: torch.Tensor,
            pose_attn_mask: torch.Tensor,
            text_lang: torch.Tensor,
            pose_shape: tuple,  # (B, pose_len, pose_dim)
            ddpm_steps: int = 1000,
            x_T: torch.Tensor | None = None,  # optional start noise
    ) -> torch.Tensor:
        """
        DDPM sampling for eps-pred model.
        Returns: x0 (approx) of shape (B, pose_len, pose_dim)
        """
        device = encoded_text.device
        B, pose_len, pose_dim = pose_shape

        if x_T is None:
            x = torch.randn((B, pose_len, pose_dim), device=device)
        else:
            x = x_T

        for t in reversed(range(1,diffusion_schedule.T)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            if self.objective=="eps":
                eps_pred=self.forward(encoded_text, x, t_batch, text_attn_mask, pose_attn_mask, text_lang)  # (B, pose_len, pose_dim)
                x0_pred=diffusion_schedule.eps_to_x0(x, t_batch, eps_pred) #(B, pose_len, pose_dim)
            elif self.objective=="v0":
                v_pred=self.forward(encoded_text, x, t_batch, text_attn_mask, pose_attn_mask, text_lang)  # (B, pose_len, pose_dim)
                x0_pred,eps_pred=diffusion_schedule.v_to_x0_eps(x,t_batch,v_pred)#(B, pose_len, pose_dim)
            else:
                # x0 prediction by the model
                x0_pred = self.forward(encoded_text, x, t_batch, text_attn_mask, pose_attn_mask, text_lang)  # (B, pose_len, pose_dim)

            # update x -> x_{t-1}
            #print(t)
            x = diffusion_schedule.ddpm_step(x, t_batch, x0_pred)
        return x
    # -------------------------
    # DDIM sampling (追加)
    # -------------------------
    @torch.no_grad()
    def ddim_sample(
            self,
            diffusion_schedule: DiffusionSchedule,
            encoded_text: torch.Tensor,
            text_attn_mask: torch.Tensor,
            pose_attn_mask: torch.Tensor,
            text_lang: torch.Tensor,
            pose_shape: tuple,  # (B, pose_len, pose_dim)
            ddim_steps: int = 50,
            eta: float = 0.0,
            x_T: torch.Tensor | None = None,  # optional start noise
    ) -> torch.Tensor:
        """
        DDIM sampling for x0-pred model.
        Returns: x0 (approx) of shape (B, pose_len, pose_dim)
        """
        device = encoded_text.device
        B, pose_len, pose_dim = pose_shape

        if x_T is None:
            x = torch.randn((B, pose_len, pose_dim), device=device)
        else:
            x = x_T

        # timestep schedule
        ts = diffusion_schedule.ddim_timesteps(ddim_steps, device=device)  # (S,) descending, last=0

        for i in range(len(ts)):
            t_i = ts[i]
            t_prev = ts[i + 1] if i + 1 < len(ts) else torch.tensor(0, device=device, dtype=torch.long)

            t_batch = torch.full((B,), t_i, device=device, dtype=torch.long)
            tprev_batch = torch.full((B,), int(t_prev.item()), device=device, dtype=torch.long)

            if self.objective=="eps":
                eps_pred=self.forward(encoded_text, x, t_batch, text_attn_mask, pose_attn_mask, text_lang)  # (B, pose_len, pose_dim)
                x0_pred=diffusion_schedule.eps_to_x0(x, t_batch, eps_pred) #(B, pose_len, pose_dim)
            elif self.objective=="v0":
                v_pred=self.forward(encoded_text, x, t_batch, text_attn_mask, pose_attn_mask, text_lang)  # (B, pose_len, pose_dim)
                x0_pred,eps_pred=diffusion_schedule.v_to_x0_eps(x,t_batch,v_pred)#(B, pose_len, pose_dim)
            else:
                # x0 prediction by the model
                x0_pred = self.forward(encoded_text, x, t_batch, text_attn_mask, pose_attn_mask, text_lang)  # (B, pose_len, pose_dim)

            # update x -> x_{tprev}
            x = diffusion_schedule.ddim_step(x, t_batch, tprev_batch, x0_pred, eta=eta)

        return x


class Text2PoseDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.objective=config.get("pred_type","x0")
        if config['text_encoder_name'] == "openai/clip-vit-base-patch32":
            self.text_encoder=CLIPTextModel.from_pretrained(config['text_encoder_name'])
            if config['text_encoder_requires_grad'] is False:
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
            self.text_encoder = AutoModel.from_pretrained(config['text_encoder_name'])
            if config['text_encoder_requires_grad'] is False:
                self.text_encoder.eval()  # テキストエンコーダーを評価モードに設定
            else:
                self.text_encoder.train()

        for param in self.text_encoder.parameters():
            param.requires_grad = config['text_encoder_requires_grad']  # テキストエンコーダーのパラメータを固定
        self.pose_decoder = PoseDecoderDiffusion(
            pose_dim=config['pose_dim'],
            hidden_dim=config['decoder_hidden_dim'] if config["text_adapter"]==True else self.text_encoder.config.hidden_size,
            cond_dim=self.text_encoder.config.hidden_size,
            time_dim=config['time_embedding_dim'],
            lang_dim=config.get('lang_embedding_dim', 32),
            ffn_dim=config['decoder_ffn_dim'],
            num_layers=config['decoder_num_layers'],
            num_heads=config['decoder_num_heads'],
            dropout=config.get('decoder_dropout', 0.1),
            num_lang=config.get('decoder_num_lang', 1),
            m_lang=config.get('decoder_m_lang', False),
            objective=self.objective
        )
        #self.frame_encoder=nn.Sequential(
        #    nn.Linear(1, config['decoder_hidden_dim']//2),
        #    nn.GELU(),
        #    nn.Linear(config['decoder_hidden_dim']//2, config['decoder_hidden_dim'])
        #)
        self.diffusion_schedule = DiffusionSchedule(T=config['diffusion_timesteps'], beta_start=config.get('beta_start', 1e-4), beta_end=config.get('beta_end', 2e-2))
        #encoderから，decoderへの特徴量の次元変換層(一旦次元を下げ，activationを挟んでから上げる)
        text_dim=self.text_encoder.config.hidden_size
        if config["text_adapter"]==False and text_dim==config['decoder_hidden_dim']:
            self.tp_mapper=nn.Identity()
        else:
            self.tp_mapper=nn.Sequential(
                nn.Linear(text_dim, config['decoder_hidden_dim']),
                nn.LayerNorm(config['decoder_hidden_dim']),
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
        #length_embed = self.sinusoidal_embedding(pose_length, encoded_text.size(-1)).unsqueeze(
        #    1)  # (batch_size, 1, hidden_size)
        #encoded_text = torch.concat([length_embed, encoded_text], dim=1)  # (batch_size, seq_len+1, hidden_size)
        encoded_text = self.tp_mapper(
            encoded_text)  #(batch_size, seq_len, decoder_hidden_dim)        # Decode to pose outputs
        #デコーダーへの入力は適宜調整する必要あり
        pose_mask=self.create_attn_mask(pose_length)  #(batch_size, pose_seq_len)
        text_mask=text_inputs['attention_mask']  #(batch_size, text_seq_len)
        #text_mask=torch.cat([torch.ones((text_mask.size(0),1),dtype=text_mask.dtype,device=text_mask.device),text_mask],dim=1)  #(batch_size, seq_len+1)
        pose_input=pose_input.permute(0,2,3,1)
        pose_input=pose_input.reshape(pose_input.size(0),pose_input.size(1),-1)
        #diffusionのtimestepをランダムにサンプリング
        batch_size=pose_input.size(0)
        t = torch.randint(0, self.diffusion_schedule.T, (batch_size,), device=pose_input.device)  # (batch_size,)
        #ノイズのサンプリング
        #noise = torch.randn_like(pose_input).to(pose_input.device)  # (batch_size, pose_seq_len, pose_dim)
        noise=pose_input[:,:1,:].repeat(1,pose_input.size(1),1)  # (batch_size, pose_seq_len, pose_dim)  # ノイズを入力の最初のフレームのコピーにする
        #timestepに基づいてノイズを加える
        noisy_pose_input = self.diffusion_schedule.q_sample(pose_input, t, noise)[:,1:]  # (batch_size, pose_seq_len, pose_dim)
        pose_mask=pose_mask[:,1:]  #(batch_size, pose_seq_len-1)
        # me_embedsを生成し、text_langに基づいてlang_embedsを生成し、それらをcond_mlpに入力してcond
        pose_outputs= self.pose_decoder(encoded_text,noisy_pose_input,t, text_mask,pose_mask,text_lang)  # (batch_size, pose_seq_len, pose_dim)
        if self.objective=="eps":
            eps_targets=noise  #(batch_size, pose_seq_len, pose_dim)
            x0_pred=self.diffusion_schedule.eps_to_x0(noisy_pose_input,t,pose_outputs) #(batch_size, pose_seq_len, pose_dim)
            return {
                "predicted_poses": pose_outputs,
                "eps_targets": eps_targets,
                "x0_pred": x0_pred,
                "pose_length": pose_length
            }
        elif self.objective=="v0":
            v_targets=self.diffusion_schedule.v_target(pose_input,t,noise)  #(batch_size, pose_seq_len, pose_dim)
            x0_pred,eps_pred=self.diffusion_schedule.v_to_x0_eps(noisy_pose_input,t,pose_outputs)
            return {
                "predicted_poses": pose_outputs,
                "v_targets": v_targets,
                "x0_pred": x0_pred,
                "eps_pred": eps_pred,
                "pose_length": pose_length-1
            }
        else:
            return {
                "predicted_poses": pose_outputs,
                "target_poses": pose_input[:,1:,:],
                "pose_length": pose_length-1
            }

    @torch.no_grad()
    def generate(self, text_inputs, text_lang, pose_input, pose_length, ddim_steps: int = 50, eta: float = 0.0,is_ddim=True):
        """
        DDIM sampling版（推論）。
        pose_input は shape確保のために使う（中身は使わない）。pose_length は attn mask 用。
        """
        encoded_text = self.text_encoder(**text_inputs).last_hidden_state
        encoded_text = self.tp_mapper(encoded_text)

        pose_mask = self.create_attn_mask(pose_length)
        text_mask = text_inputs['attention_mask']

        pose_input = pose_input.permute(0, 2, 3, 1)
        pose_input = pose_input.reshape(pose_input.size(0), pose_input.size(1), -1)

        B, pose_len, pose_dim = pose_input.shape
        if is_ddim:
            #batch_size = pose_input.size(0)
            #t = torch.randint(499,500, (batch_size,), device=pose_input.device)  # (batch_size,)
            #noise = torch.randn_like(pose_input).to(pose_input.device)  # (batch_size, pose_seq_len, pose_dim)
            noise = pose_input[:, :1, :].repeat(1, pose_input.size(1),
                                                1)  # (batch_size, pose_seq_len, pose_dim)  # ノイズを入力の最初のフレームのコピーにする

            #x_T=self.diffusion_schedule.q_sample(pose_input, t, noise)  # (batch_size, pose_seq_len, pose_dim)

            x0 = self.pose_decoder.ddim_sample(
                diffusion_schedule=self.diffusion_schedule,
                encoded_text=encoded_text,
                text_attn_mask=text_mask,
                pose_attn_mask=pose_mask,
                text_lang=text_lang,
                pose_shape=(B, pose_len, pose_dim),
                ddim_steps=ddim_steps,
                eta=eta,
                x_T=noise,  # or pass your own start noise
            )
        else:
            x0=self.pose_decoder.ddpm_sample( diffusion_schedule=self.diffusion_schedule,
                encoded_text=encoded_text,
                text_attn_mask=text_mask,
                pose_attn_mask=pose_mask,
                text_lang=text_lang,
                pose_shape=(B, pose_len, pose_dim),
                ddpm_steps=self.diffusion_schedule.T,
                x_T=None)

        #stop_logits = self.stop_logits(encoded_text[:, :1]).squeeze(-1).squeeze(-1)
        return {"predicted_poses": x0}