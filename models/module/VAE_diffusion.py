import math
import torch
from torch import nn
import torch.nn.functional as F
from models.module.UnetDiffusionDenoiser import UNetDiffusionDenoiser
from models.module.TransformerUnetDiffusionDenoiser import TransformerUNetDenoiser
from models.module.TransformerStride import *

# 例:
from models.module.CLIP_Skeleton import SkeletonTextCLIP
from models.module.DiffusionTransformer import AdaLN
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


class DenoiserBlock(nn.Module):
    """
    self-attn -> text cross-attn -> FFN
    """
    def __init__(self, d_model,cond_dim, nhead, dropout=0.1,adaln_apply=(True, True, False)):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_text = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        #self.norm1 = AdaLN(d_model,cond_dim,scale_shft_gate=adaln_apply)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        #self.norm3 = AdaLN(d_model,cond_dim,scale_shft_gate=adaln_apply)
        self.norm3=nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        text,
        time_embed,
        x_padding_mask=None,
        text_padding_mask=None,
        src_mask=None,
    ):
        res = x
        gate=None
        x=self.norm1.forward(x) if isinstance(self.norm1, AdaLN) else self.norm1(x)
        x = self.dropout(
            self.self_attn(
                x,x,x,
                attn_mask=src_mask,
                key_padding_mask=x_padding_mask,
                need_weights=False
            )[0]
        )
        x=res+gate*x if gate is not None else x+res
        res=x
        x=self.norm2(x)
        x = self.dropout(
            self.cross_text(
                x, text, text,
                key_padding_mask=text_padding_mask,
                need_weights=False
            )[0]
        )
        x=x+res
        res=x
        x,gate=self.norm3.forward(x) if isinstance(self.norm3, AdaLN) else (self.norm3(x), None)
        x = self.dropout(self.ffn(x))
        x=res+gate*x if gate is not None else x+res
        return x


class DiffusionDenoiser(nn.Module):
    def __init__(
        self,
        latent_dim,
        model_dim,
        time_dim,
        nhead,
        num_layers,
        text_cond_dim,
        dropout=0.1,
    ):
        super().__init__()
        self.model_dim = model_dim

        self.in_proj = nn.Linear(latent_dim, model_dim)
        self.text_proj = nn.Linear(text_cond_dim, model_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, text_cond_dim),
        )

        self.blocks = nn.ModuleList([
            DenoiserBlock(model_dim,text_cond_dim, nhead, dropout=dropout,adaln_apply=(False,False,False))
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, latent_dim)

    def position_embedding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(
        self,
        x_t,                      # (B,T,z_dim)
        t,                        # (B,)
        text_tokens,              # (B,L,H)
        x_padding_mask=None,      # (B,T) True=pad
        text_padding_mask=None,   # (B,L) True=pad
        src_mask=None,            # (T,T) True=masked
    ):
        x = self.in_proj(x_t)
        x = x + self.position_embedding(x.size(1), x.size(2), x.device)
        time_embed=self.time_mlp(t)
        text = self.text_proj(text_tokens+time_embed.unsqueeze(1))
        #x = x + self.time_mlp(t).unsqueeze(1)


        for block in self.blocks:
            x = block(
                x=x,
                text=text,
                time_embed=time_embed,
                x_padding_mask=x_padding_mask,
                text_padding_mask=text_padding_mask,
                src_mask=src_mask,
            )

        x = self.final_norm(x)
        return self.out_proj(x)


class DiffusionModel(nn.Module):
    """
    text 条件のみの latent diffusion + CFG
    学習時:
      - text系列の一部を学習可能 null token に置換
    推論時:
      - text条件あり / null条件 の2本でCFG
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        diff_cfg = config.get("diffusion_config", {})
        self.is_text_cond=diff_cfg.get("is_text_cond", True)

        self.z_dim = config["encoder"]["z_dim"]
        self.model_dim = diff_cfg.get("model_dim", 256)
        self.nhead = diff_cfg.get("nhead", 8)
        self.num_layers = diff_cfg.get("num_layers", 4)
        self.dropout = diff_cfg.get("dropout", 0.1)
        self.time_dim= diff_cfg.get("time_dim", 128)
        self.pred_type=diff_cfg.get("pred_type","eps") # "eps" or "v"

        self.num_train_steps = diff_cfg.get("num_train_steps", 1000)
        self.sampling_steps = diff_cfg.get("sampling_steps", 50)
        self.ddim_eta = diff_cfg.get("ddim_eta", 0.0)

        # 学習時の partial null 置換率
        self.text_drop_prob = diff_cfg.get("text_drop_prob", 0.1)

        # 推論時の CFG scale
        self.cfg_scale_text = diff_cfg.get("cfg_scale_text", 3.0)

        # CLIP-style model
        clip_cfg = config["clip_text"]
        self.clip_model = SkeletonTextCLIP(config=clip_cfg)

        if diff_cfg.get("freeze_text_encoder", True):
            for p in self.clip_model.text_encoder.parameters():
                p.requires_grad = False

        text_cond_dim = self.clip_model.text_encoder.bert.config.hidden_size if self.is_text_cond else self.clip_model.skeleton_encoder.d_model

        # 学習可能 null token
        self.null_text = nn.Parameter(torch.randn(1, 1, text_cond_dim) * 0.02)

        self.denoiser = TransformerUNetDenoiser(
            latent_dim=self.z_dim,
            model_dim=self.model_dim,
            time_dim=self.time_dim,
            nhead=self.nhead,
            #num_layers=self.num_layers,
            text_cond_dim=text_cond_dim,
            dropout=self.dropout,
            num_levels=diff_cfg.get("num_levels", 3),
            depth_per_level=diff_cfg.get("depth_per_level", 2),
            use_text_at_levels=diff_cfg.get("use_text_at_levels", [True, True, True]),
        )

        betas = self.cosine_beta_schedule(self.num_train_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-5, 0.999).float()

    def _extract(self, a, t, x_shape):
        B = t.shape[0]
        out = a.gather(0, t)
        return out.view(B, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_ac * x_start + sqrt_om * noise

    def predict_x0_from_eps(self, x_t, t, eps):
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_om * eps) / (sqrt_ac + 1e-8)
    def compute_v_target(self, x0, t, eps):
        """v-predictionのターゲット: v = sqrt(ᾱ)*ε - sqrt(1-ᾱ)*x0"""
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * eps - sqrt_om * x0

    def predict_x0_from_v(self, x_t, t, v):
        """v-predictionからx0を復元: x0 = sqrt(ᾱ)*x_t - sqrt(1-ᾱ)*v"""
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_ac * x_t - sqrt_om * v

    def predict_eps_from_v(self, x_t, t, v):
        """v-predictionからεを復元: ε = sqrt(ᾱ)*v + sqrt(1-ᾱ)*x_t"""
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_ac * v + sqrt_om * x_t
    def encode_text_condition(self, text_inputs):
        tx_out = self.clip_model.text_encoder(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            token_type_ids=text_inputs.get("token_type_ids", None),
        )
        text_tokens = tx_out["token_features"]                         # (B,L,H)
        text_padding_mask = (text_inputs["attention_mask"] == 0)      # True=pad
        return text_tokens, text_padding_mask
    def encode_pose_condition(self,pose_inputs,pose_padding_mask):
        pose_out=self.clip_model.skeleton_encoder(
            skeleton=pose_inputs,
            skeleton_mask=pose_padding_mask,
            return_sequence=True,
        )
        pose_tokens=pose_out['sequence_features']
        pose_padding_mask=pose_padding_mask
        return pose_tokens,pose_padding_mask

    def get_null_text_tokens(self, batch_size, seq_len, device):
        tokens = self.null_text.expand(batch_size, seq_len, -1).to(device)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        return tokens, padding_mask

    def random_replace_text_with_partial_null(
        self,
        text_tokens,
        text_padding_mask,
    ):
        """
        学習時:
        text token の一部だけ null token に置換
        """
        if not self.training:
            return text_tokens, text_padding_mask

        B, Lt, _ = text_tokens.shape
        device = text_tokens.device

        null_text_tokens, _ = self.get_null_text_tokens(B, Lt, device)

        text_valid = ~text_padding_mask
        text_drop_mask = (torch.rand(B, Lt, device=device) < self.text_drop_prob) & text_valid

        # valid token が全部落ちるのを防ぐ
        for b in range(B):
            valid_idx = torch.where(text_valid[b])[0]
            if len(valid_idx) > 0 and text_drop_mask[b, valid_idx].all():
                keep_idx = valid_idx[torch.randint(len(valid_idx), (1,), device=device)]
                text_drop_mask[b, keep_idx] = False

        text_tokens = torch.where(
            text_drop_mask.unsqueeze(-1),
            null_text_tokens,
            text_tokens
        )

        return text_tokens, text_padding_mask

    def denoise_once(
        self,
        x_t,
        t,
        text_tokens,
        text_padding_mask,
        x_padding_mask=None,
        src_mask=None,
    ):
        return self.denoiser(
            x_t=x_t,
            t=t,
            text_tokens=text_tokens,
            x_padding_mask=x_padding_mask,
            text_padding_mask=text_padding_mask,
            src_mask=src_mask,
        )

    def forward(
        self,
        z0,
        text_inputs,
        pose_inputs,
        pose_padding_mask,
        padding_mask=None,
        src_mask=None,
    ):
        """
        z0: (B,T,z_dim)
        #TODO: text_tokensとtext_padding_maskはposeを使う可能性があるので命名を変えたほうがいいかも
        """
        B = z0.size(0)
        device = z0.device

        t = torch.randint(0, self.num_train_steps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(z0)
        x_t = self.q_sample(z0, t, noise=noise)
        if self.is_text_cond is False:
            text_tokens, text_padding_mask = self.encode_pose_condition(pose_inputs, pose_padding_mask)
        else:
            if text_inputs is None:
                # 条件なし学習: 学習可能 null token のみを条件として使う
                text_tokens, text_padding_mask = self.get_null_text_tokens(
                    batch_size=B,
                    seq_len=self.config["diffusion_config"].get("null_text_seq_len", 64),
                    device=device
                )
            else:
                text_tokens, text_padding_mask = self.encode_text_condition(text_inputs)
                text_tokens, text_padding_mask = self.random_replace_text_with_partial_null(
                    text_tokens, text_padding_mask
                )

        pred = self.denoise_once(
            x_t=x_t,
            t=t,
            text_tokens=text_tokens,
            text_padding_mask=text_padding_mask,
            x_padding_mask=padding_mask,
            src_mask=src_mask,
        )
        if self.pred_type == "v":
            target = self.compute_v_target(z0, t, noise)
        else:  # "eps"
            target = noise
        return {
            "pred": pred,
            "target": target,
            "t": t,
            "x_t": x_t,
        }


    def predict_eps_cfg(
        self,
        x_t,
        t,
        text_inputs,
        x_padding_mask=None,
        src_mask=None,
        cfg_scale_text=None,
    ):
        """
        推論時CFG:
          eps_cfg = eps_uncond + s * (eps_text - eps_uncond)
        """
        if cfg_scale_text is None:
            cfg_scale_text = self.cfg_scale_text

        B = x_t.size(0)
        device = x_t.device

        text_tokens, text_padding_mask = self.encode_text_condition(text_inputs)
        null_text_tokens, null_text_padding = self.get_null_text_tokens(B, text_tokens.size(1), device)

        # unconditional
        eps_uncond = self.denoise_once(
            x_t=x_t,
            t=t,
            text_tokens=null_text_tokens,
            text_padding_mask=null_text_padding,
            x_padding_mask=x_padding_mask,
            src_mask=src_mask,
        )

        # text conditional
        eps_text = self.denoise_once(
            x_t=x_t,
            t=t,
            text_tokens=text_tokens,
            text_padding_mask=text_padding_mask,
            x_padding_mask=x_padding_mask,
            src_mask=src_mask,
        )

        eps_cfg = eps_uncond + cfg_scale_text * (eps_text - eps_uncond)
        return eps_cfg

    def _prepare_condition(self, batch_size, device, text_inputs=None, cfg_scale_text=None):
        """
        サンプリング用の条件トークンを準備する。
        戻り値: (text_tokens, text_padding_mask, use_cfg, cfg_scale)
        """
        if cfg_scale_text is None:
            cfg_scale_text = self.cfg_scale_text

        null_seq_len = self.config["diffusion_config"].get("null_text_seq_len", 32)
        null_tokens, null_mask = self.get_null_text_tokens(batch_size, null_seq_len, device)

        if self.is_text_cond and text_inputs is not None:
            text_tokens, text_mask = self.encode_text_condition(text_inputs)
            use_cfg = cfg_scale_text > 1.0
            return text_tokens, text_mask, null_tokens, null_mask, use_cfg, cfg_scale_text
        else:
            # unconditional: null tokensのみ使用
            return null_tokens, null_mask, null_tokens, null_mask, False, 1.0

    def _predict_and_get_x0_eps(self, x_t, t, text_tokens, text_mask, null_tokens, null_mask,
                                use_cfg, cfg_scale, padding_mask, src_mask):
        """
        denoiserの出力からx0とepsを返す。
        pred_type="v" → vからx0/εを変換
        pred_type="eps" → εからx0を変換
        CFGも考慮。
        戻り値: (x0_pred, eps_pred)
        """
        pred_uncond = self.denoise_once(
            x_t=x_t, t=t,
            text_tokens=null_tokens, text_padding_mask=null_mask,
            x_padding_mask=padding_mask, src_mask=src_mask,
        )
        if use_cfg:
            pred_cond = self.denoise_once(
                x_t=x_t, t=t,
                text_tokens=text_tokens, text_padding_mask=text_mask,
                x_padding_mask=padding_mask, src_mask=src_mask,
            )
            pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        else:
            pred = pred_uncond

        if self.pred_type == "v":
            x0_pred = self.predict_x0_from_v(x_t, t, pred)
            eps_pred = self.predict_eps_from_v(x_t, t, pred)
        else:  # "eps"
            x0_pred = self.predict_x0_from_eps(x_t, t, pred)
            eps_pred = pred

        return x0_pred, eps_pred

    @torch.no_grad()
    def ddim_sample(
        self,
        seq_len,
        device,
        batch_size=1,
        text_inputs=None,
        padding_mask=None,
        src_mask=None,
        num_steps=None,
        cfg_scale_text=None,
    ):
        """
        DDIMサンプリング。
        - text_inputs=None かつ is_text_cond=False → unconditional
        - text_inputs あり かつ is_text_cond=True  → CFG付きtext条件生成
        """
        if num_steps is None:
            num_steps = self.sampling_steps

        # 条件トークンをループ前に一度だけエンコード
        text_tokens, text_mask, null_tokens, null_mask, use_cfg, cfg_scale = \
            self._prepare_condition(batch_size, device, text_inputs, cfg_scale_text)

        x = torch.randn(batch_size, seq_len, self.z_dim, device=device)

        # T-1 → 0 の等間隔なタイムステップ列
        timesteps = torch.linspace(self.num_train_steps - 1, 0, num_steps, device=device).long()

        for i, t_now in enumerate(timesteps):
            t = torch.full((batch_size,), t_now, device=device, dtype=torch.long)

            x0_pred, eps = self._predict_and_get_x0_eps(
                x, t, text_tokens, text_mask, null_tokens, null_mask,
                use_cfg, cfg_scale, padding_mask, src_mask,
            )
            alpha_bar = self._extract(self.alphas_cumprod, t, x.shape)
            x0_pred = x0_pred.clamp(-5.0, 5.0)

            # 最終ステップはx0をそのまま返す
            if i == len(timesteps) - 1:
                x = x0_pred
                break

            t_prev = torch.full((batch_size,), timesteps[i + 1], device=device, dtype=torch.long)
            alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev, x.shape)

            # DDIM更新式 (Song et al. 2020, Eq.12)
            # σ = η * sqrt((1-ᾱ_prev)/(1-ᾱ)) * sqrt(1 - ᾱ/ᾱ_prev)
            sigma = self.ddim_eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
            ).clamp(min=0.0)

            dir_xt = torch.sqrt((1 - alpha_bar_prev - sigma ** 2).clamp(min=0.0)) * eps
            noise = torch.randn_like(x) if self.ddim_eta > 0.0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

        return x

    @torch.no_grad()
    def ddpm_sample(
        self,
        seq_len,
        device,
        batch_size=1,
        text_inputs=None,
        padding_mask=None,
        src_mask=None,
        cfg_scale_text=None,
    ):
        """
        DDPMサンプリング（全T=1000ステップ）。
        - text_inputs=None かつ is_text_cond=False → unconditional
        - text_inputs あり かつ is_text_cond=True  → CFG付きtext条件生成
        """
        # 条件トークンをループ前に一度だけエンコード
        text_tokens, text_mask, null_tokens, null_mask, use_cfg, cfg_scale = \
            self._prepare_condition(batch_size, device, text_inputs, cfg_scale_text)

        x = torch.randn(batch_size, seq_len, self.z_dim, device=device)

        for step in reversed(range(self.num_train_steps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)

            _, eps = self._predict_and_get_x0_eps(
                x, t, text_tokens, text_mask, null_tokens, null_mask,
                use_cfg, cfg_scale, padding_mask, src_mask,
            )

            beta_t = self._extract(self.betas, t, x.shape)
            sqrt_one_minus_ac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t, x.shape)

            # DDPM平均 (Ho et al. 2020, Eq.11)
            model_mean = sqrt_recip_alpha * (x - beta_t / (sqrt_one_minus_ac + 1e-8) * eps)

            if step > 0:
                posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
                x = model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)
            else:
                x = model_mean

        return x
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
        q=q+self.position_embedding(q.size(1), q.size(2), q.device).to(q.dtype)
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
class VAETransformerDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        q_input_dim = config['encoder']['q_input_dim']
        kv_input_dim = config['encoder']['kv_input_dim']
        d_model = config['encoder']['d_model']
        z_dim = config['encoder']['z_dim']
        dec_dmodel = config['decoder']['d_model']
        num_queries = config['encoder'].get('num_queries', 64)
        self.config=config

        self.encoder = Encoder(config)
        self.encoder_perceiver=PoseEncoderStride(d_model)
        self.decoder_perceiver=PoseDecoderStride(dec_dmodel)
        self.decoder = Decoder(config)

        self.mean_proj = nn.Linear(d_model, z_dim)

        self.logvar_proj = nn.Linear(d_model, z_dim)
        self.z_proj = nn.Linear(z_dim, dec_dmodel)
        self.input_proj = nn.Linear(q_input_dim, d_model)
        self.output_fc = nn.Linear(dec_dmodel, kv_input_dim)

        self.body_weight = config.get('body_weight', 0.3)
        self.hand_weight = config.get('hand_weight', 0.7)
        self.face_weight = config.get('face_weight', 0.5)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    def requres_vae_grad(self):
        for param in self.parameters():
            param.requires_grad = True
        if self.is_diffusion:
            for param in self.diffusion.parameters():
                param.requires_grad = False
    def requires_diffusion_grad(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.diffusion.parameters():
            param.requires_grad = True
        for param in self.diffusion.clip_model.text_encoder.embed_model.parameters():
            param.requires_grad = False

    def match_length(self,x, target_len):
        # 1D補間のために (B, D, T) の形状で入力
        # mode='linear' で滑らかに補間
        x_resized = F.interpolate(x, size=target_len, mode='linear', align_corners=True)
        return x_resized
    def encode(self, pose_input, pose_length):
        B, T, J, C = pose_input.size()
        pose_input = pose_input.reshape(B, T, -1)
        src_mask=None
        q_padding_mask = create_mask(pose_length, T).bool()
        pose_input= self.input_proj(pose_input)
        encoded = self.encoder(pose_input,pose_input, src_mask=src_mask,q_padding_mask=q_padding_mask,kv_padding_mask=q_padding_mask)
        encoded=self.encoder_perceiver(encoded)
        mean = self.mean_proj(encoded)
        logvar = self.logvar_proj(encoded).clamp(-30.0, 20.0)
        z = self.reparameterize(mean, logvar)
        return z
    def forward(self, pose_input, pose_length, text_inputs=None):
        B, T, J, C = pose_input.size()

        body_coord = pose_input[:, :, :8]
        hand_coord = pose_input[:, :, 8:50]
        face_coord=pose_input[:, :, 50:]
        pose_input = pose_input.reshape(B, T, -1)
        src_mask=None
        q_padding_mask = create_mask(pose_length, T).bool()

        pose_input= self.input_proj(pose_input)
        encoded = self.encoder(pose_input,pose_input, src_mask=src_mask,q_padding_mask=q_padding_mask,kv_padding_mask=q_padding_mask)
        encoded=self.encoder_perceiver(encoded)
        mean = self.mean_proj(encoded)
        logvar = self.logvar_proj(encoded).clamp(-30.0, 20.0)
        z = self.reparameterize(mean, logvar)

        z_dec = self.z_proj(z)
        z_dec=self.decoder_perceiver(z_dec)
        T_conv=z_dec.size(1)
        z_dec=z_dec.view(B, T_conv, -1).transpose(1, 2)  # (B, D, T)
        z_dec=self.match_length(z_dec, T).transpose(1, 2)
        decoded = self.decoder(z_dec, src_mask=src_mask, padding_mask=q_padding_mask)
        #フレーム数を合わせる
        decoded=F.interpolate(decoded.transpose(1, 2), size=T, mode='linear', align_corners=True).transpose(1, 2)
        output = self.output_fc(decoded).reshape(B, T, J, C)


        body_output = output[:, :, :8]
        hand_output = output[:, :, 8:50]
        face_output=output[:, :, 50:]

        body_recon_loss = F.smooth_l1_loss(body_output, body_coord, reduction='none')
        hand_recon_loss = F.smooth_l1_loss(hand_output, hand_coord, reduction='none')
        face_recon_loss=F.smooth_l1_loss(face_output, face_coord, reduction='none')

        valid_mask = (~q_padding_mask).long()
        denom = valid_mask.sum().clamp(min=1)

        body_recon_loss = (body_recon_loss.mean(dim=[2, 3]) * valid_mask).sum() / denom
        hand_recon_loss = (hand_recon_loss.mean(dim=[2, 3]) * valid_mask).sum() / denom
        face_recon_loss = (face_recon_loss.mean(dim=[2, 3]) * valid_mask).sum() / denom
        recon_loss = self.body_weight * body_recon_loss + self.hand_weight * hand_recon_loss+self.face_weight*face_recon_loss

        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        loss = self.config.get('recon_weight', 1.0) * recon_loss + self.config['kl_weight'] * kl_loss
        out = {
            'loss_total': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'output': output,
            'z': z,
            'length_loss':torch.zeros(1, device=pose_input.device)  # ダミーの長さ損失
        }

        return out

    @torch.no_grad()
    def sample(self, pose_length, text_inputs, cond_pose, cond_pose_length, use_ddim=True, num_steps=None):
        if not self.is_diffusion:
            raise ValueError("Diffusion is disabled in config.")

        B = pose_length.size(0)
        T = int(pose_length.max().item())
        device = pose_length.device

        src_mask = create_slide_window_mask(
            T,
            window_size=self.config['encoder']['window_size'],
            device=device
        )
        src_mask=None
        q_padding_mask = create_mask(pose_length, T).bool()
        cond_pose_mask = create_mask(cond_pose_length, cond_pose.size(1)).bool()

        if use_ddim:
            z = self.diffusion.ddim_sample(
                text_inputs=cond_pose,
                num_steps=num_steps,
                seq_len=T,
                device=device,
                padding_mask=q_padding_mask,
                src_mask=src_mask
            )
        else:
            z = self.diffusion.ddpm_sample(
                text_inputs=text_inputs,
                seq_len=T,
                device=device,
                padding_mask=q_padding_mask,
                src_mask=src_mask
            )

        z_dec = self.z_proj(z)
        decoded = self.decoder(z_dec, src_mask=src_mask, padding_mask=q_padding_mask)
        output = self.output_fc(decoded).view(B, T,-1)
        return output

