from models.module.VAE_diffusion import Encoder,Decoder,create_mask,create_slide_window_mask
import torch
import torch.nn as nn
from tslearn.metrics import SoftDTWLossPyTorch
class VAETransformerDTW(nn.Module):
    def __init__(self, config):
        super().__init__()
        q_input_dim = config['encoder']['q_input_dim']
        kv_input_dim = config['encoder']['kv_input_dim']
        d_model = config['encoder']['d_model']
        z_dim = config['encoder']['z_dim']
        dec_dmodel = config['decoder']['d_model']
        self.config=config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.soft_dtw_loss = SoftDTWLossPyTorch(gamma=1.0, normalize=True)

        self.text_model_name = config.get('text_model_name', None)
        self.mean_proj = nn.Linear(d_model, z_dim)
        self.logvar_proj = nn.Linear(d_model, z_dim)
        self.z_proj = nn.Linear(z_dim, dec_dmodel)
        self.input_proj = nn.Linear(q_input_dim, d_model)
        self.output_fc = nn.Linear(dec_dmodel, kv_input_dim)

        self.body_weight = config.get('body_weight', 0.3)
        self.hand_weight = config.get('hand_weight', 0.7)
        self.anchor_frame=nn.Parameter(torch.zeros(1, 1, d_model))  # (1, 1, q_input_dim)

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

    def forward(self, pose_input, pose_length, text_inputs=None):
        B, T, J, C = pose_input.size()

        body_coord = pose_input[:, :, :8]
        hand_coord = pose_input[:, :, 8:]
        kv = pose_input.reshape(B, T, -1)
        q = self.anchor_frame.expand(B, T, -1)

        src_mask = create_slide_window_mask(
            T,
            window_size=self.config['encoder']['window_size'],
            device=q.device
        )
        src_mask=None
        q_padding_mask = create_mask(pose_length, T).bool()
        cond_pose_mask = create_mask(pose_length, pose_input.size(1)).bool()

        kv = self.input_proj(kv)
        encoded = self.encoder(q,kv, src_mask=src_mask,q_padding_mask=q_padding_mask,kv_padding_mask=cond_pose_mask)

        mean = self.mean_proj(encoded)
        logvar = self.logvar_proj(encoded).clamp(-30.0, 20.0)
        z = self.reparameterize(mean, logvar)

        diffusion_loss = None

        z_dec = self.z_proj(z)
        decoded = self.decoder(z_dec, src_mask=src_mask, padding_mask=q_padding_mask)
        output = self.output_fc(decoded).view(B, T, J, C)

        body_output = output[:, :, :8]
        hand_output = output[:, :, 8:]

        body_recon_loss = F.smooth_l1_loss(body_output, body_coord, reduction='none')
        hand_recon_loss = F.smooth_l1_loss(hand_output, hand_coord, reduction='none')

        valid_mask = (~q_padding_mask).long()
        denom = valid_mask.sum().clamp(min=1)

        body_recon_loss = (body_recon_loss.mean(dim=[2, 3]) * valid_mask).sum() / denom
        hand_recon_loss = (hand_recon_loss.mean(dim=[2, 3]) * valid_mask).sum() / denom
        recon_loss = self.body_weight * body_recon_loss + self.hand_weight * hand_recon_loss

        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        loss = self.config.get('recon_weight', 1.0) * recon_loss + self.config['kl_weight'] * kl_loss
        if diffusion_loss is not None:
            loss = loss + self.config.get('diffusion_weight', 1.0) * diffusion_loss

        out = {
            'loss_total': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'output': output,
            'z': z,
            'length_loss':torch.zeros(1, device=pose_input.device)  # ダミーの長さ損失
        }
        if diffusion_loss is not None:
            out['diffusion_loss'] = diffusion_loss

        return out