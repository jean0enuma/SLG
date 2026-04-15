from transformers import AutoModel,T5Model
import torch
from torch import nn
from models.module.VQ_VAE_Transformer import VQVAETransformer1D

class VQ_T5(nn.Module):
    def __init__(self, t5_model_name, vqvae_config):
        super(VQ_T5, self).__init__()
        self.t5 = AutoModel.from_pretrained(t5_model_name)
        #encoderのみ使用するため、T5のエンコーダーのみを学習可能に，デコーダーは固定する
        for param in self.t5.parameters():
            param.requires_grad = False
        for param in self.t5.encoder.parameters():
            param.requires_grad = True
        self.t5.eval()  # T5のパラメータを固定するためにevalモードに設定
        self.vqvae = VQVAETransformer1D(**vqvae_config)
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def forward(self, x,valid_mask,input_length,text_inputs):
        with torch.no_grad():
            vq_codes=self.vqvae(x,valid_mask,input_length,no_return_loss=True)['codes']


        return vq_output
