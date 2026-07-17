import torch
from torch import nn
from torch.nn import functional as F

class EvaluationModel(nn.Module):
    def __init__(self, config):
        super(EvaluationModel, self).__init__()
        encoder_layer=nn.TransformerEncoderLayer(d_model=config['d_model'], nhead=config['nhead'])
        pose_dim=config['pose_dim']
        num_classes=config['num_classes']
        self.proj=nn.Linear(pose_dim, config['d_model'])
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        self.logits=nn.Linear(config['d_model'], num_classes)
        self.ce_loss=nn.CrossEntropyLoss()

    def position_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(1)  # (seq_len, 1, d_model)
    def forward(self, skeleton,input_length,targets):
        # skeleton: (batch_size, seq_len, pose_dim)
        # input_length: (batch_size,)
        # targets: (batch_size,)
        B,T,J,C=skeleton.shape
        skeleton=skeleton.reshape(B,T,-1)  # (batch_size, seq_len, pose_dim)
        skeleton = skeleton.permute(1, 0, 2)  # (seq_len, batch_size, pose_dim)
        mask = self._generate_mask(input_length, T).to(skeleton.device)  # (batch_size, seq_len)
        encoded=self.proj(skeleton)  # (seq_len, batch_size, d_model)
        encoded=encoded + self.position_encoding(T, encoded.size(2)).to(skeleton.device)  # (seq_len, batch_size, d_model)
        encoded = self.encoder(encoded, src_key_padding_mask=mask)  # (seq_len, batch_size, d_model)
        encoded = encoded.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        #mean pool(maskを考慮)
        encoded = encoded.masked_fill(mask.unsqueeze(2), 0)  # (batch_size, seq_len, d_model)
        encoded = encoded.sum(dim=1) / (~mask).sum(dim=1, keepdim=True)  # (batch_size, d_model)
        logits = self.logits(encoded)
        loss=self.ce_loss(logits, targets)
        return {
            'logits': logits,
            'loss': loss
        }
    def _generate_mask(self, input_length, max_len):
        batch_size = input_length.size(0)
        mask = torch.arange(max_len).expand(batch_size, max_len).to(input_length.device) >= input_length.unsqueeze(1)
        return mask
