"""
Sign Language Transformer (joint CTC recognition + autoregressive translation)

skeleton pose (N, T, K*D)
      |
SpatialEmbeddings (Linear projection + positional encoding)
      |
TransformerEncoder --(CTC)--> gloss_output_layer -> gloss recognition (optional)
      |
      v encoder_output (memory)
TransformerDecoder <-- txt_embed(BOS,...)  [autoregressive, teacher forcing]
      |
   generator (vocab projection) -> spoken language text
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SpatialEmbeddings(nn.Module):
    """skeleton pose (N, T, K*D) -> (N, T, H). Linear projection + LayerNorm + PE."""

    def __init__(self, input_dim: int, embedding_dim: int, dropout: float = 0.1,
                 norm: bool = True, max_len: int = 5000):
        super().__init__()
        self.proj = nn.Linear(input_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim) if norm else nn.Identity()
        self.scale = math.sqrt(embedding_dim)
        self.pe = PositionalEncoding(embedding_dim, max_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) * self.scale
        x = self.norm(x)
        return self.pe(x)


def generate_square_subsequent_mask(sz: int, device=None) -> torch.Tensor:
    """causal mask for decoder self-attention. True = masked position."""
    return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)


class SignLanguageTransformer(nn.Module):
    def __init__(
        self,
        pose_dim: int,            # K*D (keypoints x coord dim, e.g. 133*2 or 75*3)
        gloss_vocab_size: int,    # CTC blank included (blank=0 convention)
        txt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        # --- shared encoder ---
        self.spatial_embed = SpatialEmbeddings(pose_dim, d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)
        self.gloss_output_layer = nn.Linear(d_model, gloss_vocab_size)  # CTC head

        # --- translation decoder ---
        self.txt_embed = nn.Embedding(txt_vocab_size, d_model, padding_idx=pad_idx)
        self.txt_pe = PositionalEncoding(d_model, dropout=dropout)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)
        self.generator = nn.Linear(d_model, txt_vocab_size)  # vocab projection

    def encode(self, pose: torch.Tensor, pose_pad_mask: torch.Tensor = None):
        """
        pose: (N, T, K*D)
        pose_pad_mask: (N, T) bool, True = padding position
        returns: memory (N,T,H), gloss_logits (N,T,gloss_vocab)
        """
        x = self.spatial_embed(pose)
        memory = self.encoder(x, src_key_padding_mask=pose_pad_mask)
        gloss_logits = self.gloss_output_layer(memory)
        return memory, gloss_logits

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: torch.Tensor = None,
               tgt_key_padding_mask: torch.Tensor = None,
               memory_key_padding_mask: torch.Tensor = None):
        """tgt: (N, L) token ids, teacher-forcing input (BOS, w1, ..., w_{L-1})"""
        y = self.txt_embed(tgt) * math.sqrt(self.d_model)
        y = self.txt_pe(y)
        out = self.decoder(
            y, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(out)  # (N, L, vocab)

    def forward(self, pose, tgt_in, pose_pad_mask=None, tgt_key_padding_mask=None):
        """
        pose: (N, T, K*D)
        tgt_in: (N, L) shifted-right target (BOS, w1, ..., w_{L-1})
        returns: gloss_logits (N,T,gloss_vocab) [raw, apply log_softmax for CTC],
                 txt_logits  (N,L,vocab)
        """
        memory, gloss_logits = self.encode(pose, pose_pad_mask)
        tgt_mask = generate_square_subsequent_mask(tgt_in.size(1), tgt_in.device)
        txt_logits = self.decode(
            tgt_in, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=pose_pad_mask,
        )
        return gloss_logits, txt_logits

    @torch.no_grad()
    def greedy_decode(self, pose, bos_idx, eos_idx, max_len=100, pose_pad_mask=None):
        self.eval()
        memory, gloss_logits = self.encode(pose, pose_pad_mask)
        N = pose.size(0)
        ys = torch.full((N, 1), bos_idx, dtype=torch.long, device=pose.device)
        finished = torch.zeros(N, dtype=torch.bool, device=pose.device)
        for _ in range(max_len - 1):
            tgt_mask = generate_square_subsequent_mask(ys.size(1), pose.device)
            logits = self.decode(ys, memory, tgt_mask=tgt_mask,
                                  memory_key_padding_mask=pose_pad_mask)
            next_tok = logits[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            finished |= (next_tok.squeeze(1) == eos_idx)
            if finished.all():
                break
        return ys, gloss_logits


# ------------------------------------------------------------------
# loss例 (joint CTC + CE, SLT論文と同様の重み付き和)
# ------------------------------------------------------------------
def compute_losses(gloss_logits, txt_logits, gloss_targets, gloss_target_lengths,
                    input_lengths, txt_targets, pad_idx=0, ctc_weight=1.0, ce_weight=1.0):
    """
    gloss_logits: (N,T,gloss_vocab) raw
    txt_logits:   (N,L,vocab) raw
    gloss_targets: 1D concatenated targets for CTC (sum(gloss_target_lengths),)
    input_lengths: (N,) encoder出力の有効長 (T方向)
    txt_targets:  (N,L) teacher forcing出力側 (w1,...,EOS)
    """
    log_probs = gloss_logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T,N,gloss_vocab)
    ctc_loss = nn.functional.ctc_loss(
        log_probs, gloss_targets, input_lengths, gloss_target_lengths,
        blank=0, zero_infinity=True,
    )
    ce_loss = nn.functional.cross_entropy(
        txt_logits.reshape(-1, txt_logits.size(-1)), txt_targets.reshape(-1),
        ignore_index=pad_idx,
    )
    total = ctc_weight * ctc_loss + ce_weight * ce_loss
    return total, ctc_loss, ce_loss


if __name__ == "__main__":
    # 動作確認
    N, T, K, D = 4, 120, 75, 3
    gloss_vocab, txt_vocab, L = 300, 2000, 15

    model = SignLanguageTransformer(pose_dim=K * D, gloss_vocab_size=gloss_vocab,
                                     txt_vocab_size=txt_vocab)
    pose = torch.randn(N, T, K * D)
    tgt_in = torch.randint(1, txt_vocab, (N, L))
    gloss_logits, txt_logits = model(pose, tgt_in)
    print(gloss_logits.shape, txt_logits.shape)  # (4,120,300) (4,15,2000)

    ys, _ = model.greedy_decode(pose, bos_idx=1, eos_idx=2, max_len=20)
    print(ys.shape)