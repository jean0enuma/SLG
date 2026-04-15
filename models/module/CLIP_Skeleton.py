import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertTokenizer,CLIPTextModel
def create_mask(target_length, max_len):
    # target_length: (batch_size,)
    batch_size = target_length.size(0)
    mask = torch.ones((batch_size, max_len), dtype=torch.float32, device=target_length.device)
    for i in range(batch_size):
        mask[i, :target_length[i]] = 0.0
    return mask  # (batch_size, max_len)

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    x: (B, T, D)
    """
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        T = x.size(1)
        return x + self.pe[:, :T]


class SkeletonTransformerEncoder(nn.Module):
    """
    Skeleton encoder using TransformerEncoder.

    Input:
        skeleton: (B, T, J, C)
            B: batch size
            T: number of frames
            J: number of joints
            C: coordinate dimension (e.g. 2 or 3)

        skeleton_mask: (B, T)
            True for padding frame, False for valid frame
    """
    def __init__(
        self,
        num_joints: int,
        joint_dim: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        proj_dim: int = 256,
        max_len: int = 1000,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.input_dim = num_joints * joint_dim
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (B, T, D)
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.proj = nn.Linear(d_model, proj_dim)

    def masked_mean_pool(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: (B, T, D)
        mask: (B, T), True=padding, False=valid
        """
        if mask is None:
            return x.mean(dim=1)

        valid = (~mask).float().unsqueeze(-1)  # (B, T, 1)
        denom = valid.sum(dim=1).clamp_min(1.0)  # (B, 1)
        pooled = (x * valid).sum(dim=1) / denom
        return pooled

    def forward(
        self,
        skeleton: torch.Tensor,
        skeleton_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        skeleton: (B, T, J, C)
        skeleton_mask: (B, T), True=padding
        """
        B, T, J, C = skeleton.shape
        assert J == self.num_joints, f"Expected num_joints={self.num_joints}, got {J}"
        assert C == self.joint_dim, f"Expected joint_dim={self.joint_dim}, got {C}"

        x = skeleton.reshape(B, T, J * C)      # (B, T, J*C)
        x = self.input_proj(x)                 # (B, T, D)
        x = self.pos_enc(x)                    # (B, T, D)

        x = self.encoder(x, src_key_padding_mask=skeleton_mask)  # (B, T, D)
        x=self.proj(x)                                  # (B, T, proj_dim)
        pooled = self.masked_mean_pool(x, skeleton_mask)         # (B, D)
        emb = F.normalize(pooled, dim=-1)

        out = {
            "last_hidden_state": x,
            "pooled_features": pooled,
            "embeddings": emb,
        }
        if not return_sequence:
            out.pop("last_hidden_state")
        return out


class TextEncoder(nn.Module):
    """
    Text encoder using Hugging Face BERT.
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        proj_dim: int = 256,
        dropout: float = 0.1,
        pooling: str = "cls",  # "cls" or "mean"
        num_layers=2,
        nhead=8,
        ffn_mult=4,
    ):
        super().__init__()
        self.model_name=model_name
        if model_name=="openai/clip-vit-base-patch32":
            self.embed_model=CLIPTextModel.from_pretrained(model_name)
        else:
            self.embed_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.embed_model.config.hidden_size
        self.pooling = pooling
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, proj_dim),
        )

    def masked_mean_pool(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        attention_mask: (B, L), 1=valid, 0=padding
        """
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (x * mask).sum(dim=1) / denom
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.embed_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            #return_dict=True,
        )
        hidden_feature = outputs.last_hidden_state  # (B, L, D)
        hidden_feature=self.proj(hidden_feature)  # (B, L, proj_dim)
        if self.pooling == "cls":
            if self.model_name=="openai/clip-vit-base-patch32":
                pooled = hidden_feature[:, -1]  # [B, D], CLIP uses the first token as pooled output
            else:
                pooled = hidden_feature[:, 0]  # [CLS]
        elif self.pooling == "mean":
            pooled = self.masked_mean_pool(hidden_feature, attention_mask)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        emb = pooled
        emb = F.normalize(emb, dim=-1)

        return {
            "last_hidden_state": hidden_feature,
            "pooled_features": pooled,
            "embeddings": emb,
        }


class SkeletonTextCLIP(nn.Module):
    """
    CLIP-style contrastive model between skeleton and text.
    """
    def __init__(
        self,config,
    ):
        super().__init__()
        num_joints=config["num_joints"]
        joint_dim=config["joint_dim"]
        skeleton_d_model=config["skeleton_d_model"]
        skeleton_nhead=config["skeleton_nhead"]
        skeleton_num_layers=config["skeleton_num_layers"]
        skeleton_ff_dim=skeleton_d_model*config["skeleton_ffn_mult"]
        text_encoer_name=config["text_encoder_name"]
        proj_dim=config["proj_dim"]
        dropout=config.get("dropout", 0.1)
        text_pooling=config.get("text_pooling", "cls")
        init_temperature=config.get("init_temperature", 0.07)
        text_num_layers=config.get("text_num_layers", 2)
        text_nhead=config.get("text_nhead", 8)
        text_ffn_mult=config.get("text_ffn_mult", 4)
        self.skeleton_encoder = SkeletonTransformerEncoder(
            num_joints=num_joints,
            joint_dim=joint_dim,
            d_model=skeleton_d_model,
            nhead=skeleton_nhead,
            num_layers=skeleton_num_layers,
            dim_feedforward=skeleton_ff_dim,
            dropout=dropout,
            proj_dim=proj_dim,
        )
        self.text_encoder = TextEncoder(
            model_name=text_encoer_name,
            proj_dim=proj_dim,
            dropout=dropout,
            pooling=text_pooling,
            num_layers=text_num_layers,
            nhead=text_nhead,
            ffn_mult=text_ffn_mult,
        )

        # CLIP-style learnable logit scale
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

    def compute_logits(
        self,
        skeleton_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        skeleton_emb: (B, D), normalized
        text_emb: (B, D), normalized
        """
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * (skeleton_emb @ text_emb.t())  # (B, B)
        return logits

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Symmetric CLIP loss.
        """
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)
        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_s2t + loss_t2s)
    def encode_skeleton(
        self,
        skeleton: torch.Tensor,
        skeleton_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sk_out = self.skeleton_encoder(
            skeleton=skeleton,
            skeleton_mask=skeleton_mask,
            return_sequence=True,
        )
        return sk_out
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tx_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return tx_out
    def no_train_text_encoder(self):
        for param in self.text_encoder.embed_model.parameters():
            param.requires_grad = False
    def forward(
        self,
        skeleton: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        skeleton_lengths: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        skeleton_mask=create_mask(skeleton_lengths, skeleton.size(1)).bool() if skeleton_lengths is not None else None
        sk_out = self.skeleton_encoder(
            skeleton=skeleton,
            skeleton_mask=skeleton_mask,
            return_sequence=return_features,
        )
        tx_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        skeleton_emb = sk_out["embeddings"]  # (B, D)
        text_emb = tx_out["embeddings"]      # (B, D)

        logits = self.compute_logits(skeleton_emb, text_emb)
        loss = self.contrastive_loss(logits)

        out = {
            "loss": loss,
            "logits_per_skeleton": logits,
            "logits_per_text": logits.t(),
            "skeleton_embeddings": skeleton_emb,
            "text_embeddings": text_emb,
        }

        if return_features:
            out["skeleton_sequence_features"] = sk_out["sequence_features"]
            out["text_token_features"] = tx_out["token_features"]

        return out


if __name__ == "__main__":
    # =========================
    # Example usage
    # =========================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dummy setting
    batch_size = 4
    num_frames = 64
    num_joints = 50
    joint_dim = 3

    # Dummy skeleton input: (B, T, J, C)
    skeleton = torch.randn(batch_size, num_frames, num_joints, joint_dim).to(device)

    # Example padding mask for skeleton
    # True = padding, False = valid
    skeleton_mask = torch.zeros(batch_size, num_frames, dtype=torch.bool).to(device)
    skeleton_mask[0, 50:] = True
    skeleton_mask[1, 60:] = True

    # Example texts
    texts = [
        "a person waves with the right hand",
        "someone raises both hands",
        "a signer moves the hand to the face",
        "a person points to the left side",
    ]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    config={
        "num_joints": num_joints,
        "joint_dim": joint_dim,
        "skeleton_d_model": 256,
        "skeleton_nhead": 8,
        "skeleton_num_layers": 4,
        "skeleton_ff_dim": 1024,
        "bert_name": "bert-base-uncased",
        "proj_dim": 256,
        "text_pooling": "cls",
        "init_temperature": 0.07,
    }
    model = SkeletonTextCLIP(
        config=config,
    ).to(device)

    outputs = model(
        skeleton=skeleton,
        skeleton_mask=skeleton_mask,
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
        token_type_ids=text_inputs.get("token_type_ids", None),
        return_features=True,
    )

    print("loss:", outputs["loss"].item())
    print("logits_per_skeleton shape:", outputs["logits_per_skeleton"].shape)
    print("skeleton_embeddings shape:", outputs["skeleton_embeddings"].shape)
    print("text_embeddings shape:", outputs["text_embeddings"].shape)