import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
import numpy as np
import torch.nn.functional as F
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig



config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Base", trust_remote_code=True)
processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Base")
model = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-Base', config=config, trust_remote_code=True)


video = list(np.random.rand(16, 3, 224, 224))




# B, T, C, H, W -> B, C, T, H, W
inputs = processor(video, return_tensors="pt")
inputs['pixel_values'] = inputs['pixel_values'].permute(0, 2, 1, 3, 4)

with torch.no_grad():
  outputs = model(**inputs)

class VideoMAEv2Small(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 num_frames=16,
                 tubelet_size=2,
                 in_chans=3,
                 embed_dim=384,
                 depth=12,
                 num_heads=6,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.):
        super().__init__()

        # 1. Patch Embedding (時空間パッチ化)
        # Tubelet embedding: 時間方向にも畳み込みを行い圧縮
        self.patch_embed = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )

        num_patches = (img_size // patch_size) * (img_size // patch_size) * (num_frames // tubelet_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 重みの初期化
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x, mask=None):
        # x shape: (Batch, C, T, H, W)
        x = self.patch_embed(x)  # (B, C, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, L, C)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # マスク処理（学習時のみ使用）
        if mask is not None:
            x = x[~mask].reshape(x.shape[0], -1, x.shape[-1])

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x
    def load_pretrained_weights(self, checkpoint_path):
        """
        model: VideoMAEv2Smallのインスタンス
        checkpoint_path: .pth または .bin ファイルへのパス
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # モデルのステート辞書を取得（'model'キーの中に格納されている場合が多い）
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 現在のモデルのキー一覧
        model_state_dict = self.state_dict()

        # ロードする重みを整形
        new_state_dict = {}
        for k, v in state_dict.items():
            # Encoder部分（'patch_embed'や'blocks'）のみを抽出
            if k in model_state_dict:
                # 位置エンコーディング(pos_embed)のサイズが合わない場合の補間
                if k == 'pos_embed' and v.shape != model_state_dict[k].shape:
                    print(f"Interpolating {k} from {v.shape} to {model_state_dict[k].shape}")
                    # (1, L, C) -> (1, C, L) に変換して補間
                    v_interp = v.transpose(1, 2)
                    v_interp = F.interpolate(
                        v_interp,
                        size=model_state_dict[k].shape[1],
                        mode='linear',
                        align_corners=False
                    )
                    v = v_interp.transpose(1, 2)

                new_state_dict[k] = v
            else:
                # decoder等のキーは無視してOK
                continue

        # 重みをロード
        msg = self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded weights with result: {msg}")




class MaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches = self.frames * self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __call__(self):
        # Tube Masking: 時間軸(T)を固定し、空間(H, W)に対してマスクを決定
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches // self.frames - self.num_mask // self.frames),
            np.ones(self.num_mask // self.frames),
        ])
        np.random.shuffle(mask_per_frame)

        # 全フレームに同じマスクパターンを適用
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask  # 1がマスク、0が可視


class VideoMAEv2Decoder(nn.Module):
    def __init__(self, num_patches, patch_size=16, tubelet_size=2,
                 embed_dim=384, decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=3):
        super().__init__()

        # Encoderの次元からDecoderの次元へ変換
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(dim=decoder_embed_dim, num_heads=decoder_num_heads, mlp_ratio=4.)
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        # 出力層: パッチ内の全ピクセルを予測 (C * T_size * P_size * P_size)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 3 * tubelet_size * patch_size ** 2)

    def forward(self, x, ids_restore):
        # 1. 次元変換
        x = self.decoder_embed(x)

        # 2. マスクトークンを補完して元の長さに戻す
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        # 元の順序に並び替え
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        # 3. 位置エンコーディング追加
        x = x + self.decoder_pos_embed

        # 4. Decoder Block適用
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # 5. ピクセル値への投影
        x = self.decoder_pred(x)
        return x



        return
# 動作確認用
# model = VideoMAEv2Small()
# dummy_video = torch.randn(1, 3, 16, 224, 224) # (B, C, T, H, W)
# output = model(dummy_video)
# print(f"Output shape: {output.shape}")