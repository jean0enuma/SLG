import torch
import torch.nn as nn
import math

class EncoderOutput(object):
    def __init__(self, sentence_embedding, last_hidden_state=None):
        self.sentence_embedding = sentence_embedding  # [batch_size, d_model]
        self.last_hidden_state =  last_hidden_state    # [batch_size, seq_len, d_model] (オプション)
class TextTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3, dim_feedforward=512, max_len=512,padding_token_id=0):
        super(TextTransformerEncoder, self).__init__()

        # 1. 単語をベクトル化
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_token_id)

        # 2. 位置情報を付与（Transformerは並び順を知らないため）
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))

        # 3. Transformerのメイン層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True  # [batch, seq, feature] の順にする
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.d_model = d_model
    #引数はBatchEncodingを想定(input_idsとattention_maskを受け取る)
    def forward(self, input_ids, attention_mask):
        # input_ids: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len] (1:有効, 0:PAD)

        # Tensor変換
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.float)

        seq_len = input_ids.size(1)

        # Embedding + Positional Encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :]

        # PyTorchのTransformerは「無視する場所をTrue」にする必要があるため反転させる
        # src_key_padding_mask: [batch_size, seq_len]
        key_padding_mask = (attention_mask == 0)

        # Transformer実行
        encoded_output = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        # 平均プーリング（文全体のベクトルを作成）
        mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        masked_output = encoded_output * mask
        sentence_embedding = torch.sum(masked_output, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1e-9)

        return EncoderOutput(sentence_embedding=sentence_embedding, last_hidden_state=encoded_output)
if __name__ == "__main__":
    # --- 連携プロセス ---
    from loader.word_tokenizer import SimpleTokenizer
    import pandas as pd
    # 0. 前回のtokenizer準備
    data = {'annotation': ["I love pandas", "pandas is powerful", "I love python too"]}
    df = pd.DataFrame(data)

    # 1. 初期化と学習
    tokenizer = SimpleTokenizer()
    tokenizer.fit(df['annotation'])

    model = TextTransformerEncoder(vocab_size=tokenizer.vocab_size)

    encoded_inputs = tokenizer(["I love pandas", "Python is powerful"], padding=True)
    with torch.no_grad():
        vector = model(encoded_inputs["input_ids"], encoded_inputs["attention_mask"],return_all=True)

    print("出力ベクトルの形状:", vector.shape)  # [2, 256]