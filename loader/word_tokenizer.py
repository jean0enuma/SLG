import pandas as pd
import numpy as np
import torch
from transformers import BatchEncoding

class SimpleTokenizer:
    def __init__(self, pad_token="[PAD]", unk_token="[UNK]"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2id = {pad_token: 0, unk_token: 1}
        self.id2word = {0: pad_token, 1: unk_token}
        self.vocab_size = 2

    def fit(self, series):
        """Pandasのシリーズから語彙を作成する"""
        words = set()
        for text in series.dropna():
            #"."を除いて単語に分割
            update_word=str(text).replace(".", " ").split()
            words.update(update_word)

        for word in sorted(list(words)):
            if word not in self.word2id:
                self.word2id[word] = self.vocab_size
                self.id2word[self.vocab_size] = word
                self.vocab_size += 1

    def __call__(self, texts, padding=True, truncation=False, max_length=None, return_tensors=None):
        """
        HF準拠のインターフェース
        - truncation: Trueの場合、max_lengthでカット
        - return_tensors: 'pt' (PyTorch) or 'np' (NumPy)
        """
        if isinstance(texts, str):
            texts = [texts]

        # 1. ID化とTruncation（切り捨て）の適用
        all_input_ids = []
        for text in texts:
            ids = [self.word2id.get(word, self.word2id[self.unk_token])
                   for word in str(text).split() if word != "."]
            if truncation and max_length:
                ids = ids[:max_length]
            all_input_ids.append(ids)

        # 2. Padding（埋め草）の長さ決定
        if max_length is None or not padding:
            batch_max_len = max(len(ids) for ids in all_input_ids)
        else:
            batch_max_len = max_length

        final_input_ids = []
        attention_mask = []

        for ids in all_input_ids:
            # パディング処理
            pad_len = max(0, batch_max_len - len(ids))
            padded_ids = ids + [self.word2id[self.pad_token]] * pad_len
            mask = [1] * len(ids) + [0] * pad_len

            # 指定されたmax_lengthを超えている場合は念のため再度カット（padding=False時など）
            final_input_ids.append(padded_ids[:batch_max_len])
            attention_mask.append(mask[:batch_max_len])

        # 3. 型変換 (Return Tensors)
        output = {
            "input_ids": final_input_ids,
            "attention_mask": attention_mask
        }

        if return_tensors == "pt":
            output = {k: torch.tensor(v) for k, v in output.items()}
        elif return_tensors == "np":
            output = {k: np.array(v) for k, v in output.items()}

        return BatchEncoding(output)

if __name__ == "__main__":
    # --- 実行例 ---
    data = {'annotation': ["I love pandas", "pandas is great", "I love my family too ."]}
    df = pd.DataFrame(data)

    # 1. 初期化と学習
    tokenizer = SimpleTokenizer()
    tokenizer.fit(df['annotation'])

    # 2. 変換（Hugging Faceと同じ形式で返る）
    output = tokenizer(df['annotation'].tolist(), padding=True, truncation=True, max_length=5, return_tensors="pt")

    print("語彙数:", tokenizer.vocab_size)
    print("input_ids:\n", output["input_ids"])
    print("attention_mask:\n", output["attention_mask"])