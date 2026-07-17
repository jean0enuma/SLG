import torch
import string
import lightning as L
import torch.utils.data as data

from tqdm import tqdm
from typing import Union
from pathlib import Path

from plot import plot_pose
from helpers import pad_list_of_tensors_1d
from dataset_vq import pose_sequence_to_segments
from constants import (
    init_vocab,
    PAD_ID,
    BOS_ID,
    EOS_ID,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
)


class TranslationDataModule(L.LightningDataModule):
    def __init__(self, data_config, vq_config):
        super().__init__()
        self.data_config = data_config

        self.train = None
        self.dev = None
        self.test = None

        try:
            self.data_path = Path(data_config["data_path"])
            self.train_batch_size = data_config["train_batch_size"]
            self.dev_batch_size = data_config["dev_batch_size"]

            self.train_subset = data_config["train_subset"]
            self.val_subset = data_config["val_subset"]
            self.test_subset = data_config["test_subset"]

            self.shuffle = data_config["shuffle"]
            self.min_length = data_config["min_length"]
            self.max_length = data_config["max_length"]
            self.num_workers = data_config["num_workers"]
            self.dataset = None
            self.subsample = int(vq_config["subsample"])
            self.window_size = int(vq_config["window_size"])
            self.stride = int(vq_config["stride"])
            self.dataset = PhixDataset

        except KeyError as e:
            print(f"Error: {e} not found in data_config")
            raise e

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "train":
            self.train = self.dataset(
                self.data_path / "train.pt",
                split="train",
                subset=self.train_subset,
                min_length=self.min_length,
                max_length=self.max_length,
                subsample=self.subsample,
                window_size=self.window_size,
                stride=self.stride,
            )
            self.train.load_data()

        # Assign test dataset for use in dataloader(s)
        elif stage == "test":
            self.test = self.dataset(
                self.data_path / "test.pt",
                split="test",
                subset=self.test_subset,
                min_length=self.min_length,
                max_length=self.max_length,
                text_vocab=self.train.text_vocab,
                gloss_vocab=self.train.gloss_vocab,
                subsample=self.subsample,
                window_size=self.window_size,
                stride=self.stride,
            )
            self.test.load_data()

        elif stage == "dev":
            self.dev = self.dataset(
                self.data_path / "dev.pt",
                split="dev",
                subset=self.val_subset,
                min_length=self.min_length,
                max_length=self.max_length,
                text_vocab=self.train.text_vocab,
                gloss_vocab=self.train.gloss_vocab,
                subsample=self.subsample,
                window_size=self.window_size,
                stride=self.stride,
            )
            self.dev.load_data()
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def save_vocab(self, save_dir):
        self.train.save_vocab(save_dir=save_dir)

    def get_dataloader(self, stage: str):
        if stage == "train":
            dataset = self.train
            batch_size = self.train_batch_size
            shuffle = self.shuffle
        elif stage == "dev":
            dataset = self.dev
            batch_size = self.dev_batch_size
            shuffle = False
        elif stage == "test":
            dataset = self.test
            batch_size = self.dev_batch_size
            shuffle = False
        else:
            raise ValueError(f"Unknown stage: {stage}")

        return dataset.make_iter(
            batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers
        )


class PhixDataset(data.Dataset):
    def __init__(
        self,
        data_dir: Path = None,
        split: str = "train",
        subset: int = None,
        min_length: int = 0,
        max_length: int = 0,
        lp_filter: int = 0,
        text_vocab: set = None,
        gloss_vocab: set = None,
        subsample: int = 1,
        window_size: int = 4,
        stride: int = 4,
    ):
        self.split = split
        self.subset = subset
        self.min_length = min_length
        self.max_length = max_length

        self.n_sequences = None
        self.text_vocab = {} if text_vocab is None else text_vocab
        self.gloss_vocab = {} if gloss_vocab is None else gloss_vocab

        # load data
        self.pose_dim = None
        self.lp_filter = lp_filter
        self.input_size = None
        self.output_size = None
        self.data = None
        self.idx_to_name = None
        self.data_dir = data_dir
        self.fps = 25 // subsample
        self.codebook_size = None
        self.subsample = subsample
        self.window_size = window_size
        self.stride = stride

    def load_data(self):
        data = torch.load(self.data_dir)

        self.n_sequences = 0
        self.data = {}
        for id, info in tqdm(data.items(), desc="Loading data..."):

            # load and formate the text
            text = info["text"].lower()
            text = text.replace("-", " ")
            remove_chars = (
                string.punctuation.replace(".", "") + "„“…–’‘”‘‚´" + "0123456789€"
            )  # add additional punctuation
            text = "".join(ch for ch in text if ch not in remove_chars)
            if text == "":
                continue

            text = text.split()
            gloss = info["gloss"].split()
            pose_seq = info["poses_3d"][:: self.subsample, ...].flatten(-2, -1)

            if self.pose_dim is None:
                self.pose_dim = pose_seq.shape[-1]

            # filter training set
            if self.split == "train":
                if len(text) < self.min_length or len(text) > self.max_length:
                    continue
                if len(gloss) < self.min_length or len(gloss) > self.max_length:
                    continue

            # add to vocabulary
            if self.split == "train":
                self.add_word(text)
                self.add_gloss(gloss)
            else:
                # add UNK token
                vocab = self.text_vocab.values()
                text = [w if w in vocab else UNK_TOKEN for w in text]
                vocab = self.gloss_vocab.values()
                gloss = [w if w in vocab else UNK_TOKEN for w in gloss]

            # add BOS and EOS tokens
            text = text + [EOS_TOKEN]
            gloss = [BOS_TOKEN] + gloss + [EOS_TOKEN]

            self.data[self.n_sequences] = {
                "text": text,
                "gloss": gloss,
                "seq_pose": pose_seq,
                "name": id,
            }
            self.n_sequences += 1

            if self.n_sequences >= self.subset != -1:
                break

        if self.split == "train":
            self.make_vocab()
            self.input_size = len(self.text_vocab)

        print(f"Loaded {self.n_sequences} sequences from {self.data_dir}")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        return self.data[idx]

    def quantize_data(self, vq_model, make_batch, batch_size=32, device="cpu"):
        # set the output size to codebook size plus the special tokens
        self.output_size = vq_model.codebook.num_embeddings + len(init_vocab)

        for id, data in self.data.items():
            # convert the pose sequence to segments
            segments = pose_sequence_to_segments(
                data["seq_pose"], window_size=self.window_size, stride=self.stride
            )
            batch = make_batch(segments)
            batch = {k: v.to(device) for k, v in batch.items()}
            encoder_output = vq_model.encode(**batch)
            encoder_output = encoder_output.flatten(-2, -1)
            index = vq_model.codebook.query_codebook(encoder_output)
            # shift tokens up by 4 to avoid special tokens
            index = index + 4
            self.data[id]["vq"] = index.tolist()

    def remove_pose(self):
        # remove pose from dataset to save memory
        for sentence in self.data.values():
            sentence["seq_pose"] = None

    def add_word(self, word):
        def add(x):
            if x not in self.text_vocab:
                self.text_vocab[x] = 1
            else:
                self.text_vocab[x] += 1

        if isinstance(word, list):
            for w in word:
                add(w)
        else:
            add(word)

    def add_gloss(self, gloss):
        def add(x):
            if x not in self.gloss_vocab:
                self.gloss_vocab[x] = 1
            else:
                self.gloss_vocab[x] += 1

        if isinstance(gloss, list):
            for g in gloss:
                add(g)
        else:
            add(gloss)

    def make_vocab(self):
        _text_vocab = init_vocab.copy()
        _gloss_vocab = init_vocab.copy()

        self.text_vocab = sorted(
            self.text_vocab.items(), key=lambda item: item[1], reverse=True
        )
        for t in self.text_vocab:
            _text_vocab[len(_text_vocab)] = t[0]

        self.gloss_vocab = sorted(
            self.gloss_vocab.items(), key=lambda item: item[1], reverse=True
        )
        for g in self.gloss_vocab:
            _gloss_vocab[len(_gloss_vocab)] = g[0]

        self.text_vocab = _text_vocab
        self.gloss_vocab = _gloss_vocab

    def save_vocab(self, save_dir: Union[str, Path]):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "text_vocab.txt", "w") as f:
            f.write("\n".join(list(self.text_vocab.values())))
        with open(save_dir / "gloss_vocab.txt", "w") as f:
            f.write("\n".join(list(self.gloss_vocab.values())))

    def make_iter(
        self, batch_size: int, shuffle: bool = False, num_workers: int = 0
    ) -> torch.utils.data.DataLoader:

        data_loader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        return data_loader

    def collate_fn(self, batch):
        """
        Collate function for the mdgs dataset, apply data normalization
        :param batch: list of 3D poses (N x K x 3)
        :return:
        """
        batch = list(zip(*[b.values() for b in batch]))
        text, gloss, seq_pose, name, vq_codes = batch

        mapping = {v: k for k, v in self.text_vocab.items()}
        src = self.tokenize_batch(text, mapping)
        src, src_len = pad_list_of_tensors_1d(
            src, padding_value=PAD_ID, dtype=torch.int32
        )  # pad the source
        src_mask = (src != PAD_ID).unsqueeze(-2)  # make mask

        trg = [
            torch.tensor([BOS_ID] + codes + [EOS_ID], dtype=torch.int32)
            for codes in vq_codes
        ]
        trg, trg_len = pad_list_of_tensors_1d(
            trg, padding_value=PAD_ID, dtype=torch.int64
        )  # pad the source

        trg_input = trg[:, :-1]  # shape (batch_size, seq_length)
        trg_len = trg_len
        trg = trg[:, 1:]  # shape (batch_size, seq_length)
        # exclude the padded areas (and blank areas) from the loss computation
        trg_mask = (trg != PAD_ID).unsqueeze(1)
        ntokens = (trg != PAD_ID).data.sum().item()

        batch = {
            "name": name,
            "src": src,
            "src_text": text,
            "src_length": src_len,
            "src_mask": src_mask,
            "trg": trg,
            "trg_input": trg_input,
            "trg_length": trg_len,
            "trg_mask": trg_mask,
            "gloss": gloss,
            "ntokens": ntokens,
        }
        return batch

    def tokenize_batch(self, b, mapping):
        return [
            torch.tensor([mapping[w] for w in text], dtype=torch.int32) for text in b
        ]

    def de_tokenize_text(self, b):
        return [
            [self.text_vocab[w] for w in text if w != EOS_ID or w != PAD_ID]
            for text in b
        ]

    def de_tokenize_gloss(self, b):
        return [
            [self.gloss_vocab[w] for w in text if w != EOS_ID or w != PAD_ID]
            for text in b
        ]

    def get_pose(self):
        return [d["seq_pose"] for d in self.data.values()]

    def get_text(self):
        text = []
        for d in self.data.values():
            if d["text"][-1] == EOS_TOKEN:
                text.append(" ".join(d["text"][:-1]))
            else:
                text.append(" ".join(d["text"]))
        return text

    def get_gloss(self):
        gloss = []
        for d in self.data.values():
            if d["gloss"][-1] == EOS_TOKEN and d["gloss"][0] == BOS_TOKEN:
                gloss.append(" ".join(d["gloss"][1:-1]))
            elif d["gloss"][-1] == EOS_TOKEN:
                gloss.append(" ".join(d["gloss"][:-1]))
            elif d["gloss"][0] == BOS_TOKEN:
                gloss.append(" ".join(d["gloss"][1:]))
            else:
                gloss.append(" ".join(d["gloss"]))
        return gloss

    def get_name(self):
        return [d["name"] for d in self.data.values()]

    def get_plain_gloss(self):
        return [d["plain_gloss"] for d in self.data.values()]

    def plot_pose(self, pose):
        fig = plot_pose(pose)
        return fig


if __name__ == "__main__":
    from helpers import load_config

    # test and load a dataset
    config = load_config("config/translation/translation_config.yaml")
    vq_config = load_config("config/codebook/codebook_config.yaml")
    dataset = TranslationDataModule(config["data"], vq_config["data"])
    dataset.setup("train")
    dataset.setup("test")
    dataset.setup("dev")
    test_dataloader = dataset.test_dataloader()

    print(dataset)
    print("Done")
