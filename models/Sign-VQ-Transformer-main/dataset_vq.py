import torch
import lightning as L
import torch.utils.data as data

from tqdm import tqdm
from typing import Union
from pathlib import Path

from helpers import pad_list_of_tensors_2d, create_transformer_mask


class CodebookDataModule(L.LightningDataModule):
    def __init__(self, data_config, cuda: str = "cuda", save_path: Path = None):
        super().__init__()
        self.data_config = data_config

        self.train = None
        self.test = None
        self.dev = None

        try:
            self.data_dir = Path(data_config["data_path"])

            self.window_size = data_config["window_size"]
            self.stride = data_config["stride"]

            self.train_batch_size = data_config["train_batch_size"]
            self.test_batch_size = data_config["test_batch_size"]

            self.train_subset = data_config["train_subset"]
            self.val_subset = data_config["val_subset"]
            self.test_subset = data_config["test_subset"]

            self.model_dir = save_path

            self.shuffle = data_config["shuffle"]
            self.num_workers = data_config["num_workers"]
            self.subsample = data_config["subsample"]

            self.dataset = None
            self.cuda = cuda

            dataset_name = data_config["dataset_name"]
            if dataset_name.lower() == "phix":
                self.dataset = poseDataset
                self.fps = 25
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        except KeyError as e:
            print(f"Error: {e} not found in data_config")
            raise e

    def setup(self, stage: str):
        if stage == "train":
            self.train = self.dataset(
                self.data_dir / "train.pt",
                split=stage,
                window_size=self.window_size,
                stride=self.stride,
                subset=self.train_subset,
                subsample=self.subsample,
                fps=self.fps,
                cuda=self.cuda,
                dir=self.model_dir,
            )
            self.train.load_data()

        elif stage == "dev":
            self.dev = self.dataset(
                self.data_dir / "dev.pt",
                split=stage,  # test
                window_size=self.window_size,
                stride=self.stride,
                subset=self.train_subset,
                subsample=self.subsample,
                fps=self.fps,
                cuda=self.cuda,
                dir=self.model_dir,
            )
            self.dev.load_data()

        elif stage == "test":
            self.test = self.dataset(
                self.data_dir / "test.pt",
                split=stage,  # dev
                window_size=self.window_size,
                stride=self.stride,
                subset=self.train_subset,
                subsample=self.subsample,
                fps=self.fps,
                cuda=self.cuda,
                dir=self.model_dir,
            )
            self.test.load_data()
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def get_dataloader(self, stage: str):
        if stage == "train":
            pose_dataset = self.train
            batch_size = self.train_batch_size
            shuffle = self.shuffle
        elif stage == "dev":
            pose_dataset = self.dev
            batch_size = self.train_batch_size
            shuffle = False
        elif stage == "test":
            pose_dataset = self.test
            batch_size = self.train_batch_size
            shuffle = False
        else:
            raise ValueError(f"Unknown stage: {stage}")

        return pose_dataset.get_dataloader(
            batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers
        )


class poseDataset(data.Dataset):
    def __init__(
        self,
        data_path: Union[str, Path] = None,
        split: str = "train",
        window_size: int = 8,
        stride: int = 4,
        subset: int = None,
        subsample: int = None,
        fps: int = 25,
        cuda: str = "cuda",
        dir: Path = None,
    ):
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.subset = subset
        self.subsample = subsample

        # load data
        self.data = None
        self.data_path = data_path
        self.n_sequences = None

        self.input_size = None

        self.annotations = {}
        self.idx_to_segments = {}

        self.fps = fps // subsample
        self.cuda = True if cuda == "cuda" else False
        self.dir = dir

    def load_data(self):
        self.data = torch.load(self.data_path)

        self.n_sequences = 0
        _data = []
        for i, (k, v) in tqdm(
            enumerate(self.data.items()), total=len(self.data), desc="Loading data:"
        ):
            poses = v["poses_3d"][:: self.subsample, ...]
            # filter out sequence that are less then window size
            if poses.shape[0] < self.window_size:
                self.idx_to_segments[i] = None
                continue

            poses = pose_sequence_to_segments(
                poses, window_size=self.window_size, stride=self.stride
            )

            # track the index of the sequence
            self.idx_to_segments[i] = (
                self.n_sequences,
                self.n_sequences + poses.shape[0],
            )

            if poses.shape[0] != 0:
                self.n_sequences += poses.shape[0]
                _data.append(poses)

            if len(_data) >= self.subset != -1:
                break

        _data = torch.cat(_data, dim=0)
        _data = torch.flatten(_data, -2, -1)
        self.n_sequences = len(_data)
        self.input_size = _data[0].shape[-1]

        if self.n_sequences >= self.subset != -1:
            self.data = self.data[: self.subset]
            self.n_sequences = self.subset

        self.data = _data
        print(f"Loaded {self.n_sequences} from {self.split}")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        return self.data[idx]

    def get_dataloader(
        self, batch_size: int, shuffle: bool = False, num_workers: int = 0
    ) -> torch.utils.data.DataLoader:

        data_loader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=make_batch,
        )
        return data_loader

    def get_test_batch(self, get_n: int):
        import random

        seq_test_seq = random.sample(range(len(self)), len(self))
        ids = seq_test_seq[:get_n]
        batch = [self.__getitem__(id) for id in ids]
        batch = make_batch(batch)
        return batch


def make_batch(signs):

    signs, src_length = pad_list_of_tensors_2d(signs)

    src_mask = create_transformer_mask(src_length).unsqueeze(1)
    src_mask = src_mask.to(torch.bool)

    counter = make_counter(src_length)

    assert not torch.isnan(signs).any()
    assert not torch.isnan(src_mask).any()
    assert not torch.isnan(src_length).any()
    if torch.isnan(counter).any():
        print("Counter has nan values")
    assert not torch.isnan(counter).any()

    batch = {
        "src": signs,
        "src_mask": src_mask,
        "src_length": src_length,
        "trg_input": counter,
    }

    return batch


def make_counter(counter_length: torch.Tensor):
    counter = [
        torch.tensor(
            [i / (l - 1) for i in range(int(l.item()))], dtype=torch.float32
        ).unsqueeze(-1)
        for l in counter_length
    ]
    counter, _ = pad_list_of_tensors_2d(counter, padding_value=1.0, dtype=torch.float32)
    return counter


def pose_sequence_to_segments(
    sequence: torch.Tensor, window_size: int = 8, stride: int = 4
):
    """
    Convert a pose sequence into segments of fixed size with a given stride.
    """
    # apply stride and window to data
    num_batches = (sequence.shape[0] - window_size) // stride + 1
    # Calculate the starting indices for each batch
    batch_start_indices = torch.arange(0, num_batches * stride, stride)
    # Use advanced indexing to extract the batches
    segments = sequence[batch_start_indices.unsqueeze(1) + torch.arange(window_size)]

    return segments


if __name__ == "__main__":
    from helpers import load_config

    # test and load a dataset
    config_path = "./config/codebook/codebook_config.yaml"
    config = load_config(config_path)
    dataset = CodebookDataModule(config["data"])
    dataset.setup("test")
    test_dataloader = dataset.get_dataloader("test")
    batch = next(iter(test_dataloader))

    print(dataset)
    print("Done")
