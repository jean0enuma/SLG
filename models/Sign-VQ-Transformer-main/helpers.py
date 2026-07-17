import os
import csv
import yaml
import torch
import pickle

from pathlib import Path
from torch import Tensor, nn
from typing import Dict, Union, Any
from constants import EOS_ID, PAD_ID


def load_json_file(data_path: Union[str, Path]) -> dict:
    with open(data_path, "r") as handle:
        json_data = yaml.safe_load(handle)
    return json_data


def load_pickle_file(data_path: Union[str, Path]) -> dict:
    with open(data_path, "rb") as handle:
        csv_data = pickle.load(handle)
    return csv_data


def save_pickle_file(data: Any, path: Union[str, Path]) -> None:
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_csv(file_path, delimiter="|"):
    with open(file_path, "r", newline="") as file:
        csv_reader = csv.reader(file, delimiter=delimiter)
        csv_content = [row for row in csv_reader]
    return csv_content


def save_csv(file_path, data):
    with open(file_path, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(data)


def load_text_file(file_path):
    """
    Reads the contents of a text file and returns them as a string.
    """
    try:
        with open(file_path, "r") as file:
            content = file.readlines()
        content = [line.strip() for line in content]
        return content
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")
        return None


def save_text_file(file_path, content):
    """
    Saves the provided content to a text file.
    """
    try:
        if content[0][-1] != "\n":
            content = [line + "\n" for line in content]
        with open(file_path, "w") as file:
            file.writelines(content)
        # print(f"File '{file_path}' successfully saved.")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")


def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    ones = torch.ones(size, size, dtype=torch.bool)
    return torch.tril(ones, out=ones).unsqueeze(0)


def save_config(config: dict, save_dir: Path) -> None:
    import copy

    local_config = copy.deepcopy(config)
    # Remove non valid datatypes
    for (k, v) in local_config.items():
        if not isinstance(v, (float, int, str, list, dict, tuple)):
            local_config.update({k: str(v)})

    with open(save_dir, "w", encoding="utf-8") as ymlfile:
        yaml.safe_dump(
            local_config,
            ymlfile,
            indent=4,
            default_flow_style=False,
            sort_keys=False,
            width=1000,
        )


def load_config(path: Union[Path, str] = "configs/default.yaml") -> Dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    if isinstance(path, str):
        path = Path(path)
    with path.open("r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


class ConfigurationError(Exception):
    """Custom exception for misspecifications of configuration"""


def dict_to_string(d, indent=0):
    result = ""
    for key, value in d.items():
        result += " " * indent + str(key) + ": "
        if isinstance(value, dict):
            result += "\n" + dict_to_string(value, indent + 2)
        else:
            result += str(value) + "\n"
    return result


def pad_list_of_tensors_2d(tensors, padding_value=0, dtype=torch.float32):
    """
    Pad the first dimension of a list of tensors to the same maximum length.

    Args:
    - tensors (list): List of PyTorch tensors with the same size along the first dimension.
    - max_length (int): The maximum length to pad the tensors to.
    - padding_value (int): The value to use for padding.

    Returns:
    - padded_tensors (Tensor): A stacked tensor with padded sequences.
    """
    try:
        max_length = max([t.size(0) for t in tensors])
    except:
        return None, None

    # Find the length of each tensor in the list
    lengths = torch.tensor([t.size(0) for t in tensors], dtype=dtype)
    last_dim = tensors[0].size(1)

    # Pad each tensor to the max length
    padded_tensors = [
        torch.cat(
            [
                tensor.to(dtype),
                torch.full((max_length - len(tensor), last_dim), padding_value, dtype=dtype),
            ]
        )
        for tensor in tensors
    ]

    # Stack the padded tensors along the first dimension
    padded_tensors = torch.stack(padded_tensors, dim=0)

    return padded_tensors, lengths


def pad_list_of_tensors_1d(tensors, padding_value=0, dtype=torch.float32):
    """
    Pad the first dimension of a list of tensors to the same maximum length.

    Args:
    - tensors (list): List of PyTorch tensors with the same size along the first dimension.
    - max_length (int): The maximum length to pad the tensors to.
    - padding_value (int): The value to use for padding.

    Returns:
    - padded_tensors (Tensor): A stacked tensor with padded sequences.
    """
    try:
        max_length = max([t.size(0) for t in tensors])
    except:
        return None, None

    # Find the length of each tensor in the list
    lengths = torch.tensor([t.size(0) for t in tensors], dtype=dtype)

    # Pad each tensor to the max length
    padded_tensors = [
        torch.cat(
            [
                tensor.to(dtype),
                torch.full((max_length - len(tensor),), padding_value, dtype=dtype),
            ]
        )
        for tensor in tensors
    ]

    # Stack the padded tensors along the first dimension
    padded_tensors = torch.stack(padded_tensors, dim=0)
    return padded_tensors, lengths


def create_transformer_mask(sequence_lengths):
    # Initialize the mask with zeros
    mask = torch.ones(len(sequence_lengths), int(sequence_lengths.max().item()))
    # Set elements to -inf where the sequence is padded
    for idx, length in enumerate(sequence_lengths):
        mask[idx, int(length.item()):] = False
    return mask


def splice_tensor_efficient(data, lengths):
    """
    Efficiently splits a padded tensor based on lengths.
    Args:
        data (torch.Tensor): Padded tensor of shape (N, L)
        lengths (torch.Tensor): Tensor of lengths for each sequence (shape (N,))
    Returns:
        list: List of tensors, each containing a spliced sequence
    """
    indices = torch.arange(data.size(1)).unsqueeze(0).expand(data.size(0), -1)
    mask = indices < lengths.unsqueeze(1)
    return [row[mask[i]] for i, row in enumerate(data)]


def adjust_mask_size(mask: Tensor, batch_size: int, hyp_len: int) -> Tensor:
    """
    Adjust mask size along dim=1. used for forced decoding (trg prompting).

    :param mask: trg prompt mask in shape (batch_size, hyp_len)
    :param batch_size:
    :param hyp_len:
    """
    if mask is None:
        return None

    if mask.size(1) < hyp_len:
        _mask = mask.new_zeros((batch_size, hyp_len))
        _mask[:, :mask.size(1)] = mask
    elif mask.size(1) > hyp_len:
        _mask = mask[:, :hyp_len]
    else:
        _mask = mask
    assert _mask.size(1) == hyp_len, (_mask.size(), batch_size, hyp_len)
    return _mask


def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    # yapf: disable
    x = (x.view(batch, -1)
         .transpose(0, 1)
         .repeat(count, 1)
         .transpose(0, 1)
         .contiguous()
         .view(*out_size))
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def alternate_join_list(list_a: list, list_b: list):
    x, y = 0, 0
    output_list = []
    for i in range(len(list_a) + len(list_b)):
        if i % 2 == 0:
            output_list.append(list_a[x])
            x += 1
        else:
            output_list.append(list_b[y])
            y += 1
    return output_list


def find_best_model(dir):
    files = [f for f in os.listdir(dir) if f.endswith('.ckpt') and 'model' in f]
    score = []
    for f in files:
        score.append(float(f.split('=')[1].split('-')[0]))
    best_model = files[score.index(max(score))]
    return best_model


def divide_list_and_round(number, x):
    quotient, remainder = divmod(number, x)
    result = [quotient + 1 if i < remainder else quotient for i in range(x)]
    return result


def find_n_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batch_list(data: list, batch_size: int):
    """
    Batch a list of data into chunks of size batch_size.

    :param data: list of data
    :param batch_size: size of each batch
    :return: list of batches
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def de_tokenize(b, vocab):
    # DO NOT USE + ['.'] as there is no '.' in the isolated vocab
    return [
        " ".join([vocab[w] for w in text if w != EOS_ID and w != PAD_ID]) for text in b
    ]


def remove_variant_number(name: str) -> str:
    """
    Remove the gloss variant number from the name.
    """
    new_gloss = []
    for l in name:
        if not l.isdigit():
            new_gloss.append(l)
        else:
            break
    new_gloss = "".join(new_gloss)
    return new_gloss

