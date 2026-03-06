import torch
import numpy as np
def sliding_window_attention_mask(seq_len: int,
                                  window_size: int,
                                  device=None,
                                  dtype=torch.bool):
    """
    Create an attention mask that allows each position t to attend only to
    positions in [t - window_size, t].

    Returns
    -------
    bool_mask : (T, T) tensor, dtype=bool
        True  -> masked (disallowed)
        False -> un‑masked (allowed)

    float_mask : (T, T) tensor, dtype=torch.float32
        0      -> allowed
        -inf   -> masked
    """
    # 行 = Query 位置 t, 列 = Key 位置 τ
    idx = torch.arange(seq_len, device=device)
    # 差分 (t - τ) をブロードキャストで計算
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)   # shape (T, T)

    # 未来 (diff < 0) もウィンドウ外 (diff > window_size) も mask=True
    bool_mask = (diff < 0) | (diff > window_size)
    bool_mask = bool_mask.to(dtype=dtype, device=device)

    # 加算マスク版（float）も用意しておくと便利
    float_mask = torch.zeros_like(bool_mask, dtype=torch.float32)
    float_mask.masked_fill_(bool_mask, float('-inf'))

    return bool_mask, float_mask
def create_key_padding_mask(len_x, mask_type="bool"):
    """
    Create a key padding mask.

    :param batch: The batch size.
    :param len_x: The length tensor.
    :return: The key padding mask.
    """
    len_x=len_x.int()
    mask = torch.zeros((len_x.size(0), int(len_x.max().item())))
    for idx, l in enumerate(len_x):
        mask[idx, l:] = 1
    if mask_type== "bool":
        return mask.bool()
    else:
        mask[mask==1]=float("-inf")
        return mask

def create_src_mask(size,mask_idx=None,mask_type="bool"):
    """
    create a casual mask: inspired MLM of BERT
    :param size:
    :param stride:
    :return:
    """
    mask=torch.zeros(size,size)
    if mask_idx is None:
        mask_idx=[]
    mask[:,mask_idx]=1
    if mask_type=="bool":
        return mask.bool(),mask_idx
    else:
        return mask*float("-inf"),mask_idx
def create_local_mask(x, kernel_size=7, dilation=1, dtype=torch.float, is_cls=False):
    # x:(B,T,C)
    temporal = x.size(1) + 1 if is_cls else x.size(1)
    kernel_size = kernel_size + (dilation - 1) * (kernel_size - 1) if kernel_size != -1 else -1
    padding = kernel_size // 2
    mask = torch.ones(temporal, temporal)
    start_idx = 0
    for i in range(temporal):
        if kernel_size != -1:
            if i < padding:
                id = np.arange(start_idx, kernel_size - (padding - i))
                mask[i, id[i::dilation]] = torch.zeros(kernel_size - (padding - i))[id[i::dilation]]
                mask[i, id[i::-dilation].copy()] = torch.zeros(kernel_size - (padding - i))[id[i::-dilation].copy()]
            elif i >= temporal - padding:
                id = np.arange(start_idx, temporal)
                mask[i, id[::dilation]] = torch.zeros(temporal - start_idx)[::dilation]
                start_idx += 1
            else:
                id = np.arange(start_idx, start_idx + kernel_size)
                mask[i, id[::dilation]] = torch.zeros(kernel_size)[::dilation]
                start_idx += 1

        else:
            id = np.arange(start_idx, temporal)
            mask[i, id[i::dilation]] = torch.zeros(temporal - start_idx)[id[i::dilation]]
            mask[i, id[i::-dilation].copy()] = torch.zeros(temporal - start_idx)[id[i::-dilation].copy()]

    if dtype == torch.bool:
        mask = mask.bool()
    else:
        mask[mask == 1] = float('-inf')
        mask = mask.float()
    return mask
if __name__=="__main__":
    # Example usage
    T, d = 10, 3
    bool_mask, float_mask = sliding_window_attention_mask(T, d)
    print("bool_mask:\n", bool_mask)