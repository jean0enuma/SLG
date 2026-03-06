import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
def display_attn_map(model, x,config,save_path):
    frame=100
    # Encoderの最後のattention mapを取得
    attn = model.get_last_selfattention(x[frame].unsqueeze(0)).to("cpu")

    # Nはパッチの数
    # (1, num_head, N+1, N+1) -> (num_head, N)
    num_head = config["enc_heads"]
    attn = attn[0, :, 0, 1:].reshape(num_head, -1)  # cls tokenに対するスコアを抽出

    val, idx = torch.sort(attn)  # スコアを昇順でソート
    val /= torch.sum(val, dim=1, keepdim=True)  # スコアを[0-1]で正規化する

    # 累積和をとりスコアの合計が0.6ほどになるように残す
    cumval = torch.cumsum(val, dim=1)
    attn = cumval > (1 - 0.8)
    backward_indexes = torch.argsort(idx)

    # ソートしたものを戻す
    for head in range(num_head):
        attn[head] = attn[head][backward_indexes[head]]

    # スコアを画像の形にする
    w_featmap, h_featmap = config["image_size"][0] // config["patch_size"][0], config["image_size"][1] // config["patch_size"][1]
    attn = attn.reshape(num_head, h_featmap, w_featmap).float()

    # 入力画像と同じ大きさにする
    attn = nn.functional.interpolate(attn.unsqueeze(0), scale_factor=config["patch_size"][0], mode="nearest")[0].detach().cpu().numpy()

    # 入力画像とヘッドごとのattention mapを出力する
    fig = plt.figure(figsize=(20,20))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05,
                        wspace=0.05)

    img = (x[frame].data + 1) / 2
    img = np.clip(np.transpose(torch.squeeze(img.to("cpu")).numpy(), (1, 2, 0)), 0, 1)
    ax = fig.add_subplot(2, 16, 1, xticks=[], yticks=[])
    ax.imshow(img)
    average_map=np.mean(attn,axis=0)
    average_map/=np.max(average_map)
    #ヒートマップを作る
    ax = fig.add_subplot(2, 16, 2, xticks=[], yticks=[])
    ax.imshow(img)
    ax.imshow(average_map, alpha=0.5)


    for i in range(len(attn)):
        featmap = attn[i]
        featmap = np.concatenate((featmap[:,:,np.newaxis], np.zeros((224, 224, 2))), axis=2)
        ax = fig.add_subplot(2, 16, i+17, xticks=[], yticks=[])
        ax.imshow(img)
        ax.imshow(featmap, alpha=0.5)
    plt.savefig(f"{save_path}/attention_map.png")
    return attn,img
