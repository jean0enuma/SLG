import torch
import torch.nn.functional as F
from collections import defaultdict, deque
def sort_connections(connections):
    #all_connectionsは元データのインデックスを表すが，データからconnectionに対応するデータを取得すると，インデックスが変わる
    #all_connectionsのインデックスを変換後のデータに対応させる
    #all_connectionsにあるインデックスを重複無しで抽出
    unique_indices = []
    for conn in connections:
        for idx in conn:
            if idx not in unique_indices:
                unique_indices.append(idx)
    #print(unique_indices)
    index_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(unique_indices)}
    #print(index_mapping)
    #all_connectionsのインデックスを変換
    sorted_connections = []
    for conn in connections:
        sorted_conn = (index_mapping[conn[0]], index_mapping[conn[1]])
        sorted_connections.append(sorted_conn)
    return sorted_connections


def build_parents_from_connections(connections, root, num_joints=None):
    """
    connections: list of (i, j)  # 無向辺
    root: root joint index
    num_joints: 指定しない場合は connections から自動推定

    return:
        parents: torch.LongTensor (J,)
    """

    # ---- joint数の推定 ----
    if num_joints is None:
        max_idx = max(max(i, j) for i, j in connections)
        num_joints = max_idx + 1

    # ---- 無向グラフ作成 ----
    graph = defaultdict(list)
    for i, j in connections:
        graph[i].append(j)
        graph[j].append(i)

    # ---- parents 初期化 ----
    parents = torch.full((num_joints,), -1, dtype=torch.long)
    visited = set()

    # ---- BFSで木を作る ----
    queue = deque([root])
    visited.add(root)
    parents[root] = -1  # root

    while queue:
        current = queue.popleft()

        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                parents[neighbor] = current
                queue.append(neighbor)

    return parents
@torch.no_grad()
def _topo_order_from_parents(parents: torch.Tensor):
    """
    parents: (J,) long, -1 for root
    return: list of joints in topological order starting from root(s)
    """
    J = parents.numel()
    children = [[] for _ in range(J)]
    roots = []
    for j in range(J):
        p = int(parents[j].item())
        if p < 0:
            roots.append(j)
        else:
            children[p].append(j)

    order = []
    stack = roots[:]
    while stack:
        u = stack.pop()
        order.append(u)
        # push children
        for v in children[u]:
            stack.append(v)
    return order, roots


@torch.no_grad()
def analytic_init_3d_from_2d(
        x2d: torch.Tensor,
        parents: torch.Tensor,
        bone_pairs: torch.Tensor,
        sym_pairs=None,
        choose_negative_z: bool = True,
):
    """
    論文 3.2 の(1)-(4)の考え方を使った「再帰的解析初期化」。
    - x,y は2Dに極力合わせる（届かないケースだけスケール）
    - z は (4) の ± 解のうち「小さい方」を選ぶ（choose_negative_z=Trueだと z0 - sqrt(...)）

    x2d: (B,T,J,2)
    parents: (J,)
    bone_pairs: (E,2) edges (parent, child)
    sym_pairs: list[tuple[int,int]]  同じ長さにしたい bone edge index のペア（任意）
    return: x3d_init (B,T,J,3), L0 (E,) 初期骨長
    """
    device = x2d.device
    B, T, J, _ = x2d.shape
    E = bone_pairs.shape[0]

    # 初期 z=0、x,yは2Dコピー
    x3d = torch.zeros((B, T, J, 3), device=device, dtype=x2d.dtype)
    x3d[..., :2] = x2d

    # 初期骨長：2D長さの時間平均（論文の「平均」寄りの初期化）
    p = bone_pairs[:, 0]
    c = bone_pairs[:, 1]
    vec2d = x2d[:, :, c, :] - x2d[:, :, p, :]  # (B,T,E,2)
    len2d = torch.sqrt((vec2d ** 2).sum(dim=-1) + 1e-8)  # (B,T,E)
    L0_bt = len2d.mean(dim=1)  # (B,E)
    L0 = L0_bt.mean(dim=0)  # (E,) グローバル初期値（簡単化）

    # 左右対称などを「同一長」に寄せる（任意）
    if sym_pairs is not None:
        L0 = L0.clone()
        for e1, e2 in sym_pairs:
            m = 0.5 * (L0[e1] + L0[e2])
            L0[e1] = m
            L0[e2] = m

    # bone index lookup: (parent,child) -> e
    # Jが大きくても動くよう dict で
    edge_to_e = {(int(p[i].item()), int(c[i].item())): i for i in range(E)}

    order, roots = _topo_order_from_parents(parents)
    # rootのzは0のまま（論文も head z=0 初期化）
    # 再帰で子を埋める
    for child in order:
        parent = int(parents[child].item())
        if parent < 0:
            continue
        e = edge_to_e.get((parent, child), None)
        if e is None:
            continue

        # 目標2D（childの2D）にできるだけ合わせつつ、骨長L0[e]を満たす3Dを作る
        # 親は既に計算済みとして扱う
        x0 = x3d[:, :, parent, 0]
        y0 = x3d[:, :, parent, 1]
        z0 = x3d[:, :, parent, 2]

        xt = x2d[:, :, child, 0]
        yt = x2d[:, :, child, 1]

        dx = xt - x0
        dy = yt - y0
        Lp = torch.sqrt(dx * dx + dy * dy + 1e-8)  # L' in paper
        L = L0[e].clamp_min(1e-6)

        # if L' > L : 届かない → (1)(2)(3) のように2D方向へスケールし z=0
        # if L' <= L : x,yは目標へ、zは (4) で決める
        # ※論文の記述と比較すると不等号の書き方が本文で混乱している箇所がありますが、
        #   幾何的に「届かないとき」が L' > L なのでここはそれに従います。
        mask_reach = (Lp <= L)

        # x,y
        x1 = torch.where(
            mask_reach,
            xt,
            x0 + (L / Lp) * dx
        )
        y1 = torch.where(
            mask_reach,
            yt,
            y0 + (L / Lp) * dy
        )

        # z
        # reachできるときのみ z を計算（届かないときは z=0）
        inside = (L * L - dx * dx - dy * dy).clamp_min(0.0)
        dz = torch.sqrt(inside + 1e-8)
        if choose_negative_z:
            z1_reach = z0 - dz  # 「小さい方」を選ぶ（ヒューリスティック）
        else:
            z1_reach = z0 + dz
        z1 = torch.where(mask_reach, z1_reach, torch.zeros_like(z0))

        x3d[:, :, child, 0] = x1
        x3d[:, :, child, 1] = y1
        x3d[:, :, child, 2] = z1

    return x3d, L0


def lift_2d_to_3d_correct(
        x2d: torch.Tensor,
        parents: torch.Tensor,
        bone_pairs: torch.Tensor,
        sym_pairs=None,
        steps: int = 200,
        lr: float = 1e-2,
        w_reproj: float = 1.0,
        w_bone: float = 10.0,
        w_vel: float = 0.5,
        w_len_reg: float = 0.1,
        optimize_xy: bool = False,
):
    """
    2D skeleton (B,T,J,2) を 3D (B,T,J,3) に持ち上げて補正する実装。

    Loss:
      - reprojection: (x,y) を入力2Dに近づける
      - bone-length consistency: 各boneの長さが時間的に一定になるように
      - velocity smoothness: 速度（時間差分）を小さく
      - length regularization: 骨長自体が極端にならないように（L2）

    parents: (J,) long, root=-1
    bone_pairs: (E,2) long  (parent, child)
    sym_pairs: list[(e1,e2)]  左右対称boneを同一長にしたいとき（任意）
    optimize_xy: False なら x,yは固定して zのみ最適化（安定＆軽い）
    """
    assert x2d.dim() == 4 and x2d.size(-1) == 2, "x2d must be (B,T,J,2)"
    device = x2d.device
    dtype = x2d.dtype
    B, T, J, _ = x2d.shape
    E = bone_pairs.shape[0]

    # --- init ---
    x3d_init, L0 = analytic_init_3d_from_2d(
        x2d=x2d,
        parents=parents.to(device),
        bone_pairs=bone_pairs.to(device),
        sym_pairs=sym_pairs,
        choose_negative_z=True,
    )

    # 最適化変数
    # joint positions
    x3d = x3d_init.clone().detach()
    x3d.requires_grad_(True)

    # 骨長（グローバル）を学習変数にしても良い（論文に近づける）
    # ただし安定のため log-param にして正に保つ
    logL = torch.log(L0.clamp_min(1e-6)).detach().clone()
    logL.requires_grad_(True)

    opt = torch.optim.Adam([x3d, logL], lr=lr)

    p = bone_pairs[:, 0].to(device)
    c = bone_pairs[:, 1].to(device)

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)

        # optionally fix x,y
        if not optimize_xy:
            # x,y を入力2Dで固定し、zだけ更新する
            x3d_xy = x2d
            z = x3d[..., 2:3]  # (B,T,J,1)
            x3d_eff = torch.cat([x3d_xy, z], dim=-1)
        else:
            x3d_eff = x3d

        # ---- reprojection loss ----
        reproj = x3d_eff[..., :2]
        loss_reproj = F.mse_loss(reproj, x2d)

        # ---- bone length consistency ----
        # bone vectors (B,T,E,3)
        v = x3d_eff[:, :, c, :] - x3d_eff[:, :, p, :]
        len_bt = torch.sqrt((v ** 2).sum(dim=-1) + 1e-8)  # (B,T,E)

        L = torch.exp(logL).clamp_min(1e-6)  # (E,)
        # 時間方向で一定：len(t,e) ≈ L(e)
        loss_bone = F.mse_loss(len_bt, L.view(1, 1, E).expand_as(len_bt))

        # 左右対称boneの同一長（任意）
        loss_sym = torch.zeros([], device=device, dtype=dtype)
        if sym_pairs is not None:
            for e1, e2 in sym_pairs:
                loss_sym = loss_sym + (L[e1] - L[e2]).pow(2)
            loss_sym = loss_sym / max(1, len(sym_pairs))

        # ---- velocity smoothness ----
        # joints velocity (B,T-1,J,3)
        vel = x3d_eff[:, 1:, :, :] - x3d_eff[:, :-1, :, :]
        loss_vel = (vel ** 2).mean()

        # ---- length regularization ----
        loss_len_reg = (L ** 2).mean()

        loss = (
                w_reproj * loss_reproj
                + w_bone * loss_bone
                + 1.0 * loss_sym
                + w_vel * loss_vel
                + w_len_reg * loss_len_reg
        )
        loss.backward()
        opt.step()

    # final
    with torch.no_grad():
        if not optimize_xy:
            out = torch.cat([x2d, x3d[..., 2:3]], dim=-1)
        else:
            out = x3d
    return out


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # List of connections between landmarks (0 to 20)
    from Parameter.Parameter import *
    from loader import *
    from SLG_datasets.SLG_datasets_with_skeleton import SLG_t2s_datasets
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    import numpy as np


    def integrate_path(id, path_list):
        integrated_path = []
        for path in path_list:
            integrated_path.append((id, path))
        return integrated_path
    connections = [(10, 0), (10, 11), (10, 12), (11, 13), (12, 14)]
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20),  # Little finger
        (5, 9), (9, 13), (13, 17),
    ]
    righteye_connections = [(33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173),
                            (173, 133),
                            (133, 155), (155, 154), (154, 153), (153, 145), (145, 144), (144, 163), (163, 7), (7, 33),
                            (33, 246)]
    lefteye_connections = [(362, 398), (398, 384), (384, 385), (385, 386), (386, 387), (387, 388), (388, 466),
                           (466, 388),
                           (388, 263), (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
                           (381, 382),
                           (382, 362)]
    contour_connectrions = [(127, 234), (234, 93), (93, 132), (132, 58), (58, 172), (172, 136), (136, 150), (150, 149),
                            (149, 176), (176, 148), (148, 152), (152, 377), (377, 400), (400, 378), (378, 379),
                            (379, 365),
                            (365, 397), (397, 288), (288, 435), (435, 361), (361, 323), (323, 454), (454, 356)]
    mouth_connections = [(0, 267), (267, 269), (269, 270), (270, 409), (409, 291), (291, 375), (375, 321), (321, 405),
                         (405, 314), (314, 17), (17, 84), (84, 181), (181, 91), (91, 146), (146, 61), (61, 185),
                         (185, 40),
                         (40, 39), (39, 37), (37, 0)]
    face_connections = righteye_connections + lefteye_connections + mouth_connections + contour_connectrions
    all_connections = connections +[(13,15)]+ [(i + 15, j + 15) for i, j in hand_connections] + [(14,36)]+ [(i + 36, j + 36) for i, j in
                                                                                               hand_connections]
    all_connections=sort_connections(all_connections)
    face_connections=sort_connections(face_connections)
    #print(all_connections)
    hand_points = [(i + 17, j + 17) for i, j in hand_connections] + [(i + 38, j + 38) for i, j in hand_connections]
    body_points = [0, 10, 11, 12, 13, 14]
    B, T, J = 64, 20, 48

    # simple chain: 0(root) -> 1 -> 2 -> 3 -> 4
    parents = build_parents_from_connections(
        all_connections,
        root=1
    )
    parents_face = build_parents_from_connections(
        face_connections,
        root=1
    )
    bone_pairs =torch.tensor(all_connections,dtype=torch.long)
    dataset="how2sign"
    print("dataset:",dataset)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")#ここは何でも良い
    train_data_path = []
    dev_data_path = []
    test_data_path = []
    train_corpus = {}
    dev_corpus = {}
    test_corpus = {}
    train_cod_root = {}
    dev_cod_root = {}
    test_cod_root = {}
    train_face_root = {}
    dev_face_root = {}
    test_face_root = {}
    train_path, dev_path, test_path, train_corpus_T, dev_corpus_T, test_corpus_T =datasets_loader_T(dataset)
    if dataset=="phoenixT":
        id=0
        train_cod_root[id]=SKELETON_TRAIN_DATADIR_T
        dev_cod_root[id]=SKELETON_DEV_DATADIR_T
        test_cod_root[id]=SKELETON_TEST_DATADIR_T
        train_face_root[id]=FACE_TRAIN_DATADIR_T
        dev_face_root[id]=FACE_DEV_DATADIR_T
        test_face_root[id]=FACE_TEST_DATADIR_T
        save_train_path=SKELETON_TRAIN_DATADIR_T_PROCESSED
        save_dev_path=SKELETON_DEV_DATADIR_T_PROCESSED
        save_test_path=SKELETON_TEST_DATADIR_T_PROCESSED
        save_train_path_face=FACE_TRAIN_DATADIR_T_PROCESSED
        save_dev_path_face=FACE_DEV_DATADIR_T_PROCESSED
        save_test_path_face=FACE_TEST_DATADIR_T_PROCESSED

    elif dataset=="CSL-Daily":
        id=1
        train_cod_root[id]=SKELETON_CSL_DAILY_DATADIR
        dev_cod_root[id]=SKELETON_CSL_DAILY_DATADIR
        test_cod_root[id]=SKELETON_CSL_DAILY_DATADIR
        train_face_root[id]=FACE_CSL_DAILY_DATADIR
        dev_face_root[id]=FACE_CSL_DAILY_DATADIR
        test_face_root[id]=FACE_CSL_DAILY_DATADIR
        save_train_path=SKELETON_CSL_DAILY_DATADIR_PROCESSED
        save_dev_path=SKELETON_CSL_DAILY_DATADIR_PROCESSED
        save_test_path=SKELETON_CSL_DAILY_DATADIR_PROCESSED
        save_train_path_face=FACE_CSL_DAILY_DATADIR_PROCESSED
        save_dev_path_face=FACE_CSL_DAILY_DATADIR_PROCESSED
        save_test_path_face=FACE_CSL_DAILY_DATADIR_PROCESSED
    elif dataset=="how2sign":
        id=2
        train_cod_root[id]=SKELETON_HOW2SIGN_TRAIN_DATADIR
        dev_cod_root[id]=SKELETON_HOW2SIGN_DEV_DATADIR
        test_cod_root[id]=SKELETON_HOW2SIGN_TEST_DATADIR
        train_face_root[id]=FACE_HOW2SIGN_TRAIN_DATADIR
        dev_face_root[id]=FACE_HOW2SIGN_DEV_DATADIR
        test_face_root[id]=FACE_HOW2SIGN_TEST_DATADIR
        save_train_path=SKELETON_HOW2SIGN_TRAIN_DATADIR_PROCESSED
        save_dev_path=SKELETON_HOW2SIGN_DEV_DATADIR_PROCESSED
        save_test_path=SKELETON_HOW2SIGN_TEST_DATADIR_PROCESSED
        save_train_path_face=FACE_HOW2SIGN_TRAIN_DATADIR_PROCESSED
        save_dev_path_face=FACE_HOW2SIGN_DEV_DATADIR_PROCESSED
        save_test_path_face=FACE_HOW2SIGN_TEST_DATADIR_PROCESSED
    elif dataset=="phoenix":
        id = 3
        train_cod_root[id] = SKELETON_TRAIN_DATADIR
        dev_cod_root[id] = SKELETON_DEV_DATADIR
        test_cod_root[id] = SKELETON_TEST_DATADIR
        train_face_root[id] = FACE_TRAIN_DATADIR
        dev_face_root[id] = FACE_DEV_DATADIR
        test_face_root[id] = FACE_TEST_DATADIR
        save_train_path = SKELETON_TRAIN_DATADIR_PROCESSED
        save_dev_path = SKELETON_DEV_DATADIR_PROCESSED
        save_test_path = SKELETON_TEST_DATADIR_PROCESSED
        save_train_path_face = FACE_TRAIN_DATADIR_PROCESSED
        save_dev_path_face = FACE_DEV_DATADIR_PROCESSED
        save_test_path_face = FACE_TEST_DATADIR_PROCESSED
    else:
        raise ValueError("Invalid dataset name")
    train_corpus[id] = train_corpus_T
    dev_corpus[id] = dev_corpus_T
    test_corpus[id] = test_corpus_T
    train_data_path=integrate_path(id,train_path)
    dev_data_path=integrate_path(id,dev_path)
    test_data_path=integrate_path(id,test_path)
    ds_train=SLG_t2s_datasets(train_data_path, train_cod_root, train_face_root, train_corpus, tokenizer, trainable=False, is_processed=False,
                              is_sg_filter=False)
    ds_dev=SLG_t2s_datasets(dev_data_path, dev_cod_root, dev_face_root, dev_corpus, tokenizer, trainable=False, is_processed=False,
                            is_sg_filter=False)
    ds_test=SLG_t2s_datasets(test_data_path, test_cod_root, test_face_root, test_corpus, tokenizer, trainable=False, is_processed=False,
                             is_sg_filter=False)
    dl_train=DataLoader(ds_train,batch_size=B,shuffle=False,collate_fn=ds_train.collate_fn,drop_last=False)
    dl_dev=DataLoader(ds_dev,batch_size=B,shuffle=False,collate_fn=ds_dev.collate_fn,drop_last=False)
    dl_test=DataLoader(ds_test,batch_size=B,shuffle=False,collate_fn=ds_test.collate_fn,drop_last=False)

    if os.path.exists(save_train_path):
        shutil.rmtree(save_train_path)
    if os.path.exists(save_dev_path):
        shutil.rmtree(save_dev_path)
    if os.path.exists(save_test_path):
        shutil.rmtree(save_test_path)

    if os.path.exists(save_train_path_face):
        shutil.rmtree(save_train_path_face)
    if os.path.exists(save_dev_path_face):
        shutil.rmtree(save_dev_path_face)
    if os.path.exists(save_test_path_face):
        shutil.rmtree(save_test_path_face)

    os.makedirs(save_train_path,exist_ok=True)
    os.makedirs(save_dev_path,exist_ok=True)
    os.makedirs(save_test_path,exist_ok=True)

    os.makedirs(save_train_path_face,exist_ok=True)
    os.makedirs(save_dev_path_face,exist_ok=True)
    os.makedirs(save_test_path_face,exist_ok=True)
    print("Start processing...")
    print("--- Processing train set ---")
    for batch in dl_train:
        cod_data, input_length, sentence, _, path = batch
        x2d=cod_data[0].permute(0,2,3,1)
        x2d_face=cod_data[1].permute(0,2,3,1)
        B,T,J,C=x2d.shape

        x3d = lift_2d_to_3d_correct(
            x2d,
            parents=parents,
            bone_pairs=torch.tensor(all_connections,dtype=torch.long),
            sym_pairs=None,
            steps=100,
            lr=1e-2,
            optimize_xy=True,  # まずはzだけ
        ).reshape(B,T,-1).detach().cpu().numpy()

        x3d_face=x2d_face.reshape(B,T,-1)
        for i in range(B):
            print(f"Processing {path[i]}...")
            if id!=3:
                save_file_path=f"{save_train_path}/{path[i].split('/')[-1].split('.csv')[0]}.csv"
                save_face_file_path=f"{save_train_path_face}/{path[i].split('/')[-1].split('.csv')[0]}.csv"
            else:
                save_file_path=f"{save_train_path}/{path[i].split('/')[-2].split('.csv')[0]}.csv"
                save_face_file_path=f"{save_train_path_face}/{path[i].split('/')[-2].split('.csv')[0]}.csv"
            np.savetxt(save_file_path,x3d[i][:input_length[i]],delimiter=",")
            np.savetxt(save_face_file_path,x3d_face[i][:input_length[i]].numpy(),delimiter=",")
    if dataset=="CSL-Daily":
        exit()
    print("--- Processing dev set ---")
    for batch in dl_dev:
        cod_data, input_length, sentence, _, path = batch
        x2d=cod_data[0].permute(0,2,3,1)
        x2d_face=cod_data[1].permute(0,2,3,1)
        B,T,J,C=x2d.shape

        x3d = lift_2d_to_3d_correct(
            x2d,
            parents=parents,
            bone_pairs=torch.tensor(all_connections,dtype=torch.long),
            sym_pairs=None,
            steps=100,
            lr=1e-2,
            optimize_xy=True,  # まずはzだけ
        ).reshape(B,T,-1).detach().cpu().numpy()

        x3d_face=x2d_face.reshape(B,T,-1)
        for i in range(B):
            print(f"Processing {path[i]}...")
            if id!=3:
                save_file_path=f"{save_dev_path}/{path[i].split('/')[-1].split('.csv')[0]}.csv"
                save_face_file_path=f"{save_dev_path_face}/{path[i].split('/')[-1].split('.csv')[0]}.csv"
            else:
                save_file_path=f"{save_dev_path}/{path[i].split('/')[-2].split('.csv')[0]}.csv"
                save_face_file_path=f"{save_dev_path_face}/{path[i].split('/')[-2].split('.csv')[0]}.csv"
            np.savetxt(save_file_path,x3d[i][:input_length[i]],delimiter=",")
            np.savetxt(save_face_file_path,x3d_face[i][:input_length[i]].numpy(),delimiter=",")

    print("--- Processing test set ---")
    for batch in dl_test:
        cod_data, input_length, sentence, _, path = batch
        x2d=cod_data[0].permute(0,2,3,1)
        x2d_face=cod_data[1].permute(0,2,3,1)
        B,T,J,C=x2d.shape

        x3d = lift_2d_to_3d_correct(
            x2d,
            parents=parents,
            bone_pairs=torch.tensor(all_connections,dtype=torch.long),
            sym_pairs=None,
            steps=100,
            lr=1e-2,
            optimize_xy=True,  # まずはzだけ
        ).reshape(B,T,-1).detach().cpu().numpy()
        x3d_face=x2d_face.reshape(B,T,-1)
        for i in range(B):
            print(f"Processing {path[i]}...")
            if id!=3:
                save_file_path=f"{save_test_path}/{path[i].split('/')[-1].split('.csv')[0]}.csv"
                save_face_file_path=f"{save_test_path_face}/{path[i].split('/')[-1].split('.csv')[0]}.csv"
            else:
                save_file_path=f"{save_test_path}/{path[i].split('/')[-2].split('.csv')[0]}.csv"
                save_face_file_path=f"{save_test_path_face}/{path[i].split('/')[-2].split('.csv')[0]}.csv"
            np.savetxt(save_file_path,x3d[i][:input_length[i]],delimiter=",")
            np.savetxt(save_face_file_path,x3d_face[i][:input_length[i]].numpy(),delimiter=",")
