import os,glob

import torch.cuda
from pandas.core.algorithms import value_counts_arraylike

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from Parameter.Parameter import *
from loader.data_loader import *
from loader.coordinate_preprocess import *
from utils.missing_joints import *
from SLG_datasets.SLG_datasets_Units import SLGText2UnitsDatasets
from models.module.Hand_gcn_vae_6d import *
from models.ema import EMA
from loader.skeleton_video import save_skeleton_video
from models.module.STGCNHand import make_mask
from ProgressiveTransformers.model import Model
from ProgressiveTransformers.helpers import *
from ProgressiveTransformers.embeddings import Embeddings
from ProgressiveTransformers.encoders import Encoder,TransformerEncoder
from ProgressiveTransformers.decoders import Decoder,TransformerDecoder
from ProgressiveTransformers.vocabulary import Vocabulary
from ProgressiveTransformers.loss import RegLoss
from ProgressiveTransformers.search import greedy
from transformers import AutoTokenizer,AutoModel
import shutil
import json,yaml
import wandb
import torch.nn.functional as F
try:
    from numba import njit
except ImportError:  # numba が無ければ素の Python（遅いが同じ結果）
    njit = None
torch.autograd.set_detect_anomaly(False)
# seedを固定
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#----------------------------------------------------------------------------- masks


def make_trg_mask(valid_len: torch.Tensor, T: int) -> torch.Tensor:
    """(B,) 有効長 → (B, 1, T, T) の query 行マスク。元実装の trg_mask と同じ意味。

    mask[b, 0, q, k] = [q < valid_len_b]。key 側は因果マスク＋右詰めで暗黙に遮断される。
    """
    valid = lengths_to_mask(valid_len.clamp(max=T), max_len=T, dtype=torch.bool)  # (B, T)
    return valid[:, None, :, None].expand(-1, 1, -1, T)


# ----------------------------------------------------------------------------- DTW
def _dtw_acc(C: np.ndarray) -> float:
    """D(i,j) = C(i,j) + min(D(i-1,j-1), D(i-1,j), D(i,j-1)),  D(0,0)=0, 境界は inf。"""
    N, M = C.shape
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            a, b, c = D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
            m = a if a < b else b
            m = m if m < c else c
            D[i, j] = C[i - 1, j - 1] + m
    return D[N, M]


if njit is not None:
    _dtw_acc = njit(cache=True)(_dtw_acc)


def dtw_l1(ref: torch.Tensor, hyp: torch.Tensor) -> float:
    """ref (N_r, J), hyp (N_h, J) → DTW(ref, hyp) / N_r。コストは L1（元実装と同じ）。"""
    C = torch.cdist(ref[None].float(), hyp[None].float(), p=1)[0]  # (N_r, N_h), GPU 可
    return float(_dtw_acc(C.cpu().numpy().astype(np.float64))) / ref.shape[0]


def _cut_index(x: torch.Tensor, trim: str, lengths: torch.Tensor | None) -> torch.Tensor:
    """各系列の切り詰め位置 (B,)。"""
    if trim == "counter":  # 元実装: counter の argmax（ピーク自体は含まない）
        return x[:, :, -1].float().argmax(dim=1).clamp_min(1)
    if trim == "length":  # 実長で切る
        return lengths.clamp(min=1, max=x.shape[1])
    raise ValueError("trim must be 'counter' or 'length'")


def calculate_dtw(references: torch.Tensor, hypotheses: torch.Tensor,
                  trim: str = "length", ref_lengths: torch.Tensor | None = None) -> list[float]:
    """references (B, T′, D) = trg, hypotheses (B, T′+1, D) = greedy 出力（先頭は BOS）。"""
    hypotheses = hypotheses[:, 1:]  # BOS（GT 第0フレーム）を除去
    r_idx = _cut_index(references, trim, ref_lengths)
    h_idx = _cut_index(hypotheses, trim, ref_lengths)
    return [dtw_l1(references[b, :r_idx[b], :-1], hypotheses[b, :h_idx[b], :-1])  # counter 列は除く
            for b in range(references.shape[0])]


# ----------------------------------------------------------------------------- greedy
@torch.no_grad()
def greedy_decode(model, src, src_length, src_mask, trg_input, trg_valid_len):
    """search.greedy と同じ手順。GT の y_0 から開始し、各ステップで入力 counter を GT で上書き。

    戻り値: (B, T′+1, D_out)  先頭は y_0（calculate_dtw 内で除去）。
    """
    m = model.module if hasattr(model, "module") else model  # DDP / DataParallel 対応
    encoder_output, _ = m.encode(src, src_length, src_mask)
    T = trg_input.shape[1]
    ys = trg_input[:, :1, :].float().clone()
    ys_out = ys.clone()
    fp = getattr(m, "future_prediction", 0)

    for i in range(T):
        ys[:, -1, -1:] = trg_input[:, i, -1:]  # GT counter で時刻を駆動
        padding_mask = make_trg_mask(trg_valid_len, T)[:, :, :i + 1, :i + 1]
        out, _, _, _ = m.decoder(trg_embed=m.trg_embed(ys), encoder_output=encoder_output,
                                 src_mask=src_mask, trg_mask=padding_mask)
        out = out[:, -1:, :].float()
        if fp != 0:  # 1 フレーム目の予測 + counter のみ
            out = torch.cat((out[:, :, :out.shape[2] // fp], out[:, :, -1:]), dim=2)
        ys = torch.cat([ys, out], dim=1)
        ys_out = torch.cat([ys_out, out], dim=1)
    return ys_out
def append_counter(src,lengths):
    # src: (B, T, D)
    B, T, D = src.shape
    counter=torch.zeros(B,T,1,device=src.device)
    for i in range(B):
        counter[i,:lengths[i],0]=torch.arange(lengths[i].item(),device=src.device)/lengths[i].item()
    return torch.cat([src,counter],dim=-1)






def create_word_tokenizer(corpus,key='annotation',is_lower=False):
    # corpusの中のすべての単語を取得
    all_words = []
    text=corpus[key].values
    for sentence in text:
        if is_lower:
            words = sentence.lower().split()
        else:
            words = sentence.split()
        if "." in words:
            words.remove(".")
        all_words.extend(words)
    # 単語の出現回数をカウント
    word_count = {}
    for word in all_words:
        if word not in word_count:
            word_count[word] = 0
        word_count[word] += 1
    # 出現回数が多い順にソート
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    # 単語とインデックスの対応表を作成
    word2index = {word: index+4 for index, (word, count) in enumerate(sorted_word_count)}
    index2word = {index+4: word for index, (word, count) in enumerate(sorted_word_count)}
    word2index['<pad>'] = 0
    word2index['<s>'] = 1
    word2index['</s>'] = 2
    word2index['<unk']=3
    index2word[0] = '<pad>'
    index2word[1] = '<s>'
    index2word[2] = '</s>'
    index2word[3] = '<unk>'

    return word2index, index2word
@torch.no_grad()
def sample_with_diagnostics(model, n, T, steps=50, cond=None, cond_mask=None,
                            guidance_scale=1.0, lengths=None, device=None):
    """sampleと同じ軌道を辿りつつ、各stepの統計を記録して返す."""
    device = device or next(model.parameters()).device
    use_cfg = (cond is not None and guidance_scale != 1.0)
    if lengths is not None:
        lengths = torch.as_tensor(lengths, device=device)
        T = int(lengths.max().item())
    T_lat = max(1, T // model.vae.t_stride)
    valid = model._length_mask(lengths, T_lat, device)
    time_pad = ~valid if valid is not None else None

    z = torch.randn(n, T_lat, model.num_tokens, model.latent_dim, device=device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    log = {"t": [], "z_norm": [], "v_norm": [], "dz_norm": [], "is_high": []}
    for i in range(steps):
        t_now = ts[i].expand(n)
        if use_cfg:
            v_c = model.velocity(z, t_now, context=cond, context_mask=cond_mask,
                                 time_pad=time_pad)
            v_u = model.velocity(z, t_now, context=None, time_pad=time_pad)
            v = v_u + guidance_scale * (v_c - v_u)
        else:
            v = model.velocity(z, t_now, context=cond, context_mask=cond_mask,
                               time_pad=time_pad)
        dz = (ts[i] - ts[i + 1]) * v
        # 有効フレームのみで統計を取る (パディングはゼロで薄まるため)
        m = valid[:, :, None, None].float() if valid is not None else \
            torch.ones_like(z[..., :1])
        nel = m.sum() * z.shape[2] * z.shape[3] / m.numel() * m.numel()
        def rms(x):  # マスク付きRMSノルム
            return ((x * x * m).sum() / (m.sum() * x.shape[2] * x.shape[3])).sqrt().item()
        log["t"].append(ts[i].item())
        log["z_norm"].append(rms(z))
        log["v_norm"].append(rms(v))
        log["dz_norm"].append(rms(dz))
        log["is_high"].append(bool(model.router(t_now)[0]))
        z = z - dz
    return z, log
def normalize_batch(padded_cod_data, used_3d):
    all_inputs   = padded_cod_data.permute(0, 1, 3, 2).contiguous()
    body_inputs  = padded_cod_data[:, :, :-42].permute(0, 1, 3, 2).clone()
    left_inputs  = padded_cod_data[:, :, -42:-21].permute(0, 1, 3, 2).clone()
    left_inputs[...,0]=body_inputs[...,15].clone()
    right_inputs = padded_cod_data[:, :, -21:].permute(0, 1, 3, 2).clone()
    right_inputs[...,0]=body_inputs[...,16].clone()
    norm_info = None
    if used_3d:
        center = all_inputs[:, :, :, 10].clone()
        s = torch.sqrt(
            (all_inputs[:,:,0,11]-all_inputs[:,:,0,12])**2 +
            (all_inputs[:,:,1,11]-all_inputs[:,:,1,12])**2 +
            (all_inputs[:,:,2,11]-all_inputs[:,:,2,12])**2)
        B, T = s.shape
        s = s.reshape(B, T, 1, 1)
        body_inputs = (body_inputs - center.unsqueeze(3)) / (s + 1e-8)
        all_inputs  = (all_inputs  - center.unsqueeze(3)) / (s + 1e-8)
        lc = left_inputs[:, :, :, 0].clone()
        rc = right_inputs[:, :, :, 0].clone()
        left_inputs  = (left_inputs  - lc.unsqueeze(3)) / (s/2 + 1e-8)
        right_inputs = (right_inputs - rc.unsqueeze(3)) / (s/2 + 1e-8)
        norm_info = dict(center=center, shoulder=s, left_center=lc, right_center=rc)
    return all_inputs, body_inputs, left_inputs, right_inputs, norm_info
def integrate_path(id, path_list):
    integrated_path = []
    for path in path_list:
        integrated_path.append((id, path))
    return integrated_path
def compute_bone_lengths(x, bones, bone_mask=None, eps=1e-6):
    """
    x:         (B, T, 3, J)
    bone_mask: (B, T, Nb) bool  有効ボーン (make_bone_maskの出力)
               Noneなら「長さ > eps」で自動判定
    return:    (B, Nb)  有効フレームのみの中央値ボーン長
    """
    pos = x.movedim(-2, -1)                                   # (B, T, J, 3)
    idx = torch.as_tensor([b[:2] for b in bones], device=x.device)
    vec = pos[..., idx[:, 1], :] - pos[..., idx[:, 0], :]
    lens = vec.norm(dim=-1)                                   # (B, T, Nb)

    valid = bone_mask if bone_mask is not None else lens > eps
    lens = lens.masked_fill(~valid, float("nan"))
    L = lens.nanmedian(dim=1).values                          # (B, Nb)

    # 全フレーム無効のボーンへのフォールバック(バッチ内中央値で補完)
    all_nan = torch.isnan(L)
    if all_nan.any():
        batch_med = L.nanmedian(dim=0).values                 # (Nb,)
        L = torch.where(all_nan, batch_med.expand_as(L), L)
        if torch.isnan(L).any():
            raise ValueError("some bones have no valid frame in entire batch")
    return L
def visualize_skeletons(pred_cods,gt_cods,data_path, lengths, save_dir, bones, mask=None,):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    #print(gt_cods.shape)
    os.makedirs(save_dir, exist_ok=True)
    if gt_cods.shape[-1] != pred_cods.shape[-1]:
        gt_cods=gt_cods[:,:,:,bones_used_joints(bones)]
    #pred_coordinates=pred_coordinates[:,:,:,bones_used_joints(bones)]
    #gt_cods=gt_cods[:,:,:,bones_used_joints(bones)]
    for i in range(pred_cods.shape[0]):
        pred_video_path = f"{save_dir}/{os.path.basename(data_path[i])}.mp4"
        save_skeleton_video(pred_cods[i][:lengths[i]].cpu().numpy(), pred_video_path,
                            bones=bones,
                            x_ref=gt_cods[i][:lengths[i]].cpu().numpy())


def main(config, dataset,save_path,visualize=False,used_3d=False,weights=None):
    print("---Loading datasets---")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # trainデータのパスを結合(id,pathのタプルorリスト)
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
    i = 0
    is_islr=False
    is_gloss=True
    if dataset=="phoenixT":
        if is_islr==True:
            phoenixT_train_path, phoenixT_dev_path, phoenixT_test_path, phoenixT_gloss2class, phoenixT_class2gloss, phoenixT_video2gloss = islr_datasets_loader(
                "phoenixT")
            train_corpus[3] = phoenixT_video2gloss
            dev_corpus[3] = phoenixT_video2gloss
            test_corpus[3] = phoenixT_video2gloss
            #gloss2class=np.load(f"./Parameter/gloss_dict_T.npy",allow_pickle=True).item()
            train_cod_root[3] = f"{WORDS_DATADIR_T_SKELETON}/train"
            dev_cod_root[3] = f"{WORDS_DATADIR_T_SKELETON}/dev"
            test_cod_root[3] = f"{WORDS_DATADIR_T_SKELETON}/test"

            train_face_root[3] = f"{WORDS_DATADIR_T_SKELETON_FACE}/train"
            dev_face_root[3] = f"{WORDS_DATADIR_T_SKELETON_FACE}/dev"
            test_face_root[3] = f"{WORDS_DATADIR_T_SKELETON_FACE}/test"
            is_3d = True

            train_data_path += integrate_path(3, phoenixT_train_path)
            dev_data_path += integrate_path(3, phoenixT_dev_path)
            test_data_path += integrate_path(3, phoenixT_test_path)
            gloss2class=phoenixT_gloss2class
            class2gloss=phoenixT_class2gloss
            train_data_path += integrate_path(3, phoenixT_train_path)
            dev_data_path += integrate_path(3, phoenixT_dev_path)
            test_data_path += integrate_path(3, phoenixT_test_path)
            i += 1
        else:
            phoenixT_train_path, phoenixT_dev_path, phoenixT_test_path, phoenixTgn_train_corpus, phoenixT_dev_corpus, phoenixT_test_corpus= datasets_loader_T(
                "phoenixT",gloss=is_gloss)
            gloss2class = np.load(f"./Parameter/gloss_dict_T.npy", allow_pickle=True).item()

            train_corpus[1] = phoenixTgn_train_corpus
            dev_corpus[1] = phoenixT_dev_corpus
            test_corpus[1] = phoenixT_test_corpus
            train_cod_root[1] = SKELETON_TRAIN_DATADIR_T_3D
            dev_cod_root[1] = SKELETON_DEV_DATADIR_T_3D
            test_cod_root[1] = SKELETON_TEST_DATADIR_T_3D

            train_face_root[1] = FACE_TRAIN_DATADIR_T_3D
            dev_face_root[1] = FACE_DEV_DATADIR_T_3D
            test_face_root[1] = FACE_TEST_DATADIR_T_3D
            train_data_path += integrate_path(1, phoenixT_train_path)
            dev_data_path += integrate_path(1, phoenixT_dev_path)
            test_data_path += integrate_path(1, phoenixT_test_path)
            is_3d = True
    if dataset=="CSL-Daily":
        csl_daily_train_path, csl_daily_dev_path, csl_daily_test_path, csl_daily_train_corpus, csl_daily_dev_corpus, csl_daily_test_corpus = datasets_loader_T(
            "CSL-Daily")

        train_corpus[1] = csl_daily_train_corpus
        dev_corpus[1] = csl_daily_dev_corpus
        test_corpus[1] = csl_daily_test_corpus
        if config['is_processed']:
            train_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_PROCESSED
            dev_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_PROCESSED
            test_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_PROCESSED

            train_face_root[1] = FACE_CSL_DAILY_DATADIR_PROCESSED
            dev_face_root[1] = FACE_CSL_DAILY_DATADIR_PROCESSED
            test_face_root[1] = FACE_CSL_DAILY_DATADIR_PROCESSED
            is_3d = True
        else:
            train_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_3D
            dev_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_3D
            test_cod_root[1] = SKELETON_CSL_DAILY_DATADIR_3D

            train_face_root[1] = FACE_CSL_DAILY_DATADIR_3D
            dev_face_root[1] = FACE_CSL_DAILY_DATADIR_3D
            test_face_root[1] = FACE_CSL_DAILY_DATADIR_3D
            is_3d = True
        train_data_path += integrate_path(1, csl_daily_train_path)
        dev_data_path += integrate_path(1, csl_daily_dev_path)
        test_data_path += integrate_path(1, csl_daily_test_path)
        i += 1
    if dataset=="how2sign":
        how2sign_train_path, how2sign_dev_path, how2sign_test_path, how2sign_train_corpus, how2sign_dev_corpus, how2sign_test_corpus = datasets_loader_T(
            "how2sign")
        train_corpus[2] = how2sign_train_corpus
        dev_corpus[2] = how2sign_dev_corpus
        test_corpus[2] = how2sign_test_corpus
        if config['is_processed']:
            train_cod_root[2] = SKELETON_HOW2SIGN_TRAIN_DATADIR_PROCESSED
            dev_cod_root[2] = SKELETON_HOW2SIGN_DEV_DATADIR_PROCESSED
            test_cod_root[2] = SKELETON_HOW2SIGN_TEST_DATADIR_PROCESSED

            train_face_root[2] = FACE_HOW2SIGN_TRAIN_DATADIR_PROCESSED
            dev_face_root[2] = FACE_HOW2SIGN_DEV_DATADIR_PROCESSED
            test_face_root[2] = FACE_HOW2SIGN_TEST_DATADIR_PROCESSED
            is_3d = True
        else:
            train_cod_root[2] = SKELETON_HOW2SIGN_TRAIN_DATADIR_3D
            dev_cod_root[2] = SKELETON_HOW2SIGN_DEV_DATADIR_3D
            test_cod_root[2] = SKELETON_HOW2SIGN_TEST_DATADIR_3D

            train_face_root[2] = FACE_HOW2SIGN_TRAIN_DATADIR_3D
            dev_face_root[2] = FACE_HOW2SIGN_DEV_DATADIR_3D
            test_face_root[2] = FACE_HOW2SIGN_TEST_DATADIR_3D
            is_3d = True

        train_data_path += integrate_path(2, how2sign_train_path)
        dev_data_path += integrate_path(2, how2sign_dev_path)
        test_data_path += integrate_path(2, how2sign_test_path)
        i += 1
    if dataset=="phoenix":
        phoenix_train_path, phoenix_dev_path, phoenix_test_path, phoenix_gloss2class, phoenix_class2gloss, phoenix_video2gloss =islr_datasets_loader(
            "phoenix")
        train_corpus[3] = phoenix_video2gloss
        dev_corpus[3] = phoenix_video2gloss
        test_corpus[3] = phoenix_video2gloss
        num_class=len(phoenix_gloss2class)
        train_cod_root[3] = f"{WORDS_DATADIR_SKELETON}/train"
        dev_cod_root[3] = f"{WORDS_DATADIR_SKELETON}/dev"
        test_cod_root[3] = f"{WORDS_DATADIR_SKELETON}/test"

        train_face_root[3] = f"{WORDS_DATADIR_SKELETON_FACE}/train"
        dev_face_root[3] = f"{WORDS_DATADIR_SKELETON_FACE}/dev"
        test_face_root[3] = f"{WORDS_DATADIR_SKELETON_FACE}/test"
        is_3d = True

        train_data_path += integrate_path(3, phoenix_train_path)
        dev_data_path += integrate_path(3, phoenix_dev_path)
        test_data_path += integrate_path(3, phoenix_test_path)
        is_islr=True
        i += 1
    if dataset=="AUTSL":
        autsl_train_path, autsl_dev_path, autsl_test_path,autsl_gloss2class, autsl_class2gloss,autsl_video2gloss  = islr_datasets_loader(
            "AUTSL")
        train_corpus[5] = autsl_video2gloss
        dev_corpus[5] =autsl_video2gloss
        test_corpus[5] = autsl_video2gloss
        num_class = len(autsl_gloss2class)
        if config['is_processed']:
           raise NotImplementedError("Processed data for AUTSL is not available yet.")
        else:
            train_cod_root[5] = SKELETON_AUTSL_TRAIN_DATADIR_3D
            dev_cod_root[5] = SKELETON_AUTSL_DEV_DATADIR_3D
            test_cod_root[5] = SKELETON_AUTSL_TEST_DATADIR_3D

            train_face_root[5] = FACE_AUTSL_TRAIN_DATADIR_3D
            dev_face_root[5] = FACE_AUTSL_DEV_DATADIR_3D
            test_face_root[5] = FACE_AUTSL_TEST_DATADIR_3D
            is_3d=True
        train_data_path += integrate_path(5, autsl_train_path)
        dev_data_path += integrate_path(5, autsl_dev_path)
        test_data_path += integrate_path(5, autsl_test_path)
        is_islr=True
        i += 1
    print("Datasets loaded.")
    print("保存場所:", save_path)
    print("Is GPU available?:", torch.cuda.is_available())
    flow_config=config['flow_matching']

    device = config["device"] if torch.cuda.is_available() else "cpu"
    # deviceからgpuの名前を取得して表示
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(int(device[-1]))
        print("GPU name:", gpu_name)
    print("---Loading tokenizer---")
    tokenizer, decoder = create_word_tokenizer(pd.concat([train_corpus[1], dev_corpus[1], test_corpus[1]], axis=0),
                                               key="annotation")
    print("num tokens:", len(tokenizer))
    print("---Creating datasets---")
    ds_train = SLGText2UnitsDatasets(train_data_path, train_cod_root, train_face_root, is_3d=True,
                                     is_processed=False, is_sg_filter=False,
                                     is_coarse=False,
                                     trainable=True, tokenizer=tokenizer,texts_corpus=train_corpus,is_islr=is_islr,
                                     scale_ratio=(1.0,1.0),gloss2class=None,is_delete_cod=False,is_norm=False)
    ds_dev = SLGText2UnitsDatasets(dev_data_path, dev_cod_root, dev_face_root, trainable=False, is_3d=True,
                                   is_processed=False, is_sg_filter=False,
                                   is_coarse=False,tokenizer=tokenizer,texts_corpus=dev_corpus,is_islr=is_islr,
                                   scale_ratio=(1.0,1.0),gloss2class=None,is_delete_cod=False,is_norm=False)
    ds_test = SLGText2UnitsDatasets(test_data_path, test_cod_root, test_face_root, trainable=False, is_3d=True,
                                    is_processed=False, is_sg_filter=False,
                                    is_coarse=False,tokenizer=tokenizer,texts_corpus=test_corpus,is_islr=is_islr,
                                    scale_ratio=(1.0,1.0),gloss2class=None,is_delete_cod=False,is_norm=False)
    dl_train=torch.utils.data.DataLoader(ds_train, batch_size=flow_config['lr_parameters']['batch_size'], shuffle=True, num_workers=4, collate_fn=ds_train.collate_fn,drop_last=True)
    dl_dev=torch.utils.data.DataLoader(ds_dev, batch_size=flow_config['lr_parameters']['batch_size'], shuffle=True, num_workers=4, collate_fn=ds_dev.collate_fn)
    dl_test=torch.utils.data.DataLoader(ds_test, batch_size=flow_config['lr_parameters']['batch_size'], shuffle=False, num_workers=4, collate_fn=ds_test.collate_fn)
    print("Datasets created.")
    bones=ALL_BONES
    hand_bones=HAND_BONES
    body_bnones=BODY_BONES
    # Input target size is the joint vector length plus one for counter
    in_trg_size = 144 + 1
    # Output target size is the joint vector length plus one for counter
    out_trg_size = 144 + 1
    trg_linear=nn.Linear(145,512)
    src_embed=Embeddings(embedding_dim=512,dropout=0.0,vocab_size=len(tokenizer))
    vocab_list=tokenizer.keys()
    encoder=TransformerEncoder(
        num_layers=2,
        num_heads=4,
        hidden_size=512,
        ff_size=2048,
        dropout=0.0,
        emb_size=src_embed.embedding_dim,
        emb_dropout=0.0,
    )
    decoder=TransformerDecoder(
        num_layers=2,
        num_heads=4,
        hidden_size=512,
        ff_size=2048,
        dropout=0.0,
        emb_size=trg_linear.out_features,
        emb_dropout=0.0,
        trg_size=145,decoder_trg_trg_=True
    )
    trg_vocab = [None] * 145
    src_vocab=Vocabulary(tokens=list(vocab_list))
    cfg=load_config("/home/caffe/work/SLG/ProgressiveTransformers/Configs/Base.yaml")
    model=Model(encoder=encoder,
                decoder=decoder,
                src_embed=src_embed,
                trg_embed=trg_linear,
                src_vocab=src_vocab,
                trg_vocab=trg_vocab,
                cfg=cfg,
                in_trg_size=in_trg_size,
                out_trg_size=out_trg_size,
                )
    with open(f"{save_path}/flow_match.yaml", "w") as f:
        yaml.dump(config, f)

    model=model.to(device)
    loss_function=RegLoss(cfg)

    if visualize!=True:


        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3, betas=(0.9, 0.95))
        epochs = flow_config['lr_parameters']['epoch']
        warmup, total = epochs * len(dl_train) * 0.1 // (
                    flow_config['lr_parameters']['batch_size'] * flow_config['lr_parameters'][
                'accumulation_steps']), epochs * len(dl_train) // (
                                    flow_config['lr_parameters']['batch_size'] * flow_config['lr_parameters'][
                                'accumulation_steps'])

        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            prog = (step - warmup) / max(1, total - warmup)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=7,threshold=1e-8,threshold_mode="abs")
        # ema_model=torch.optim.swa_utils.AveragedModel(model,
        #                                              torch.optim.swa_utils.get_ema_multi_avg_fn(0.9), use_buffers=True)
        step=0
        ema=EMA(model, decay=0.9999, warmup_steps=10)
        wandb.init(project="SLG_flow", entity="tkeda-jean-tokyo-city-university")
        accumulation_steps=flow_config['lr_parameters']['accumulation_steps']
        num_batches=len(dl_train)
        scaler=torch.amp.GradScaler('cuda')
        min_loss=1e5
        stop=False
        if weights!=None:
            checkpoint=torch.load(weights,map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            step=checkpoint['step']
            init_epoch=checkpoint['epoch']
            ema.load_state_dict(checkpoint['ema'])
            # cross-attn の gate (g_x) が開いているか

        else:
            init_epoch=0
        used_jts = bones_used_joints(bones=bones, include_aux=True)
        o2c = {o: i for i, o in enumerate(used_jts)}
        mapped_bones = [(o2c[p], o2c[c], o2c[a]) for p, c, a in bones]
        for epoch in range(init_epoch,epochs):
            torch.cuda.empty_cache()
            model.train()
            total_loss=0
            dev_total_loss=0
            test_total_loss=0
            for batch_idx,batch in enumerate(dl_train):
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens = batch
                # --- src（従来通り）
                text_tokens = text_tokens.to(device)
                src = text_tokens["input_ids"][:, 1:]
                src_mask = text_tokens["attention_mask"][:, 1:]
                src_length = mask_to_lengths(src_mask)
                src_mask = src_mask.unsqueeze(1).bool()

                trg_input, trg, loss_mask, trg_mask, norm_info = prepare_trg(
                    padded_cod_data, input_length_tensor, used_jts, device)
                trg_len=(input_length_tensor - 1).clamp(min=1)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16,enabled=False):
                         # (B, 1, T′, T′)
                    skel_out, _ = model(src=src, trg_input=trg_input, src_mask=src_mask, src_lengths=src_length,
                                            trg_mask=trg_mask)

                    raw_loss = masked_regression_loss(skel_out, trg, loss_mask)

                # 最後の区間がaccumulation_steps未満の場合にも対応
                window_start = (
                                       batch_idx // accumulation_steps
                               ) * accumulation_steps

                current_accumulation_steps = min(
                    accumulation_steps,
                    num_batches - window_start,
                )

                loss =raw_loss / current_accumulation_steps

                scaler.scale(loss).backward()
                is_update_step = (batch_idx + 1) % accumulation_steps == 0
                if is_update_step:
                    # Gradient Clipping前にscaleを解除
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=flow_config['lr_parameters']['grad_clip_norm'],
                    )

                    # optimizer.step()の代わり
                    scaler.step(optimizer)

                    # 次回用にscale値を更新
                    scaler.update()
                    ema.update(model)
                    # 蓄積した勾配をリセット
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                else:
                    if (batch_idx + 1) == num_batches:
                        # 最後のステップで更新が残っている場合は更新せず，勾配をリセットする
                        optimizer.zero_grad(set_to_none=True)
                total_loss+=raw_loss.item()*len(padded_cod_data)
            #scheduler.step()
            avg_loss=total_loss/len(ds_train)
            all_dtw_dev=[]
            for batch in dl_dev:
                model.eval()
                with torch.no_grad():
                    padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens = batch
                    text_tokens = text_tokens.to(device)
                    src = text_tokens["input_ids"][:, 1:]
                    src_mask = text_tokens["attention_mask"][:, 1:]
                    src_length = mask_to_lengths(src_mask)
                    src_mask = src_mask.unsqueeze(1).bool()
                    trg_input, trg, loss_mask, trg_mask, norm_info = prepare_trg(
                        padded_cod_data, input_length_tensor, used_jts, device)
                    trg_len = (input_length_tensor - 1).clamp(min=1)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16,enabled=False):
                        # (B, 1, T′, T′)
                        skel_out, _ = model(src=src, trg_input=trg_input, src_mask=src_mask, src_lengths=src_length,
                                            trg_mask=trg_mask)

                        loss = masked_regression_loss(skel_out, trg, loss_mask)
                    dev_total_loss+=loss.item()*len(padded_cod_data)
                    trg_in_len=input_length_tensor.to(device)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16,enabled=False):
                        hyp = greedy_decode(model, src, src_length, src_mask, trg_input, trg_in_len)
                    all_dtw_dev += calculate_dtw(trg.float(), hyp, trim="counter", ref_lengths=trg_len)
            dev_avg_loss=dev_total_loss/len(ds_dev)
            dev_dtw=float(np.mean(all_dtw_dev))
            all_dtw_test=[]
            for batch in dl_test:
                model.eval()
                with torch.no_grad():
                    padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens = batch
                    text_tokens = text_tokens.to(device)
                    src = text_tokens["input_ids"][:, 1:]
                    src_mask = text_tokens["attention_mask"][:, 1:]
                    src_length = mask_to_lengths(src_mask)
                    src_mask = src_mask.unsqueeze(1).bool()
                    trg_input, trg, loss_mask, trg_mask, norm_info = prepare_trg(
                        padded_cod_data, input_length_tensor, used_jts, device)
                    trg_len = (input_length_tensor - 1).clamp(min=1)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16,enabled=False):
                        # (B, 1, T′, T′)
                        skel_out, _ = model(src=src, trg_input=trg_input, src_mask=src_mask, src_lengths=src_length,
                                            trg_mask=trg_mask)

                        loss = masked_regression_loss(skel_out, trg, loss_mask)
                    test_total_loss+=loss.item()*len(padded_cod_data)
                    trg_in_len=input_length_tensor.to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16,enabled=False):
                        hyp = greedy_decode(model, src, src_length, src_mask, trg_input, trg_in_len)
                    all_dtw_test += calculate_dtw(trg.float(), hyp, trim="counter", ref_lengths=trg_len)

            test_avg_loss=test_total_loss/len(ds_test)
            test_dtw=float(np.mean(all_dtw_test))

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}"
                  f", Dev Loss: {dev_avg_loss:.4f}, Test Loss: {test_avg_loss:.4f}")
            torch.save({"model":model.state_dict(),"ema":ema.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch+1,},os.path.join(save_path,f"checkpoint.pth"))
            scheduler.step(metrics=dev_avg_loss)

            if dev_dtw<min_loss:
                min_loss=dev_dtw
                print(f"Best loss updated: {min_loss:.4f}.")
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "epoch": epoch + 1, },
                           os.path.join(save_path, f"best_checkpoint.pth"))


            log_dict = {"train_loss": avg_loss, "dev_loss": dev_avg_loss, "test_loss": test_avg_loss}
            wandb.log(log_dict)
            current_lr = -1
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']  # 最後の param group の lr だけを見る            if current_lr <  0.0002:
            if current_lr < 0.0002:
                stop = True
            if stop:
                print("Learning rate is too low. Stopping training.")
                break
            if step>1:
                i=0
                model.eval()
                for batch in dl_dev:
                    if i >= 1:
                        break
                    with torch.no_grad():
                        padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens = batch
                        text_tokens = text_tokens.to(device)
                        all_inputs=padded_cod_data[:,:,bones_used_joints(bones=bones,include_aux=True)]
                        all_inputs=all_inputs.transpose(-1,-2).float().to(device)
                        src = text_tokens["input_ids"][:, 1:]
                        src_mask = text_tokens["attention_mask"][:, 1:]
                        src_length = mask_to_lengths(src_mask)
                        src_mask = src_mask.unsqueeze(1).bool()
                        trg_input, trg, loss_mask, trg_mask, norm_info = prepare_trg(
                            padded_cod_data, input_length_tensor, used_jts, device)
                        trg_len = (input_length_tensor - 1).clamp(min=1)
                        all_mask=lengths_to_mask(trg_len)
                        encoder_output,encoder_hidden=model.encode(src,src_length,src_mask)
                        stacked_output,_=greedy(
                            encoder_output=encoder_output,
                            src_mask=src_mask,
                            embed=model.trg_embed,
                            decoder=model.decoder,
                            trg_input=trg_input,
                            model=model
                        )
                        stacked_output = denormalize(stacked_output, norm_info, offset=0, center_mode="median")  # (B, T′+1, 3, J′)
                        idx=random.randint(0,stacked_output.shape[0]-1)
                        stacked_output=stacked_output[idx:idx+1].float()

                        _,T,J,C=stacked_output.shape
                        # reshape: データは [x0..x47, y0..y47, z0..z47] の順で格納
                        pred_cods=stacked_output
                        length=input_length_tensor[idx].item()
                        visualize_skeletons(pred_cods[:,:length],all_inputs[idx:idx+1,:length],data_path[idx:idx+1],input_length_tensor[idx:idx+1],
                                             f"{save_path}/visualize/",bones=mapped_bones,mask=all_mask[idx:idx+1])
                        i+=1

                step=0
        wandb.alert(
            title="Finish",

            text='無事学習が終了しました。'
        )

    else:

        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=ds_test.collate_fn)
        dl_dev= torch.utils.data.DataLoader(ds_dev, batch_size=1, shuffle=False, num_workers=4, collate_fn=ds_dev.collate_fn)


        """
        # ---- boundary連続性の診断 (最初のdevバッチで1回だけ) ----
        diag_batch = next(iter(dl_dev))
        padded_cod_data, _, input_length_tensor, _, _, text_tokens = diag_batch
        padded_cod_data = padded_cod_data.float().to(device)
        text_tokens = text_tokens.to(device)
        input_length_tensor = input_length_tensor.to(device)
        with torch.no_grad():
            text_emb = text_encoder(**text_tokens).last_hidden_state
        context_mask = text_tokens['attention_mask'].to(device)
        torch.manual_seed(42)  # 再現可能な診断のため固定
        sample_with_diagnostics(
            ema.ema_model, n=padded_cod_data.shape[0],
            T=int(input_length_tensor.max()),
            steps=50, cond=text_emb, cond_mask=context_mask,
            guidance_scale=3.0, lengths=input_length_tensor,
            save_plot=f"{save_path}/sampling_diagnostics.png")
        """
        model.load_state_dict(torch.load(f"{save_path}/best_checkpoint.pth", map_location=device)['model'])
        model.eval()
        print("ema_model training:",model.training)
        if os.path.exists(f"{save_path}/visualize_dev/"):
            shutil.rmtree(f"{save_path}/visualize_dev/")
        if os.path.exists(f"{save_path}/visualize_test/"):
            shutil.rmtree(f"{save_path}/visualize_test/")
        os.makedirs(f"{save_path}/visualize_dev/", exist_ok=True)
        os.makedirs(f"{save_path}/visualize_test/", exist_ok=True)
        used_jts = bones_used_joints(bones=bones, include_aux=True)
        o2c = {o: i for i, o in enumerate(used_jts)}
        mapped_bones = [(o2c[p], o2c[c], o2c[a]) for p, c, a in bones]
        for batch in dl_dev:
            model.eval()
            with torch.no_grad():
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens = batch
                text_tokens = text_tokens.to(device)
                src = text_tokens["input_ids"][:, 1:]
                src_mask = text_tokens["attention_mask"][:, 1:]
                src_length = mask_to_lengths(src_mask)
                src_mask = src_mask.unsqueeze(1).bool()
                trg_input, trg, loss_mask, trg_mask, norm_info = prepare_trg(
                    padded_cod_data, input_length_tensor, used_jts, device)
                trg_len = (input_length_tensor - 1).clamp(min=1)
                encoder_output, encoder_hidden = model.encode(src, src_length, src_mask)
                stacked_output, _ = greedy(
                    encoder_output=encoder_output,
                    src_mask=src_mask,
                    embed=model.trg_embed,
                    decoder=model.decoder,
                    trg_input=trg_input,
                    model=model
                )
                B, T, D = stacked_output.shape
                stacked_output = denormalize(stacked_output, norm_info, offset=0,
                                             center_mode="median")  # (B, T′+1, 3, J′)

                pred_cods=stacked_output
                all_inputs=padded_cod_data[:,:,bones_used_joints(bones=bones,include_aux=True)]
                all_inputs=all_inputs.transpose(-1,-2).float().to(device)
                for i in range(pred_cods.shape[0]):
                    pred_video_path = f"{save_path}/visualize_dev/video/{os.path.basename(data_path[i])}.mp4"
                    pred_csv_path = f"{save_path}/visualize_dev/csv/{os.path.basename(data_path[i])}.csv"
                    os.makedirs(os.path.dirname(pred_video_path), exist_ok=True)
                    os.makedirs(os.path.dirname(pred_csv_path), exist_ok=True)

                    #pred_cods = np.where(pred_cods == 0, np.nan, pred_cods)
                    #pred_cods = apply_savgol_filter(pred_cods, window_size=7, poly_order=2)
                    #pred_cods = np.where(np.isnan(pred_cods), 0, pred_cods)
                    save_pred_cods=np.zeros((input_length_tensor[i],D-1))
                    save_pred_cods[:,0::3]=pred_cods[i][:input_length_tensor[i],0,:].cpu().numpy()
                    save_pred_cods[:,1::3]=pred_cods[i][:input_length_tensor[i],1,:].cpu().numpy()
                    save_pred_cods[:,2::3]=pred_cods[i][:input_length_tensor[i],2,:].cpu().numpy()
                    np.savetxt(pred_csv_path, save_pred_cods, delimiter=",")
                    # pred_coordinates=pred_coordinates[:,:,:,bones_used_joints(bones)]
                    # gt_cods=gt_cods[:,:,:,bones_used_joints(bones)]
                    save_skeleton_video(pred_cods[i][:input_length_tensor[i]].cpu().numpy(), pred_video_path,
                                        bones=mapped_bones,
                                        x_ref=all_inputs[i][:input_length_tensor[i]].cpu().numpy())
        for batch in dl_test:
            model.eval()
            with torch.no_grad():
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, text_tokens = batch
                text_tokens = text_tokens.to(device)
                src = text_tokens["input_ids"][:, 1:]
                src_mask = text_tokens["attention_mask"][:, 1:]
                src_length = mask_to_lengths(src_mask)
                src_mask = src_mask.unsqueeze(1).bool()
                trg_input, trg, loss_mask, trg_mask, norm_info = prepare_trg(
                    padded_cod_data, input_length_tensor, used_jts, device)
                trg_len = (input_length_tensor - 1).clamp(min=1)
                encoder_output, encoder_hidden = model.encode(src, src_length, src_mask)
                stacked_output, _ = greedy(
                    encoder_output=encoder_output,
                    src_mask=src_mask,
                    embed=model.trg_embed,
                    decoder=model.decoder,
                    trg_input=trg_input,
                    model=model
                )
                B, T, D = stacked_output.shape
                stacked_output = denormalize(stacked_output, norm_info, offset=0,
                                             center_mode="median")  # (B, T′+1, 3, J′)

                pred_cods = stacked_output
                all_inputs=padded_cod_data[:,:,bones_used_joints(bones=bones,include_aux=True)]
                all_inputs=all_inputs.transpose(-1,-2).float().to(device)
                for i in range(pred_cods.shape[0]):
                    pred_video_path = f"{save_path}/visualize_test/video/{os.path.basename(data_path[i])}.mp4"
                    pred_csv_path = f"{save_path}/visualize_test/csv/{os.path.basename(data_path[i])}.csv"
                    os.makedirs(os.path.dirname(pred_video_path), exist_ok=True)
                    os.makedirs(os.path.dirname(pred_csv_path), exist_ok=True)

                    # pred_cods = np.where(pred_cods == 0, np.nan, pred_cods)
                    # pred_cods = apply_savgol_filter(pred_cods, window_size=7, poly_order=2)
                    # pred_cods = np.where(np.isnan(pred_cods), 0, pred_cods)
                    save_pred_cods = np.zeros((input_length_tensor[i], D-1))
                    save_pred_cods[:, 0::3] = pred_cods[i][:input_length_tensor[i], 0, :].cpu().numpy()
                    save_pred_cods[:, 1::3] = pred_cods[i][:input_length_tensor[i], 1, :].cpu().numpy()
                    save_pred_cods[:, 2::3] = pred_cods[i][:input_length_tensor[i], 2, :].cpu().numpy()
                    np.savetxt(pred_csv_path, save_pred_cods, delimiter=",")
                    # pred_coordinates=pred_coordinates[:,:,:,bones_used_joints(bones)]
                    # gt_cods=gt_cods[:,:,:,bones_used_joints(bones)]
                    save_skeleton_video(pred_cods[i][:input_length_tensor[i]].cpu().numpy(), pred_video_path,
                                        bones=mapped_bones,
                                        x_ref=all_inputs[i][:input_length_tensor[i]].cpu().numpy())
if __name__ == "__main__":
    dataset = "phoenixT"  # or "CSL-Daily", "how2sign", "phoenix", "AUTSL"
    with open(f"/home/caffe/work/SLG/Parameter/config_flowmatch.yaml", "r") as f:
        config = yaml.safe_load(f)
    save_path = ("/media/caffe/data_storage/CSLR/keyword_models/FlowMatching/results_progressive_transformer")
    if os.path.exists(f"{save_path}/flow_match.yaml"):
        with open(f"{save_path}/flow_match.yaml", "r") as f:
            config = yaml.safe_load(f)
    dataset=config['dataset']
    #main(config, dataset, save_path, visualize=False, used_3d=False)
    main(config, dataset, save_path,  visualize=True, used_3d=False)


