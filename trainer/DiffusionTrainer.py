"""
DiffusionTrainer
================
SkeletonTextDiffusion (mode = "vae" | "lddm") を学習するトレーナー。
lddm は text を in-context に用いた条件付き Latent Diffusion。
元の VAESyncTrainer のスキャフォールド (AMP / EMA / grad clip / scheduler timing /
wandb / checkpoint / CSV / early stopping) を踏襲しつつ、損失キーがモードごとに
異なるため「モデルが返す loss dict を汎用的に集計・記録する」方式にしている。

期待する config (dict):
    config["mode"]                      : 省略可。未指定なら model.mode を使用
    config["save_path"]                 : 保存先ディレクトリ
    config["init_epoch"]                : 開始エポック (resume 用, 既定 0)
    config["use_wandb"]                 : wandb を使うか (既定 True)
    config["vae_checkpoint"]            : lddm/text モードで読み込む VAE 重み (任意)
    config["history"]                   : resume 用の履歴 dict (任意)
    config["lr_parameters"]:
        "epoch", "amp", "amp_dtype" ("bfloat16"/"float16"),
        "grad_clip_norm", "ema", "ema_beta",
        "scheduler_timing" ("step"/"epoch"), "scheduler_type"

バッチ形式 (どちらでも可):
  - dict : {"skeleton": (B,T,F), "skeleton_length": (B,),
            "input_ids": (B,L), "attention_mask": (B,L)}   # lddm の text 条件付き時のみ後者2つ
  - tuple: 元コード互換 (padded_cod_data, padded_mask, input_length_tensor,
            id_list, data_path, sequence[, input_ids, attention_mask])
"""

import gc
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from loader.coordinate_preprocess import apply_savgol_filter
import shutil
import cv2


try:
    import wandb
except Exception:
    wandb = None

try:
    from models.module.EMA import EMA           # ある場合は既存実装を使う
except Exception:
    EMA = None


def _length_to_padding_mask(lengths, max_len):
    idx = torch.arange(max_len, device=lengths.device)[None, :]
    return idx >= lengths[:, None]


class DiffusionTrainer:
    def __init__(self, config, scheduler=None):
        self.config = config
        self.scheduler = scheduler
        self.step = 0
        self.scaler = None
        self.use_scaler = False
        self.history = defaultdict(list, config.get("history", {}))
        self.use_wandb = config.get("use_wandb", True) and (wandb is not None)

        lp = config["lr_parameters"]
        self.amp = lp.get("amp", False)
        self.amp_dtype = torch.float16 if lp.get("amp_dtype", "bfloat16") == "float16" else torch.bfloat16
        self.grad_clip = lp.get("grad_clip_norm", None)

    # ------------------------------------------------------------------ #
    #  バッチ準備
    # ------------------------------------------------------------------ #
    def _prepare_inputs(self, batch, device, mode):
        if isinstance(batch, dict):
            kwargs = {
                "skeleton": batch["skeleton"].float().to(device),
                "hand_skeleton":batch['skeleton'][:,:,:,-42:].float().to(device),
                "body_skeleton":batch['skeleton'][:,:,:,:-42].float().to(device),
                "skeleton_length": batch["skeleton_length"].to(device),
            }
            # lddm (text 条件付き) では HF 入力を渡す。無条件 LDDM では省略可。
            if mode == "lddm" and "input_ids" in batch:
                kwargs["input_ids"] = batch["input_ids"].to(device)
                kwargs["attention_mask"] = batch["attention_mask"].to(device)
            return kwargs

        # tuple (元コード互換): padded_cod_data, padded_mask, input_length_tensor, ...
        kwargs = {
            "skeleton": batch[0].float().to(device),
            "skeleton_length": batch[2].to(device),
        }
        if mode == "lddm" and len(batch) >= 8:
            # collate で tuple 末尾に input_ids, attention_mask を付与した場合
            kwargs["input_ids"] = batch[-2].to(device)
            kwargs["attention_mask"] = batch[-1].to(device)
        return kwargs

    # ------------------------------------------------------------------ #
    #  AMP / scaler
    # ------------------------------------------------------------------ #
    def _ensure_scaler(self, device):
        if self.scaler is not None:
            return
        device_type = "cuda" if torch.device(device).type == "cuda" else "cpu"
        # GradScaler が要るのは fp16 + cuda のときだけ。bf16 では不要。
        self.use_scaler = self.amp and (self.amp_dtype == torch.float16) and (device_type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler(device_type, enabled=self.use_scaler)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

    def _autocast(self, device):
        device_type = "cuda" if torch.device(device).type == "cuda" else "cpu"
        return torch.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.amp)

    # ------------------------------------------------------------------ #
    #  train / eval
    # ------------------------------------------------------------------ #
    def train(self, model, optimizer, train_loader, device, ema=None):
        mode = getattr(model, "mode", self.config.get("mode"))
        model.train()
        if mode in ("lddm", "text"):
            model.vae.eval()  # 凍結 VAE は eval 固定
        self._ensure_scaler(device)
        accum = defaultdict(list)

        total = len(train_loader)
        for batch_idx, batch in tqdm(enumerate(train_loader), total=total):
            inputs = self._prepare_inputs(batch, device, mode)
            optimizer.zero_grad(set_to_none=True)

            with self._autocast(device):
                output = model(**inputs)
            loss = output["loss"]

            if self.use_scaler:
                self.scaler.scale(loss).backward()
                if self.grad_clip is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()

            if ema is not None:
                ema.update()

            for k, v in output.items():
                if torch.is_tensor(v) and v.ndim == 0:
                    accum[k].append(v.detach().item())

            # step 単位の scheduler
            if self.scheduler is not None and self.config["lr_parameters"]["scheduler_timing"] == "step":
                if self.config["lr_parameters"]["scheduler_type"] == "CosineAnnealingWarmRestarts":
                    self.scheduler.step(self.step)
                else:
                    self.scheduler.step()
                self.step += 1

            if batch_idx % 100 == 0:
                msg = "  ".join(f"{k}: {np.mean(v):.4f}" for k, v in accum.items())
                tqdm.write(f"[train] {msg}")

        return {k: float(np.mean(v)) for k, v in accum.items()}

    @torch.no_grad()
    def eval(self, model, eval_loader, device):
        mode = getattr(model, "mode", self.config.get("mode"))
        model.eval()
        accum = defaultdict(list)
        total = len(eval_loader)
        for batch in tqdm(eval_loader, total=total):
            inputs = self._prepare_inputs(batch, device, mode)
            with self._autocast(device):
                output = model(**inputs)
            for k, v in output.items():
                if torch.is_tensor(v) and v.ndim == 0:
                    accum[k].append(v.detach().item())
        return {k: float(np.mean(v)) for k, v in accum.items()}

    # ------------------------------------------------------------------ #
    #  fit
    # ------------------------------------------------------------------ #
    def fit(self, model, optimizer, scheduler, train_loader, eval_loader, test_loader,
            device, criterion=None, early_stopping=None):
        if scheduler is not None:
            self.scheduler = scheduler
        mode = getattr(model, "mode", self.config.get("mode"))

        # lddm / text モードは事前学習済み VAE を読み込む
        if mode in ("lddm", "text") and self.config.get("vae_checkpoint"):
            model.load_vae(self.config["vae_checkpoint"], map_location=device)
            print(f"loaded VAE from {self.config['vae_checkpoint']}")

        ema = None
        if self.config["lr_parameters"].get("ema", False):
            if EMA is None:
                print("warning: EMA module が見つからないため EMA を無効化します")
            else:
                ema = EMA(model, self.config["lr_parameters"]["ema_beta"])

        num_epochs = self.config["lr_parameters"]["epoch"]
        init_epoch = self.config.get("init_epoch", 0)
        save_path = self.config["save_path"]

        for epoch in range(init_epoch, num_epochs):
            print(f"saved path: {save_path}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.scheduler is not None:
                print(f"base_lr: {self.scheduler.get_last_lr()}")
            print(f"epoch: {epoch}/{num_epochs}  (mode={mode})")
            os.makedirs(f"{save_path}/{epoch}", exist_ok=True)

            print("--train--")
            train_metrics = self.train(model, optimizer, train_loader, device, ema=ema)
            print("--eval--")
            eval_metrics = self.eval(model, eval_loader, device)
            print("--test--")
            test_metrics = self.eval(model, test_loader, device)

            # 履歴へ追記 (キーはモード依存で動的)
            for k, v in train_metrics.items():
                self.history[f"train_{k}"].append(v)
            for k, v in eval_metrics.items():
                self.history[f"eval_{k}"].append(v)
            for k, v in test_metrics.items():
                self.history[f"test_{k}"].append(v)

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("  train:", {k: round(v, 4) for k, v in train_metrics.items()})
            print("  eval :", {k: round(v, 4) for k, v in eval_metrics.items()})
            print("  test :", {k: round(v, 4) for k, v in test_metrics.items()})

            # wandb
            log_dict = {}
            log_dict.update({f"train_{k}": v for k, v in train_metrics.items()})
            log_dict.update({f"eval_{k}": v for k, v in eval_metrics.items()})
            log_dict.update({f"test_{k}": v for k, v in test_metrics.items()})
            if self.use_wandb:
                wandb.log(log_dict)

            # チェックポイント
            torch.save(model.state_dict(), f"{save_path}/{epoch}/model_epoch{epoch}.pth")
            if ema is not None:
                torch.save(ema.ema_model.state_dict(), f"{save_path}/{epoch}/ema_model_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "mode": mode,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "history": dict(self.history),
                "random": random.getstate(),
                "np_random": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda_random": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            }, f"{save_path}/checkpoint.cpt")

            # CSV
            df = pd.DataFrame(dict(self.history))
            df.insert(0, "epoch", list(range(len(df))))
            df.to_csv(f"{save_path}/log.csv", index=False)

            # epoch 単位の scheduler
            if self.scheduler is not None and self.config["lr_parameters"]["scheduler_timing"] == "epoch":
                if self.config["lr_parameters"]["scheduler_type"] == "CosineAnnealingWarmRestarts":
                    self.scheduler.step(epoch + 1)
                else:
                    self.scheduler.step()

            # early stopping (スカラーの eval loss を渡す)
            if early_stopping is not None:
                early_stopping(eval_metrics["loss"], model)
                if getattr(early_stopping, "early_stop", False):
                    print("Early stopping")
                    break

        if self.use_wandb:
            wandb.alert(title="Finish", text="無事学習が終了しました。")
        return

    # ------------------------------------------------------------------ #
    #  推論 (可視化用) : モードごとに pose を生成
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def infer_skeleton(self, model, skeleton, skeleton_length, inputs=None,
                       guidance_scale=None, sampler="ddim",
                       num_inference_steps=50, eta=0.0, device="cpu"):
        """(B,T,F) の skeleton を生成/復元する。
        - vae : 入力 skeleton を VAE で再構成
        - lddm: text を in-context に与えた条件付きサンプリング (text なしなら無条件)
        sampler: "ddpm" | "ddim"。ddim では num_inference_steps / eta が有効。
        inputs: text 条件付き生成のとき {"input_ids":..., "attention_mask":...} を渡す
        """
        B,T,J,C= skeleton.shape
        skeleton=skeleton.reshape(B,T,J*C)
        mode = getattr(model, "mode", self.config.get("mode"))
        T = skeleton.shape[1]
        pad_mask = _length_to_padding_mask(skeleton_length, T)
        if mode == "vae":
            recon, _, _, _ = model.vae(skeleton, pad_mask)
            return recon
        # mode == "lddm"
        input_ids = inputs.get("input_ids") if inputs else None
        if len(input_ids.shape)==1 : # (L,) -> (1,L)
            input_ids = input_ids.unsqueeze(0)
        attention_mask = inputs.get("attention_mask") if inputs else None
        if len(attention_mask.shape)==1 : # (L,) -> (1,L)
            attention_mask = attention_mask.unsqueeze(0)
        if sampler == "ddim":
            return model.sample_ddim(
                seq_len=T, input_ids=input_ids, attention_mask=attention_mask,
                skeleton_length=skeleton_length, guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps, eta=eta, device=device)
        return model.sample(
            seq_len=T, input_ids=input_ids, attention_mask=attention_mask,
            skeleton_length=skeleton_length, guidance_scale=guidance_scale, device=device)

    def visualize(self, model, dataset, device, visualize_dir_name="visualize"):
        # TODO: 出力のposeを可視化する関数を実装
        # model: VAEモデル
        # dataset: 可視化に使用するデータセット
        # device: 使用するデバイス
        # 可視化用のディレクトリを作成
        # 予測したポーズとGTポーズをそれぞれ256x256の白画像にプロットする
        if os.path.exists(f"{self.config['save_path']}/{visualize_dir_name}"):
            shutil.rmtree(f"{self.config['save_path']}/{visualize_dir_name}")
        os.makedirs(f"{self.config['save_path']}/{visualize_dir_name}", exist_ok=True)
        # 予測したポーズとGTポーズを保存するディレクトリを作成
        os.makedirs(f"{self.config['save_path']}/{visualize_dir_name}/GT", exist_ok=True)
        os.makedirs(f"{self.config['save_path']}/{visualize_dir_name}/Pred", exist_ok=True)
        model.eval()
        dataset.set_return_length()
        with torch.no_grad():
            for batch in tqdm(dataset, total=len(dataset)):
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, sequence, center_data, shoulder_length, left_center_data, left_length, right_center_data, right_length = batch
                padded_cod_data = padded_cod_data.float().unsqueeze(0).to(device)
                padded_mask = padded_mask.unsqueeze(0).to(device)
                input_length_tensor = input_length_tensor.unsqueeze(0).to(device)
                id_list = torch.tensor(id_list).to(device)
                sequence = sequence.to(device)
                input_ids=sequence["input_ids"].unsqueeze(0).to(device)
                attention_mask=sequence["attention_mask"].unsqueeze(0).to(device)
                batch = (padded_cod_data, padded_mask, input_length_tensor, id_list)
                output = self.infer_skeleton(model, padded_cod_data,input_length_tensor,sequence,sampler="ddim",device=device)
                output = output.squeeze(0).cpu().numpy()
                T, JC = output.shape
                output = output.reshape(T, JC // 3, 3)
                T, J, C = output.shape
                # outputを元のスケールに戻す
                output = output.reshape(T, J, C)
                output[:, :8] *= shoulder_length.cpu().numpy()[:, None, None]
                output[:, :8] += center_data.cpu().transpose(0, 1).numpy()[:, None, :]
                pd_left_center_data = output[:, 6]
                pd_right_center_data = output[:, 7]
                output[:, 8:29] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                output[:, 8:29] += pd_left_center_data[:, None, :]
                output[:, 29:] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                output[:, 29:] += pd_right_center_data[:, None, :]

                # output=average_movint(output.reshape(T,J*C),window_size=7).reshape(T,J,C)
                output = np.where(output == 0, np.nan, output)
                output = apply_savgol_filter(output.transpose(1, 0, 2), window_size=7, poly_order=2).transpose(1, 0, 2)
                output = np.where(np.isnan(output), 0, output)
                # output[:,8:29]で，全てが0.01以下のときは，手が検出されなかったとして，0にする
                hand_mask = (np.max(np.abs(output[:, 8:29]), axis=2) < 0.01)
                output[:, 8:29][hand_mask] = 0.0
                # output[:,29:]も同様
                hand_mask = (np.max(np.abs(output[:, 29:]), axis=2) < 0.01)
                output[:, 29:][hand_mask] = 0.0

                padded_cod_data = padded_cod_data.squeeze(0).cpu().numpy()
                padded_cod_data = padded_cod_data.reshape(T, J, C)
                padded_cod_data[:, :8] *= shoulder_length.cpu().numpy()[:, None, None]
                padded_cod_data[:, :8] += center_data.cpu().transpose(0, 1).numpy()[:, None, :]
                padded_cod_data[:, 8:29] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                padded_cod_data[:, 8:29] += left_center_data.cpu().transpose(0, 1).numpy()[:, None, :]
                padded_cod_data[:, 29:] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                padded_cod_data[:, 29:] += right_center_data.cpu().transpose(0, 1).numpy()[:, None, :]
                # outputの点群を動画として保存
                # 同時に，元の動画も保存
                video_size = (512, 512)
                v_writer = cv2.VideoWriter(
                    f"{self.config['save_path']}/{visualize_dir_name}/Pred/pred_{os.path.basename(data_path)}.mp4",
                    cv2.VideoWriter_fourcc(*'mp4v'), 30, video_size)
                v_writer_gt = cv2.VideoWriter(
                    f"{self.config['save_path']}/{visualize_dir_name}/GT/gt_{os.path.basename(data_path)}.mp4",
                    cv2.VideoWriter_fourcc(*'mp4v'), 30, video_size)
                for t in range(T):
                    base_frame = np.ones((video_size[0], video_size[1], 3), dtype=np.uint8) * 255
                    base_frame_gt = np.ones((video_size[0], video_size[1], 3), dtype=np.uint8) * 255
                    for j in range(J):
                        x = int(output[t, j, 0] * video_size[0])
                        y = int(output[t, j, 1] * video_size[1])
                        x_gt = int(padded_cod_data[t, j, 0] * video_size[0])
                        y_gt = int(padded_cod_data[t, j, 1] * video_size[1])
                        pd_frame = cv2.circle(base_frame, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
                        gt_frame = cv2.circle(base_frame_gt, (x_gt, y_gt), radius=2, color=(0, 255, 0), thickness=-1)
                    v_writer.write(pd_frame)
                    v_writer_gt.write(gt_frame)
                v_writer.release()
                v_writer_gt.release()
        return
    def visualize_hand(self, model, dataset, device, visualize_dir_name="visualize"):
        # TODO: 出力のposeを可視化する関数を実装
        # model: VAEモデル
        # dataset: 可視化に使用するデータセット
        # device: 使用するデバイス
        # 可視化用のディレクトリを作成
        # 予測したポーズとGTポーズをそれぞれ256x256の白画像にプロットする
        if os.path.exists(f"{self.config['save_path']}/{visualize_dir_name}"):
            shutil.rmtree(f"{self.config['save_path']}/{visualize_dir_name}")
        os.makedirs(f"{self.config['save_path']}/{visualize_dir_name}", exist_ok=True)
        # 予測したポーズとGTポーズを保存するディレクトリを作成
        os.makedirs(f"{self.config['save_path']}/{visualize_dir_name}/GT", exist_ok=True)
        os.makedirs(f"{self.config['save_path']}/{visualize_dir_name}/Pred", exist_ok=True)
        model.eval()
        dataset.set_return_length()
        with torch.no_grad():
            for batch in tqdm(dataset, total=len(dataset)):
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, sequence, center_data, shoulder_length, left_center_data, left_length, right_center_data, right_length = batch
                padded_cod_data = padded_cod_data.float().unsqueeze(0).to(device)
                padded_mask = padded_mask.unsqueeze(0).to(device)
                input_length_tensor = input_length_tensor.unsqueeze(0).to(device)
                id_list = torch.tensor(id_list).to(device)
                sequence = sequence.to(device)
                input_ids=sequence["input_ids"].unsqueeze(0).to(device)
                attention_mask=sequence["attention_mask"].unsqueeze(0).to(device)
                batch = (padded_cod_data, padded_mask, input_length_tensor, id_list)
                output = self.infer_skeleton(model, padded_cod_data,input_length_tensor,sequence,sampler="ddim",device=device)
                output = output.squeeze(0).cpu().numpy()
                T, JC = output.shape
                output = output.reshape(T, JC // 3, 3)
                T, J, C = output.shape
                # outputを元のスケールに戻す
                output = output.reshape(T, J, C)
                pd_left_center_data = left_center_data.cpu().numpy()
                pd_right_center_data = right_center_data.cpu().numpy()
                output[:, :21] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                output[:, :21] += pd_left_center_data[:, None, :]
                output[:, 21:] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                output[:, 21:] += pd_right_center_data[:, None, :]
                # output=average_movint(output.reshape(T,J*C),window_size=7).reshape(T,J,C)
                output = np.where(output == 0, np.nan, output)
                output = apply_savgol_filter(output.transpose(1, 0, 2), window_size=7, poly_order=2).transpose(1, 0, 2)
                output = np.where(np.isnan(output), 0, output)
                # output[:,8:29]で，全てが0.01以下のときは，手が検出されなかったとして，0にする
                hand_mask = (np.max(np.abs(output[:, 8:29]), axis=2) < 0.01)
                output[:, 8:29][hand_mask] = 0.0
                # output[:,29:]も同様
                hand_mask = (np.max(np.abs(output[:, 29:]), axis=2) < 0.01)
                output[:, 29:][hand_mask] = 0.0

                padded_cod_data = padded_cod_data.squeeze(0).cpu().numpy()
                padded_cod_data = padded_cod_data.reshape(T, J, C)
                padded_cod_data[:, :21] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                padded_cod_data[:, :21] += left_center_data.cpu().transpose(0, 1).numpy()[:, None, :]
                padded_cod_data[:, 21:] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                padded_cod_data[:, 21:] += right_center_data.cpu().transpose(0, 1).numpy()[:, None, :]
                # outputの点群を動画として保存
                # 同時に，元の動画も保存
                video_size = (512, 512)
                v_writer = cv2.VideoWriter(
                    f"{self.config['save_path']}/{visualize_dir_name}/Pred/pred_{os.path.basename(data_path)}.mp4",
                    cv2.VideoWriter_fourcc(*'mp4v'), 30, video_size)
                v_writer_gt = cv2.VideoWriter(
                    f"{self.config['save_path']}/{visualize_dir_name}/GT/gt_{os.path.basename(data_path)}.mp4",
                    cv2.VideoWriter_fourcc(*'mp4v'), 30, video_size)
                for t in range(T):
                    base_frame = np.ones((video_size[0], video_size[1], 3), dtype=np.uint8) * 255
                    base_frame_gt = np.ones((video_size[0], video_size[1], 3), dtype=np.uint8) * 255
                    for j in range(J):
                        x = int(output[t, j, 0] * video_size[0])
                        y = int(output[t, j, 1] * video_size[1])
                        x_gt = int(padded_cod_data[t, j, 0] * video_size[0])
                        y_gt = int(padded_cod_data[t, j, 1] * video_size[1])
                        pd_frame = cv2.circle(base_frame, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
                        gt_frame = cv2.circle(base_frame_gt, (x_gt, y_gt), radius=2, color=(0, 255, 0), thickness=-1)
                    v_writer.write(pd_frame)
                    v_writer_gt.write(gt_frame)
                v_writer.release()
                v_writer_gt.release()
        return
