"""
model_translation.py の PyTorch Lightning 依存を排した plain PyTorch 版。

対応関係:
  L.LightningModule            -> nn.Module
  save_hyperparameters()       -> self.config に保持するのみ
  self.log(...)                -> self._log(name, value) (dict に蓄積 + 任意のlogger呼び出し)
  training_step / validation_step / test_step
      -> それぞれ内部で同じロジックを持つが、epoch loop 自体は
         train_epoch / validate_epoch / test_epoch が dataloader を回して呼び出す
  on_train_epoch_end 等          -> 各 *_epoch メソッドの末尾に統合
  configure_optimizers()       -> build_optimizer() (Lightning形式のtupleではなく
                                    optimizer, scheduler をそのまま返す)
  self.current_epoch / self.global_step
      -> 明示的なインスタンス変数として自前でインクリメント

外部の学習ループ (Trainer.fit 相当) は Transformer.fit() に、
Trainer.test 相当は Transformer.run_test() にまとめてある。
"""

import torch

from typing import Union, Optional, Callable
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from search import beam_search
from stitch import stitch_poses
from helpers import save_text_file
from losses import set_loss_function
from metrics import bleu, rouge, chrf, wer
from initialization import initialize_model
from constants import PAD_ID, BOS_ID, EOS_ID
from transformer.encoder import TransformerEncoder
from transformer.decoder import TransformerDecoder
from transformer.token_embedding import Token_Embeddings
from back_translation.back_translate import back_translate, make_back_translation_model


class Transformer(nn.Module):
    def __init__(
        self,
        config: dict,
        save_path: Union[Path, str],
        train_batch_size: int,
        dev_batch_size: int,
        src_vocab: dict,
        output_size: int,
        fps: int,
        ground_truth_text: dict,
        codebook_pose: torch.Tensor,
        logger: Optional[Callable[[str, float, int], None]] = None,
    ):
        """
        logger: 任意のcallback。呼ばれ方は logger(metric_name, value, step)。
                None の場合は self.history に蓄積するだけで外部出力はしない。
        """
        super().__init__()
        model_config = config["model"]

        # Lightning の save_hyperparameters() に相当する情報保持。
        # チェックポイントに config を含めたい場合は self.config を state_dict と
        # 一緒に手動で pickle/torch.save すればよい。
        self.config = config

        self.train_cal_metrics = model_config["train_cal_metrics"]

        self.encoder = None
        self.decoder = None
        self.text_vocab = src_vocab
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.input_size = len(src_vocab)
        self.output_size = output_size
        self.make_encoder(model_config["encoder"])
        self.make_decoder(model_config["decoder"])

        self.beam_settings = model_config.get("beam_setting", None)
        if self.beam_settings is None:
            self.decoding = "greedy"
        else:
            self.decoding = "beam" if self.beam_settings["beam_size"] > 1 else "greedy"

        self.src_embedding = Token_Embeddings(
            **model_config["encoder"]["embeddings"],
            vocab_size=self.input_size,
            padding_idx=PAD_ID,
        )

        self.trg_embedding = Token_Embeddings(
            **model_config["decoder"]["embeddings"],
            vocab_size=self.output_size,
            padding_idx=PAD_ID,
        )

        self.vocab_layer = torch.nn.Linear(
            model_config["encoder"]["hidden_size"], self.output_size
        )

        self.translation_dir = Path(save_path) / "translation_hyps"
        self.translation_dir.mkdir(parents=True, exist_ok=True)

        initialize_model(
            self,
            cfg=model_config["initialization"],
            src_padding_idx=PAD_ID,
            trg_padding_idx=PAD_ID,
        )

        # get mapping from token to pose
        self.codebook_pose = codebook_pose

        # loss function
        loss_cfg = model_config["losses"]
        self.token_loss = set_loss_function(loss_cfg["token_loss"])
        self.token_weight = loss_cfg["token_weight"]

        # learning rate
        self.learning_rate = model_config["learning_rate"]
        self.scheduler_settings = model_config["scheduler"]
        self.optimizer_settings = model_config["optimizer"]

        # validation plotting
        self.save_val_frequency = model_config["save_val_frequency"]
        self.keep_predictions = False

        # pose evaluation
        self.stitch_config = config["stitch"]
        self.bt_model = make_back_translation_model(config["backTran_model_dir"])
        self.plot_n = config.get("plot_n", 3)
        self.plot_path = Path(save_path) / "plots"
        self.fps = fps
        self.gt_text = ground_truth_text

        # store the predictions
        splits = ["train", "dev", "test"]
        self.predictions = {}
        for split in splits:
            self.predictions[split] = {
                "pred": {"vq_codes": []},
                "trg": {"vq_codes": []},
            }

        # Lightning が自動で持っていた状態を手動管理する
        self.current_epoch = 0
        self.global_step = 0

        # metric logging: {metric_name: [(step, value), ...]}
        self.history = {}
        self._logger = logger

    # ------------------------------------------------------------------
    # ロギング (self.log の代替)
    # ------------------------------------------------------------------
    def _log(self, name: str, value, step: Optional[int] = None):
        if torch.is_tensor(value):
            value = value.detach().item() if value.numel() == 1 else value.detach().cpu().tolist()
        step = self.global_step if step is None else step
        self.history.setdefault(name, []).append((step, value))
        if self._logger is not None:
            self._logger(name, value, step)

    def on_train_start(self):
        if self.decoding == "beam":
            self._log("Beam_size", self.beam_settings["beam_size"], step=self.current_epoch)
            self._log("Beam_alpha", self.beam_settings["beam_alpha"], step=self.current_epoch)

        # Log all parameters
        for name, param in self.named_parameters():
            if name == "src_vocab":
                continue
            self.log_param(name, param)

    def log_param(self, name, param):
        self._log(f"param/{name}", param.norm().item())

    # ------------------------------------------------------------------
    # モデル本体
    # ------------------------------------------------------------------
    def encode_decode(self, src, src_length, src_mask, trg_input, trg_mask, **kwargs):
        x = self.src_embedding(src)
        encoder_output = self.encoder(x, src_length, src_mask)

        trg_embed = self.trg_embedding(trg_input)
        decoder_output, attention = self.decoder(
            encoder_output=encoder_output,
            src_mask=src_mask,
            trg_embed=trg_embed,
            trg_mask=trg_mask,
            return_attention=False,
        )
        return self.vocab_layer(decoder_output)

    def cal_loss(self, phase, batch_size, vq_codes, trg, trg_mask, **kwargs):
        # cal loss
        vq_codes = F.log_softmax(vq_codes, dim=-1)
        token_loss = self.token_loss(vq_codes, trg)

        self._log(f"{phase}_token_loss", self.token_weight * token_loss)

        loss = self.token_weight * token_loss
        norm_batch_loss = loss / trg.shape[0]

        self._log(f"{phase}_loss", norm_batch_loss)

        return norm_batch_loss

    def cal_metrics(self, phase):
        self.post_process_prediction(phase, "pred")
        self.post_process_prediction(phase, "trg")

        vq_pred = self.predictions[phase]["pred"]["vq_codes"]
        vq_trg = self.predictions[phase]["trg"]["vq_codes"]

        if len(vq_pred) != 0 and len(vq_trg) != 0:
            # convert pred and trg to str
            vq_pred = [" ".join(map(str, code)) for code in vq_pred]
            vq_trg = [" ".join(map(str, code)) for code in vq_trg]

            # save translation hyps
            if self.current_epoch % self.save_val_frequency == 0 and phase == "dev":
                save_text_file(
                    self.translation_dir
                    / f"{self.current_epoch}_{self.global_step}.hyp",
                    vq_pred,
                )
                # save trg for first epoch
                if self.global_step == 0 and self.current_epoch == 0:
                    save_text_file(
                        self.translation_dir
                        / f"{self.current_epoch}_{self.global_step}.trg",
                        vq_trg,
                    )

            bleu_score = bleu(vq_pred, vq_trg)
            rouge_score = rouge(vq_pred, vq_trg)
            chrf_score = chrf(vq_pred, vq_trg)

            for i in range(1, 5):
                self._log(f"{phase}_bleu_{i}", bleu_score[f"bleu{i}"])
            self._log(f"{phase}_rouge", rouge_score)
            self._log(f"{phase}_chrf", chrf_score)

    def beam_decode(self, src, src_length, src_mask, **kwargs):
        x = self.src_embedding(src)
        encoder_output = self.encoder(x, src_length, src_mask)

        code_prediction, _, _ = beam_search(
            model=self,
            encoder_output=encoder_output,
            src_mask=src_mask,
            **self.beam_settings,
        )
        return code_prediction

    def greedy_decode(
        self, src, src_length, src_mask, max_output_length: int, **kwargs
    ):
        batch_size = src_mask.size(0)

        x = self.src_embedding(src)
        encoder_output = self.encoder(x, src_length, src_mask)

        # start with BOS-symbol for each sentence in the batch
        ys = encoder_output.new_full([batch_size, 1], BOS_ID, dtype=torch.long)

        # a subsequent mask is intersected with this in decoder forward pass
        trg_mask = src_mask.new_ones([1, 1, 1])

        finished = src_mask.new_zeros(batch_size).byte()

        for _ in range(max_output_length):
            # pylint: disable=unused-variable
            with torch.no_grad():
                trg_embed = self.trg_embedding(ys)
                x, _ = self.decoder(
                    encoder_output=encoder_output,
                    src_mask=src_mask,
                    trg_embed=trg_embed,
                    trg_mask=trg_mask,
                    return_attention=False,
                )

                logits = self.vocab_layer(x)
                logits = logits[:, -1]
                _, next_word = torch.max(logits, dim=1)
                next_word = next_word.data
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

            # check if previous symbol was <eos>
            is_eos = torch.eq(next_word, EOS_ID)
            finished += is_eos
            # stop predicting if <eos> reached for all elements in batch
            if (finished >= 1).sum() == batch_size:
                break

        ys = ys[:, 1:]  # remove BOS-symbol
        return ys

    def evaluate_pose(self, split: str, test: str = "pred", log: bool = True):
        predicted_poses = []
        none_index = []
        for i, codes in enumerate(self.predictions[split][test]["vq_codes"]):
            if len(codes) == 0:
                none_index.append(i)
                continue
            pred_pose = self.codebook_pose[codes]
            n, window_size, _ = pred_pose.shape
            pred_pose = pred_pose.reshape(n, window_size, -1, 3)
            pred_pose = stitch_poses(poses=pred_pose, stitch_config=self.stitch_config)
            predicted_poses.append(pred_pose)
        pred_text = back_translate(model=self.bt_model, poses=predicted_poses)

        # add back empty sequences
        for i in none_index:
            pred_text.insert(i, "")
            predicted_poses.insert(i, None)

        bleu_score = bleu(pred_text, self.gt_text[split])
        rouge_score = rouge(pred_text, self.gt_text[split])
        chrf_score = chrf(pred_text, self.gt_text[split])
        wer_score = wer(pred_text, self.gt_text[split])

        if log:
            for i in range(1, 5):
                self._log(f"bt_{split}_bleu_{i}", bleu_score[f"bleu{i}"])
            self._log(f"bt_{split}_rouge", rouge_score)
            self._log(f"bt_{split}_chrf", chrf_score)
            self._log(f"bt_{split}_wer", wer_score)

        return pred_text, predicted_poses

    def make_encoder(self, enc_cfg):
        self.encoder = TransformerEncoder(
            **enc_cfg, emb_dropout=enc_cfg["embeddings"].get("dropout", 0.0)
        )

    def make_decoder(self, dec_cfg):
        # build decoder
        self.decoder = TransformerDecoder(
            **dec_cfg,
            emb_dropout=dec_cfg["embeddings"].get("dropout", 0.0),
            output_size=self.output_size,
        )

    def post_process_prediction(self, split, pred_type):
        vq_codes = self.predictions[split][pred_type]["vq_codes"]

        def crop_at_eos(output_sequence):
            cropped_sequence = []
            # Find EOS token index (assuming the output is a PyTorch tensor)
            for seq in output_sequence:

                eos_index = (seq == EOS_ID).nonzero()[0]
                if eos_index.size == 0:
                    eos_index = -1
                else:
                    eos_index = eos_index[0].item()

                # Slice up to the EOS token
                cropped_sequence.append(seq[:eos_index])
            return cropped_sequence

        vq_codes = crop_at_eos(vq_codes)
        vq_codes = [codes - 4 for codes in vq_codes]  # remove special tokens

        self.predictions[split][pred_type]["vq_codes"] = vq_codes

    # ------------------------------------------------------------------
    # 最適化 (configure_optimizers の代替)
    # ------------------------------------------------------------------
    def build_optimizer(self):
        if self.optimizer_settings["type"].lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                betas=self.optimizer_settings["adam_betas"],
                weight_decay=self.optimizer_settings["weight_decay"],
            )
        else:
            raise NotImplementedError(
                f'Optimizer {self.optimizer_settings["type"]} not implemented'
            )

        if self.scheduler_settings["type"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                mode=self.scheduler_settings["mode"],
                factor=self.scheduler_settings["factor"],
                patience=self.scheduler_settings["patience"],
                min_lr=self.scheduler_settings["min_lr"],
            )
        else:
            raise NotImplementedError(
                f'Scheduler {self.scheduler_settings["type"]} not implemented'
            )

        self._scheduler_monitor = self.scheduler_settings["tracking_metric"]
        self._scheduler_interval = self.scheduler_settings["interval"]  # "epoch" or "step"
        self._scheduler_frequency = self.scheduler_settings["frequency"]

        return optimizer, scheduler

    # ------------------------------------------------------------------
    # epoch loop (Lightning の *_step / on_*_epoch_end をまとめたもの)
    # ------------------------------------------------------------------
    def train_epoch(self, dataloader, optimizer, device):
        self.train()
        for batch_idx, batch in enumerate(dataloader):
            batch = self._batch_to_device(batch, device)

            optimizer.zero_grad()
            code_prediction = self.encode_decode(**batch)
            loss = self.cal_loss(
                "train", self.train_batch_size, code_prediction, **batch
            )
            loss.backward()
            optimizer.step()
            self.global_step += 1

            if self.train_cal_metrics:
                with torch.no_grad():
                    log_probs = F.log_softmax(code_prediction, dim=-1)
                    _, code_prediction_ids = torch.max(log_probs, dim=-1)
                self.predictions["train"]["trg"]["vq_codes"].extend(
                    batch["trg"].detach().cpu().numpy()
                )
                self.predictions["train"]["pred"]["vq_codes"].extend(
                    code_prediction_ids.detach().cpu().numpy()
                )

        self.cal_metrics("train")
        self.predictions["train"]["trg"]["vq_codes"].clear()
        self.predictions["train"]["pred"]["vq_codes"].clear()
        self.current_epoch += 1

    @torch.no_grad()
    def validate_epoch(self, dataloader, device):
        self.eval()
        for batch in dataloader:
            batch = self._batch_to_device(batch, device)

            code_prediction = self.encode_decode(**batch)
            self.cal_loss("dev", self.dev_batch_size, code_prediction, **batch)

            if self.decoding == "greedy":
                max_len = self.beam_settings["max_output_length"]
                if max_len > code_prediction.shape[1]:
                    max_len = code_prediction.shape[1]
                code_prediction = self.greedy_decode(**batch, max_output_length=max_len)
            else:
                code_prediction = self.beam_decode(**batch)

            self.predictions["dev"]["trg"]["vq_codes"].extend(
                batch["trg"].detach().cpu().numpy()
            )
            self.predictions["dev"]["pred"]["vq_codes"].extend(
                code_prediction.detach().cpu().numpy()
            )

        self.cal_metrics("dev")
        self.evaluate_pose("dev")

        if not self.keep_predictions:
            self.predictions["dev"]["trg"]["vq_codes"].clear()
            self.predictions["dev"]["pred"]["vq_codes"].clear()

        # scheduler.step(monitor) はこの戻り値を使って呼び出し側で実行する
        monitor_value = self.history[self._scheduler_monitor][-1][1]
        return monitor_value

    @torch.no_grad()
    def test_epoch(self, dataloader, device):
        self.eval()
        for batch in dataloader:
            batch = self._batch_to_device(batch, device)

            code_prediction = self.encode_decode(**batch)
            self.cal_loss("test", self.dev_batch_size, code_prediction, **batch)

            if self.decoding == "greedy":
                max_len = self.beam_settings["max_output_length"]
                if max_len > code_prediction.shape[1]:
                    max_len = code_prediction.shape[1]
                code_prediction = self.greedy_decode(**batch, max_output_length=max_len)
            else:
                code_prediction = self.beam_decode(**batch)
                self._log("Beam_size", self.beam_settings["beam_size"], step=self.current_epoch)
                self._log("Beam_alpha", self.beam_settings["beam_alpha"], step=self.current_epoch)

            self.predictions["test"]["trg"]["vq_codes"].extend(
                batch["trg"].detach().cpu().numpy()
            )
            self.predictions["test"]["pred"]["vq_codes"].extend(
                code_prediction.detach().cpu().numpy()
            )

        self.cal_metrics("test")
        pred_text, predicted_poses = self.evaluate_pose("test")

        if not self.keep_predictions:
            self.predictions["test"]["trg"]["vq_codes"].clear()
            self.predictions["test"]["pred"]["vq_codes"].clear()

        return pred_text, predicted_poses

    @staticmethod
    def _batch_to_device(batch: dict, device) -> dict:
        return {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }

    # ------------------------------------------------------------------
    # Trainer.fit() / Trainer.test() 相当のドライバ
    # ------------------------------------------------------------------
    def fit(
        self,
        train_dataloader,
        dev_dataloader,
        max_epochs: int,
        device: str = "cuda",
        early_stopping_patience: Optional[int] = None,
    ):
        """
        Lightning の trainer.fit(model, datamodule) に相当。
        ModelCheckpoint / EarlyStopping などの Callback は持たないので、
        必要ならこのループの中に自前で追加する。
        """
        optimizer, scheduler = self.build_optimizer()
        self.to(device)
        self.on_train_start()

        best_monitor = None
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            self.train_epoch(train_dataloader, optimizer, device)
            monitor_value = self.validate_epoch(dev_dataloader, device)

            if self._scheduler_interval == "epoch":
                scheduler.step(monitor_value)

            improved = (
                best_monitor is None
                or (self.scheduler_settings["mode"] == "min" and monitor_value < best_monitor)
                or (self.scheduler_settings["mode"] == "max" and monitor_value > best_monitor)
            )
            if improved:
                best_monitor = monitor_value
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(
                f"[epoch {self.current_epoch}] "
                f"{self._scheduler_monitor}={monitor_value:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            if (
                early_stopping_patience is not None
                and epochs_no_improve >= early_stopping_patience
            ):
                print(f"Early stopping at epoch {self.current_epoch}")
                break

    def run_test(self, test_dataloader, device: str = "cuda"):
        """Lightning の trainer.test(model, datamodule) に相当。"""
        self.to(device)
        return self.test_epoch(test_dataloader, device)

    # ------------------------------------------------------------------
    # checkpoint 互換ロード
    # Lightning のckpt ({"state_dict": ..., "hyper_parameters": ...}) を
    # そのまま読み込めるようにしておく
    # ------------------------------------------------------------------
    def load_lightning_checkpoint(self, ckpt_path: Union[str, Path], strict: bool = True):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        self.load_state_dict(state_dict, strict=strict)
        return ckpt