import torch
import random
import lightning as L

from tqdm import tqdm
from pathlib import Path

from NSVQ import NSVQ
from metrics import mse
from losses import set_loss_function
from scheduler import make_scheduler
from initialization import initialize_model
from helpers import create_transformer_mask
from transformer.encoder import TransformerEncoder
from transformer.decoder import TransformerDecoder
from plot import (
    make_pose_video,
    plot_codebook_usage,
    plot_codebook_pca,
    make_square_pose_video,
)


class VQ_Transformer(L.LightningModule):
    def __init__(
        self,
        config: dict,
        train_batch_size: int,
        dev_batch_size: int,
        dataset: L.LightningDataModule,
        input_size: int,
        # signs: dict,
        model_dir: str,
        fps: int,
        loggers: dict,
    ):
        super().__init__()
        model_config = config["model"]
        self.save_hyperparameters(
            ignore=["text_vocab", "gloss_vocab", "dataset", "loggers"]
        )
        self.train_cal_metrics = model_config["train_cal_metrics"]

        self.encoder = None
        self.decoder = None
        self.codebook = None

        self.input_size = input_size
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.window_size = config["data"]["window_size"]

        self.make_encoder(model_config["encoder"])
        self.make_decoder(model_config["decoder"])
        model_config["codebook"]["embedding_dim"] = (
            model_config["encoder"]["hidden_size"] * self.window_size
        )
        self.make_codebook(model_config["codebook"], loggers)
        self.register_parameter("codebooks_parameter", self.codebook.codebooks)
        self.use_codebook = model_config.get("use_codebook", True)

        hidden_size = model_config["encoder"]["hidden_size"]

        self.pose_input_layer = torch.nn.Linear(self.input_size, hidden_size)
        self.counter_input_layer = torch.nn.Linear(1, hidden_size)

        self.pose_output_layer = torch.nn.Linear(hidden_size, self.input_size)
        self.counter_output_layer = torch.nn.Linear(hidden_size, 1)

        # loss function
        self.recon_loss = set_loss_function("mse")
        self.recon_scale = model_config["losses"]["recon_weight"]

        self.counter_loss = set_loss_function("mse")
        self.counter_scale = model_config["losses"]["counter_weight"]

        # learning rate
        self.learning_rate = model_config["learning_rate"]
        self.scheduler_settings = model_config["scheduler"]
        self.optimizer_settings = model_config["optimizer"]

        initialize_model(self, cfg=model_config["initialization"])

        # validation plotting
        self.val_frequency = model_config.get("val_frequency", 1)
        self.plot_val_frequency = model_config["plot_val_frequency"]
        self.keep_predictions = False
        self.plot_path = Path(model_dir) / "plots"
        self.plot_path.mkdir(parents=True, exist_ok=True)
        self.fps = fps

        self.plot_n_seq = config.get("plot_n_sequences", 1)

        # track which gloss map to which codebook
        self.codebook_map = None

        # tack first batch of val to plot
        self.batch_one = True
        # get test images to plot
        if dataset is not None:
            if dataset.test is not None:
                self.test_batch = dataset.test.get_test_batch(self.plot_n_seq)
            if dataset.dev is not None:
                self.val_batch = dataset.dev.get_test_batch(self.plot_n_seq)

        self.encoder_predictions = []  # used to run codebook replacement
        self.codebook_predictions = []  # used to plot codebook pca
        self.total_train_batches = 0

    def on_train_start(self):
        # Log all parameters
        for name, param in self.named_parameters():
            self.log(f"param/{name}", param.norm().item())

    def encode(self, src, src_length, src_mask, **kwargs):
        src = self.pose_input_layer(src)
        return self.encoder(src, src_length, src_mask)

    def query_codebook(self, encoder_output):
        # new method, query codebook with CLS token ONLY
        quantized_input, hard_quantized_input = self.codebook(encoder_output)
        quantized_input = quantized_input.unsqueeze(1)
        hard_quantized_input = hard_quantized_input.unsqueeze(1)

        return quantized_input, hard_quantized_input

    def decode(self, trg_embed, encoder_output, src_mask, trg_mask):
        return self.decoder(trg_embed, encoder_output, src_mask, trg_mask)

    def model_forward(
        self, src, src_length, src_mask, decode_codebook: bool = False, **kwargs
    ):
        encoder_output = self.encode(src, src_length, src_mask, **kwargs)

        if self.use_codebook:
            encoder_output_shape = encoder_output.shape
            encoder_output = torch.flatten(encoder_output, -2, -1)
            quantized_input, hard_quantized_input = self.query_codebook(encoder_output)
            quantized_input = quantized_input.reshape(encoder_output_shape)
        else:
            quantized_input = encoder_output.unsqueeze(1)
            hard_quantized_input = None

        trg_embed = self.counter_input_layer(kwargs["trg_input"])
        decoder_output, _ = self.decode(trg_embed, quantized_input, src_mask, src_mask)

        pose = self.pose_output_layer(decoder_output)
        counter = self.counter_output_layer(decoder_output)

        codebook_pose = None
        if hard_quantized_input is not None and decode_codebook:
            hard_quantized_input = hard_quantized_input.reshape(encoder_output_shape)
            codebook_output, _ = self.decoder(
                trg_embed, hard_quantized_input, src_mask, src_mask
            )
            codebook_pose = self.pose_output_layer(codebook_output)

        return encoder_output, quantized_input, pose, counter, codebook_pose

    def cal_loss(self, phase, pose_pred, counter_pred, **kwargs):
        batch_size = pose_pred.shape[0]
        # cal loss
        reconstruction_loss = self.recon_loss(pose_pred, kwargs["src"]) / batch_size

        reconstruction_loss = reconstruction_loss * self.recon_scale

        counter_loss = self.counter_loss(counter_pred, kwargs["trg_input"]) / batch_size

        assert not torch.isnan(pose_pred).any()
        assert not torch.isnan(kwargs["src"]).any()
        assert not torch.isnan(counter_pred).any()
        assert not torch.isnan(kwargs["trg_input"]).any()

        self.log(
            f"{phase}_Reconstruction_loss",
            reconstruction_loss,
            prog_bar=False,
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
        )

        self.log(
            f"{phase}_counter_loss",
            counter_loss,
            prog_bar=False,
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
        )

        loss = (reconstruction_loss * self.recon_scale) + (
            counter_loss * self.counter_scale
        )

        self.log(
            f"{phase}_loss",
            loss,
            prog_bar=False,
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
        )

        return loss

    def cal_metrics(self, phase, trg_pose, pred_pose, trg_counter, pred_counter, mask):
        if len(trg_pose) != 0 and len(pred_pose) != 0:
            pred_pose = pred_pose.detach().cpu()
            trg_pose = trg_pose.detach().cpu()

            pose_mse = mse(trg_pose.numpy(), pred_pose.numpy())
            counter_mse = mse(
                trg_counter.detach().cpu().numpy(), pred_counter.detach().cpu().numpy()
            )

            self.log(f"{phase}_pose_MSE", pose_mse, prog_bar=True)
            self.log(f"{phase}_counter_MSE", counter_mse, prog_bar=True)

    def training_step(self, batch, batch_idx):
        self.total_train_batches += 1
        encoder_output, codebook_output, pose, counter, _ = self.model_forward(**batch)

        self.codebook_predictions.extend(codebook_output.detach().cpu())
        self.encoder_predictions.extend(encoder_output.detach().cpu())

        loss = self.cal_loss(
            phase="train",
            batch_size=pose.shape[0],
            pose_pred=pose,
            counter_pred=counter,
            trg=batch["src"],
            **batch,
        )

        if self.train_cal_metrics:
            self.cal_metrics(
                phase="train",
                trg_pose=batch["src"],
                pred_pose=pose,
                trg_counter=batch["trg_input"],
                pred_counter=counter,
                mask=batch["src_mask"],
            )

        return loss

    def on_train_epoch_end(self) -> None:
        if self.use_codebook:
            if self.current_epoch % self.plot_val_frequency == 0:
                plot_codebook_usage(
                    usage=(self.codebook.codebooks_used / self.total_train_batches),
                    epoch=self.current_epoch,
                    save_dir=self.plot_path,
                )

                plot_codebook_pca(
                    embeddings=torch.stack(self.codebook_predictions, dim=0),
                    codebook=self.codebook.codebooks.data,
                    epoch=self.current_epoch,
                    save_dir=self.plot_path,
                )

            _, self.total_train_batches = self.codebook.run_replacement(
                self.current_epoch,
                self.total_train_batches,
                torch.stack(self.encoder_predictions, dim=0),
            )
            self.codebook_predictions.clear()
            self.encoder_predictions.clear()

    def validation_step(self, batch, batch_idx):
        # called during training
        self.eval()

        encoder_output, codebook_output, pose, counter, _ = self.model_forward(**batch)

        self.cal_loss(
            phase="val",
            batch_size=pose.shape[0],
            pose_pred=pose,
            counter_pred=counter,
            trg=batch["src"],
            **batch,
        )

        self.cal_metrics(
            phase="val",
            trg_pose=batch["src"],
            pred_pose=pose,
            trg_counter=batch["trg_input"],
            pred_counter=counter,
            mask=batch["src_mask"],
        )

        # plot results of embedding
        if self.current_epoch % self.plot_val_frequency == 0 and self.batch_one:
            self.plot_results("val", self.val_batch)
            self.batch_one = False

    def on_validation_epoch_end(self):
        self.batch_one = True

    def test_step(self, batch, batch_idx):
        # called at the end training
        self.eval()

        if self.batch_one:
            self.plot_results("test", self.test_batch)
            self.batch_one = False

        encoder_output, codebook_output, pose, counter, _ = self.model_forward(**batch)

        self.cal_loss(
            phase="test",
            batch_size=pose.shape[0],
            pose_pred=pose,
            counter_pred=counter,
            trg=batch["src"],
            **batch,
        )

        self.cal_metrics(
            phase="test",
            trg_pose=batch["src"],
            pred_pose=pose,
            trg_counter=batch["trg_input"],
            pred_counter=counter,
            mask=batch["src_mask"],
        )

    def on_test_epoch_start(self):
        # plot the codebook
        if self.use_codebook:
            self.plot_codebook()

    def configure_optimizers(self):
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

        schedular = make_scheduler(self.scheduler_settings, optimizer)

        schedular = {
            "scheduler": schedular,
            "monitor": self.scheduler_settings["tracking_metric"],
            "interval": self.scheduler_settings["interval"],
            "frequency": self.scheduler_settings["frequency"],
        }

        return [optimizer], [schedular]

    def make_encoder(self, enc_cfg):
        self.encoder = TransformerEncoder(
            **enc_cfg, emb_dropout=enc_cfg.get("dropout", 0.0)
        )

    def make_decoder(self, dec_cfg):
        self.decoder = TransformerDecoder(
            **dec_cfg, emb_dropout=dec_cfg.get("dropout", 0.0)
        )

    def make_codebook(self, code_cfg, loggers):
        self.codebook = NSVQ(
            num_embeddings=code_cfg["codebook_size"],
            embedding_dim=code_cfg["embedding_dim"],
            discarding_threshold=code_cfg["discard_threshold"],
            initialization=code_cfg["initialization"],
            replace=code_cfg["codebook_replacement"]["replace"],
            replace_epoch=code_cfg["codebook_replacement"]["epochs"],
            call_every=code_cfg["codebook_replacement"]["call_every"],
            mode=code_cfg["codebook_replacement"]["mode"],
            device=torch.device("cuda"),
            loggers=loggers,
        )

    def plot_results(self, phase, batch):
        batch = {k: v.to("cuda") for k, v in batch.items() if v is not None}

        _, _, pred_pose, _, codebook_pose = self.model_forward(
            **batch, decode_codebook=True
        )
        pred_pose = pred_pose.detach().cpu()
        trg_pose = batch["src"].detach().cpu()
        length = batch["src_length"].detach().cpu()

        for i, (p, t, c, l) in enumerate(
            zip(pred_pose, trg_pose, codebook_pose, length)
        ):
            l = int(l.item() - 1)  # remove cls token
            p = p[:l]
            t = t[:l]
            c = c[:l]
            make_pose_video(
                poses=[c, p, t],
                names=[f"{phase}_Codebook", f"{phase}_pred", f"{phase}_trg"],
                video_name=f"{self.current_epoch}_{phase}_{i}.mp4",
                save_dir=self.plot_path,
                fps=self.fps,
                slow=2,
            )

    def plot_codebook(self):
        with torch.no_grad():
            plot_n = 25
            codebook = self.codebook.codebooks.data
            # get random codebook entries
            idx = random.sample(range(codebook.shape[0]), plot_n)
            codebook_pose = self.get_codebook_pose()
            pose = codebook_pose[idx]

            make_square_pose_video(
                poses=pose,
                names=[f"codebook_{i}" for i in idx],
                width=5,
                height=5,
                video_name=f"codebook_L{self.window_size}_ID:{idx}.mp4",
                save_dir=self.plot_path,
                fps=self.fps,
                slow=2,
                main_title="",
            )

    def get_codebook_pose(self):
        """
        Convert codebook tokens to pose.
        :return: poses as torch.Tensor
        """
        codebook_pose = []
        with torch.no_grad():
            codebook = self.codebook.codebooks.data
            batch_size = 128
            for i in tqdm(
                range(0, codebook.shape[0], batch_size), desc="Decoding codebook"
            ):
                idx = list(range(i, min(i + batch_size, codebook.shape[0])))
                embeddings = codebook[idx].unsqueeze(1)
                embeddings = embeddings.reshape(len(idx), self.window_size, -1)

                src_length = torch.tensor([self.window_size for _ in range(len(idx))])
                src_mask = (
                    create_transformer_mask(src_length).unsqueeze(1).to(torch.bool)
                )
                counter = torch.tensor(
                    [i / (self.window_size - 1) for i in range(self.window_size)]
                ).unsqueeze(-1)
                counter = counter.unsqueeze(0).repeat(len(idx), 1, 1)
                trg_embed = self.counter_input_layer(counter.to(self.device))
                src_mask = src_mask.to(self.device)
                decoder_output, _ = self.decode(
                    trg_embed, embeddings, src_mask, src_mask
                )
                pose = self.pose_output_layer(decoder_output)
                pose = pose.cpu()
                codebook_pose.extend(pose)

        return torch.stack(codebook_pose)
