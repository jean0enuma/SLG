from pathlib import Path

from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)


def make_callbacks(cfg: dict, save_path: Path):
    callbacks = []

    # Define the ModelCheckpoint callback
    if cfg["training"]["checkpoint"].get("save", True):
        checkpoint_callback = ModelCheckpoint(
            monitor=cfg["training"]["checkpoint"][
                "follow_metric"
            ],  # Metric to monitor for saving the best model
            dirpath=save_path,  # Directory to save the checkpoints
            filename="model-{epoch:02d}-{val_loss:.2f}-{val_MSE:.2f}",  # Checkpoint filename format
            save_top_k=cfg["training"]["checkpoint"].get("keep_top_ckpts", 1),
            mode=cfg["training"]["checkpoint"].get("mode", 1),
            save_last=True,  # Save the latest checkpoint
        )
        callbacks.append(checkpoint_callback)

    # Define the Learning Rate Monitor callback
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    early_stopping_metric = CustomEarlyStopping(
        monitor=cfg["model"]["early_stopping"]["metric"],
        min_delta=0.00,
        patience=cfg["model"]["early_stopping"]["patience"],
        verbose=False,
        mode=cfg["model"]["early_stopping"]["mode"],
    )
    callbacks.append(early_stopping_metric)

    early_stopping_lr = CustomEarlyStopping(
        monitor="lr-Adam",
        min_delta=0.00,
        patience=cfg["training"]["epochs"],
        verbose=False,
        mode="min",
        stopping_threshold=cfg["model"]["scheduler"]["min_lr"] * 1.01,
    )
    callbacks.append(early_stopping_lr)

    return callbacks


class CustomEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pad = "*" * 15
        if self.mode == "min":
            self.stopping_message = (
                f"Early stopping due to {self.monitor} metric not decrease"
            )
        else:
            self.stopping_message = (
                f"Early stopping due to {self.monitor} metric not increase"
            )
        self.ealry_stopping_message = f"{pad} {self.stopping_message} {pad}"

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if (
            trainer.fast_dev_run
            or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
                logs
            )
        ):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self._log_info(trainer, self.stopping_message, self.log_rank_zero_only)
            trainer.logger.log_text(self.stopping_message)
            print(self.ealry_stopping_message)
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)
