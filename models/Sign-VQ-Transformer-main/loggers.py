import logging

from pathlib import Path
from typing import Dict, Optional

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import Logger


def make_loggers(
    cfg: dict,
    save_path: Optional[Path] = None,
    mode: str = "train",
    train_type: str = "translate",
):
    loggers = {}
    if cfg.get("local_logger", False):
        local_logger = LocalLogger(
            log_dir=save_path, level=cfg.get("log_level", "debug"), mode=mode
        )
        local_logger.log_config(cfg)
        loggers.update({"local_logger": local_logger})

    # wandb logger
    if cfg.get("wandb_logger", False):
        id = cfg.get("wandb_logger_id", None)
        if train_type == "translate":
            project = "Translation SignVqTransformer"
        elif train_type == "vq":
            project = "VQ SignVqTransformer"
        wb_logger = WandbLogger(
            project=project,
            name=cfg.get("name", "Sign_VQ_Transformer"),
            id=id,
            config=cfg,
            log_model=False,
            save_dir=save_path.as_posix(),
            resume="allow",
        )
        if id is None:
            id = wb_logger.experiment.id
            cfg.update({"wandb_logger_id": id})
            wb_logger.experiment.config.update(cfg)
        # add config to logger
        loggers.update({"wb_logger": wb_logger})

    return loggers, cfg


class LocalLogger(Logger):
    def __init__(
        self, level: str = "info", log_dir: Optional[Path] = None, mode: str = "train"
    ):

        super().__init__()
        self.logger_name = "local_logger"
        self.logger = logging.getLogger("lightning.pytorch")
        while self.logger.hasHandlers():
            self.logger.removeHandler(self.logger.handlers[0])

        if level.lower() == "debug":
            self.logger.setLevel(level=logging.DEBUG)
        elif level.lower() == "warning":
            self.logger.setLevel(level=logging.WARNING)
        else:
            self.logger.setLevel(level=logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Add Console Handler (New!)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if log_dir is not None:
            if log_dir.is_dir():
                log_file = log_dir / f"{mode}.log"

                fh = logging.FileHandler(log_file.as_posix())
                fh.setLevel(level=logging.DEBUG)
                self.logger.addHandler(fh)
                fh.setFormatter(formatter)

        self.logger.info("Sign Level VQVAE Ready!!!")

    def log_config(self, cfg: Dict, prefix: str = "cfg") -> None:
        """
        Write configuration to log.

        :param cfg: configuration to log
        :param prefix: prefix for logging
        """
        for k, v in cfg.items():
            if isinstance(v, dict):
                p = ".".join([prefix, k])
                self.log_config(v, prefix=p)
            else:
                p = ".".join([prefix, k])
                self.logger.info("%34s : %s", p, v)

    def log_metrics(self, metrics, step=None):
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value}")

    def log_hyperparams(self, params):
        self.logger.info("Hyperparameters:")
        for key, value in params.items():
            self.logger.info(f"{key}: {value}")

    def log_text(self, text, step=None):
        self.logger.info(text)

    def save(self):
        # Optional: Implement saving logic if needed
        pass

    @classmethod
    def load(cls, version, tags=None):
        # Optional: Implement loading logic if needed
        pass

    @property
    def name(self) -> str:
        return self.logger_name

    @property
    def version(self) -> str:
        # Optional: Implement version logic if needed
        return "1.0"
