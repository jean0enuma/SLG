import os
import torch
import shutil
import lightning as L

from pathlib import Path

from loggers import make_loggers
from plot import make_pose_video
from model_vq import VQ_Transformer
from callbacks import make_callbacks
from dataset_vq import CodebookDataModule
from model_translation import Transformer
from dataset_vq import make_batch as make_batch_vq
from dataset_translation import TranslationDataModule
from helpers import load_config, save_config, find_best_model, find_n_parameters


torch.set_float32_matmul_precision("medium")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def make_dataset_and_model(
    mode: str = "translate",
    split: str = "train",
    cfg: dict = None,
    save_path: str = None,
    loggers: dict = None,
    cuda: str = "cuda",
):
    if mode == "translate":
        # load the VQ config and settings
        vq_model_dir = Path(cfg["data"].get("vq_model_dir", None))
        vq_config = load_config(vq_model_dir / "config.yaml")

        dataset = TranslationDataModule(cfg["data"], vq_config=vq_config["data"])
        dataset.setup("train")
        dataset.setup("dev")
        dataset.setup("test")
        # save translation vocab
        dataset.save_vocab(save_dir=save_path)

        vq_model = VQ_Transformer(
            vq_config,
            train_batch_size=cfg["data"]["train_batch_size"],
            dev_batch_size=cfg["data"]["dev_batch_size"],
            dataset=None,
            input_size=dataset.train.pose_dim,
            model_dir=save_path,
            fps=dataset.train.fps,
            loggers=loggers,
        )

        checkpoint_path = find_best_model(str(vq_model_dir))
        checkpoint_path = (vq_model_dir / checkpoint_path).as_posix()
        checkpoint = torch.load(checkpoint_path, map_location=cuda)
        vq_model.load_state_dict(checkpoint["state_dict"], strict=True)
        vq_model = vq_model.to(cuda).eval()

        # use vq model to quantize the data
        with torch.no_grad():
            if split == "train":
                dataset.train.quantize_data(
                    vq_model, make_batch=make_batch_vq, batch_size=512, device=cuda
                ) if dataset.train is not None else None
            else:
                dataset.train.remove_pose()
            dataset.dev.quantize_data(
                vq_model, make_batch=make_batch_vq, batch_size=512, device=cuda
            ) if dataset.dev is not None else None
            dataset.test.quantize_data(
                vq_model, make_batch=make_batch_vq, batch_size=512, device=cuda
            ) if dataset.test is not None else None

        # get mapping from token to pose
        codebook_pose = vq_model.get_codebook_pose()

        # make the translation model
        ground_truth_text = {
            "dev": dataset.dev.get_text(),
            "test": dataset.test.get_text(),
        }

        model = Transformer(
            cfg,
            save_path=save_path,
            train_batch_size=cfg["data"]["train_batch_size"],
            dev_batch_size=cfg["data"]["dev_batch_size"],
            src_vocab=dataset.dev.text_vocab,
            output_size=dataset.dev.output_size,
            fps=dataset.train.fps,
            ground_truth_text=ground_truth_text,
            codebook_pose=codebook_pose,
        )

    elif mode == "vq":
        dataset = CodebookDataModule(cfg["data"], cuda=cuda, save_path=save_path)
        dataset.setup("train")
        dataset.setup("dev")
        dataset.setup("test")

        model = VQ_Transformer(
            cfg,
            train_batch_size=cfg["data"]["train_batch_size"],
            dev_batch_size=cfg["data"]["test_batch_size"],
            dataset=dataset,
            input_size=dataset.train.input_size,
            model_dir=save_path,
            fps=dataset.train.fps,
            loggers=loggers,
        )
    else:
        raise ValueError(f"Unknown mode, {mode}")

    return model, dataset


def train(cfg_file: str, mode: str):
    # Load config
    cfg_file = Path(cfg_file)
    cfg = load_config(cfg_file)
    save_path = Path(cfg["save_path"]) / cfg["name"]
    save_path.mkdir(parents=True, exist_ok=True)

    # Resume training if checkpoint exists
    if (save_path / "last.ckpt").exists():
        resume = (save_path / "last.ckpt").as_posix()
        cfg = load_config(save_path / "config.yaml")
    else:
        resume = None
        shutil.copy2(cfg_file, (save_path / "init_config.yaml").as_posix())

    # Make logger
    loggers, cfg = make_loggers(
        cfg=cfg, save_path=save_path, mode="train", train_type=mode
    )
    save_config(cfg, save_path / "config.yaml")

    # Set the seed
    L.seed_everything(cfg["training"].get("random_seed", 42))
    cuda = "cuda" if torch.cuda.is_available() else "cpu"

    # Make model
    model, dataset = make_dataset_and_model(
        mode=mode,
        cfg=cfg,
        save_path=save_path,
        loggers=loggers,
        split="train",
        cuda=cuda,
    )

    # log the number of parameters
    n_parameters = find_n_parameters(model)
    [logger.log_metrics({"n_parameters": n_parameters}) for logger in loggers.values()]

    # Make callbacks
    callbacks = make_callbacks(cfg, save_path)

    # Train
    trainer = L.Trainer(
        num_sanity_val_steps=0,
        benchmark=True,
        callbacks=callbacks,
        logger=list(loggers.values()),
        log_every_n_steps=cfg["training"].get("log_every_n_steps", 10),
        max_epochs=cfg["training"]["epochs"],
        precision="16-mixed" if cfg["training"]["fp16"] else 32,  # or try bf16
        enable_checkpointing=cfg["training"]["checkpoint"].get("save", True),
        check_val_every_n_epoch=cfg["training"].get("test_every_n_epochs", 1),
        gradient_clip_val=cfg["training"].get("gradient_clip_val", 0),
        gradient_clip_algorithm=cfg["training"].get("gradient_clip_algorithm", "norm"),
    )

    loggers["wb_logger"].watch(
        model
    ) if "wb_logger" in loggers.keys() else None  # log gradients and model topology
    trainer.fit(
        model,
        dataset.get_dataloader("train"),
        dataset.get_dataloader("dev"),
        ckpt_path=resume,
    )

    # Return cfg path to test the model
    del dataset, model, trainer
    cfg_file = save_path / "config.yaml"
    return cfg_file.as_posix()


def test(cfg_file: str, mode: str):
    # Load config
    cfg_file = Path(cfg_file)
    cfg = load_config(cfg_file)
    # load data from the save path
    if cfg_file.parent == cfg["save_path"]:
        save_path = Path(cfg["save_path"]) / cfg["name"]
    elif (cfg_file.parent / "last.ckpt").exists():
        save_path = cfg_file.parent  # load the model from the current cfg path
    else:
        raise ValueError("No model found in the current directory")

    cfg = load_config(save_path / "config.yaml")  # load the config from the save path

    model_path = find_best_model(save_path)
    model_path = (save_path / model_path).as_posix()
    cuda = "cuda" if torch.cuda.is_available() else "cpu"

    # Make logger
    loggers, cfg = make_loggers(cfg=cfg, save_path=save_path, mode="test")

    # Make model
    model, dataset = make_dataset_and_model(
        mode=mode,
        cfg=cfg,
        save_path=save_path,
        loggers=loggers,
        split="test",
        cuda=cuda,
    )

    plot_path = save_path / "test_plots"
    model.plot_path = plot_path
    model.plot_path.mkdir(parents=True, exist_ok=True)

    trainer = L.Trainer(
        max_epochs=cfg["training"]["epochs"],
        logger=list(loggers.values()),
        log_every_n_steps=cfg["training"].get("log_every_n_steps", 10),
        callbacks=None,
        precision=16 if cfg["training"]["fp16"] else 32,
        enable_checkpointing=cfg["training"]["checkpoint"].get("save", True),
        check_val_every_n_epoch=cfg["training"].get("test_every_n_epochs", 1),
        gradient_clip_val=cfg["training"].get("gradient_clip_val", 0),
        gradient_clip_algorithm=cfg["training"].get("gradient_clip_algorithm", "norm"),
    )

    # Test and Validate model
    model.keep_predictions = True
    if (save_path / "last.ckpt").exists() or model_path is not None:
        trainer.validate(
            model=model, ckpt_path=model_path, dataloaders=dataset.get_dataloader("dev")
        )
        trainer.test(
            model=model,
            ckpt_path=model_path,
            dataloaders=dataset.get_dataloader("test"),
        )

        if mode == "translate":
            gt_pose = {"dev": dataset.dev.get_pose(), "test": dataset.test.get_pose()}
            for split in ["test", "dev"]:
                model.codebook_pose.to(cuda)
                model.bt_model.to(cuda)
                pred_text, pred_pose = model.evaluate_pose(split, log=False)
                plot_speedup = 2
                for i in range(model.plot_n):
                    if pred_pose[i] is None:
                        continue
                    make_pose_video(
                        poses=[
                            pred_pose[i][::plot_speedup],
                            gt_pose[split][i][::plot_speedup],
                        ],
                        names=["Prediction", "Ground Truth"],
                        video_name=f"{i}_{split}_Pred_pose",
                        save_dir=model.plot_path,
                        fps=model.fps // plot_speedup,
                        overwrite=True,
                    )
