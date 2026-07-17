import os
import argparse

from train import train, test


def main():
    os.environ["WANDB_DIR"] = "/tmp"
    ap = argparse.ArgumentParser("Sign Level VQVAE!!")

    ap.add_argument("mode", choices=["train", "test"], help="train or test a model")
    ap.add_argument("type", choices=["translate", "vq"], help=" translation or VQ")
    ap.add_argument(
        "config_path", metavar="config-path", type=str, help="path to YAML config file"
    )
    # VQ model
    ap.add_argument(
        "--var-vq_path", type=str, help="path to the VQ model for translation"
    )
    args = ap.parse_args()

    if args.mode == "train":
        config_path = train(cfg_file=args.config_path, mode=args.type)
        test(cfg_file=config_path, mode=args.type)
    elif args.mode == "test":
        test(cfg_file=args.config_path, mode=args.type)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
