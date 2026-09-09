import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.config import load_config
from src.trainer import train_from_config


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "unet_resnet34.yaml"
        ),
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    train_from_config(
        cfg
    )


if __name__ == "__main__":
    main()