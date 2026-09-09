import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.config import load_config
from src.predictor import run_prediction


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/unet_resnet34.yaml",
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=[
            "train",
            "val",
            "test",
        ],
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    run_prediction(
        cfg=cfg,
        split=args.split,
    )


if __name__ == "__main__":
    main()