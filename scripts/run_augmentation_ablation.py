import argparse
import sys
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.config import (
    load_config,
)

from src.augmentation_ablation import (
    run_augmentation_ablation,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    run_augmentation_ablation(
        cfg
    )


if __name__ == "__main__":
    main()