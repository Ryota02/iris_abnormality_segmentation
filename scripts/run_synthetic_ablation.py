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

from src.synthetic_ablation import (
    run_synthetic_ablation,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "synthetic_ablation.yaml"
        ),
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    run_synthetic_ablation(
        cfg
    )


if __name__ == "__main__":
    main()