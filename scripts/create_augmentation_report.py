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

from src.augmentation_ablation_report import (
    create_augmentation_ablation_report,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "augmentation_ablation.yaml"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Config
    # ========================================================

    cfg = load_config(
        args.config
    )

    report_cfg = cfg[
        "augmentation_report"
    ]

    # ========================================================
    # Create report
    # ========================================================

    create_augmentation_ablation_report(
        summary_csv=
            report_cfg[
                "summary_csv"
            ],

        output_pdf=
            report_cfg[
                "output_pdf"
            ],

        model_name=
            report_cfg.get(
                "model_name",
                "ResNet34 U-Net",
            ),
    )


if __name__ == "__main__":
    main()