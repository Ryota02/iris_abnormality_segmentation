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

from src.synthetic_ablation_report import (
    create_synthetic_ablation_report,
)


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

    report_cfg = cfg[
        "synthetic_report"
    ]

    csv_path  = cfg["synthetic_ablation"]["summary_csv"]

    output_pdf = (
        report_cfg[
            "output_pdf"
        ]
    )

    create_synthetic_ablation_report(
        csv_path=csv_path,
        output_pdf=output_pdf,
        report_cfg=report_cfg,
    )


if __name__ == "__main__":
    main()