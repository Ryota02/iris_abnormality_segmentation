import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.config import load_config

from src.prediction_report import (
    generate_prediction_report,
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

    data_cfg = cfg[
        "data"
    ]

    output_cfg = cfg[
        "output"
    ]

    report_cfg = cfg.get(
        "report",
        {},
    )

    split = report_cfg.get(
        "split",
        "test",
    )

    prediction_root = (
        Path(
            output_cfg[
                "prediction_dir"
            ]
        )
        / split
    )

    output_pdf = (
        report_cfg.get(
            "output_pdf",
            str(
                prediction_root.parent
                / (
                    f"{split}"
                    "_results.pdf"
                )
            ),
        )
    )

    categories = (
        data_cfg.get(
            "categories",
            [
                "Geometry",
                "Tissue",
                "Healthy",
            ],
        )
    )

    rows_per_page = int(
        report_cfg.get(
            "rows_per_page",
            4,
        )
    )

    generate_prediction_report(
        prediction_root=
            prediction_root,

        output_pdf=
            output_pdf,

        categories=
            categories,

        rows_per_page=
            rows_per_page,
    )


if __name__ == "__main__":
    main()