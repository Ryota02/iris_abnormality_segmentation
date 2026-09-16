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

from src.prediction_report import (
    collect_prediction_samples,
    create_prediction_pdf,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    args = parser.parse_args()

    # ========================================================
    # Config
    # ========================================================

    cfg = load_config(
        args.config
    )

    split = args.split

    # ========================================================
    # Synthetic ablation
    # ========================================================

    ablation_cfg = cfg[
        "synthetic_ablation"
    ]

    experiments = (
        ablation_cfg[
            "experiments"
        ]
    )

    output_root = Path(
        ablation_cfg[
            "output_root"
        ]
    )

    # ========================================================
    # Prediction report config
    # ========================================================

    report_cfg = cfg.get(
        "prediction_report",
        {},
    )

    rows_per_page = int(
        report_cfg.get(
            "rows_per_page",
            4,
        )
    )

    categories = report_cfg.get(
        "categories",
        cfg[
            "data"
        ][
            "categories"
        ],
    )

    selected_experiments = (
        report_cfg.get(
            "experiments",
            [],
        )
    )

    output_pdf = Path(
        report_cfg.get(
            "output_pdf",
            (
                output_root
                / f"synthetic_prediction_report_{split}.pdf"
            ),
        )
    )

    # ========================================================
    # Collect ALL experiments
    # ========================================================

    all_samples = []

    for experiment in experiments:

        experiment_name = (
            experiment[
                "name"
            ]
        )

        # ----------------------------------------------------
        # Optional experiment filter
        # ----------------------------------------------------

        if (
            selected_experiments
            and
            experiment_name
            not in selected_experiments
        ):
            continue

        prediction_root = (
            output_root
            / experiment_name
            / "predictions"
            / split
        )

        print(
            "\n"
            "========================================"
        )

        print(
            f"Experiment: "
            f"{experiment_name}"
        )

        print(
            f"Prediction root: "
            f"{prediction_root}"
        )

        if not prediction_root.exists():

            print(
                "[WARNING] "
                "Prediction directory "
                "not found."
            )

            print(
                f"Skip: "
                f"{experiment_name}"
            )

            continue

        samples = (
            collect_prediction_samples(
                prediction_root=
                    prediction_root,

                categories=
                    categories,

                experiment_name=
                    experiment_name,
            )
        )

        print(
            f"Collected samples: "
            f"{len(samples)}"
        )

        all_samples.extend(
            samples
        )

    # ========================================================
    # Check
    # ========================================================

    if len(all_samples) == 0:

        raise RuntimeError(
            "No prediction samples "
            "were found."
        )

    print(
        "\n"
        "========================================"
    )

    print(
        "CREATE COMBINED PDF"
    )

    print(
        "========================================"
    )

    print(
        f"Total samples: "
        f"{len(all_samples)}"
    )

    print(
        f"Output PDF: "
        f"{output_pdf}"
    )

    # ========================================================
    # One PDF
    # ========================================================

    create_prediction_pdf(
        samples=
            all_samples,

        output_pdf=
            output_pdf,

        rows_per_page=
            rows_per_page,
    )


if __name__ == "__main__":
    main()