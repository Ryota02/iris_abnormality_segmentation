import argparse
import copy
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

from src.predictor import (
    run_prediction,
)

from src.synthetic_ablation import (
    build_experiment_config,
)


def find_experiment(
    cfg,
    experiment_name,
):
    """
    Find one experiment definition
    from synthetic_ablation.experiments.
    """

    experiments = (
        cfg[
            "synthetic_ablation"
        ][
            "experiments"
        ]
    )

    for experiment in experiments:

        if (
            experiment[
                "name"
            ]
            == experiment_name
        ):
            return experiment

    available = [
        experiment[
            "name"
        ]
        for experiment
        in experiments
    ]

    raise ValueError(
        f"Experiment not found: "
        f"{experiment_name}\n"
        f"Available experiments: "
        f"{available}"
    )


def run_one_experiment(
    base_cfg,
    experiment,
    split,
):
    """
    Rebuild the same config that was used
    during synthetic ablation training,
    then run prediction.
    """

    (
        cfg,
        experiment_name,
    ) = build_experiment_config(
        base_cfg=
            base_cfg,

        experiment=
            experiment,
    )

    checkpoint_path = (
        Path(
            cfg[
                "output"
            ][
                "dir"
            ]
        )
        /
        cfg[
            "output"
        ].get(
            "checkpoint",
            "best_model.pt",
        )
    )

    print(
        "\n"
        "========================================"
    )

    print(
        f"Experiment : "
        f"{experiment_name}"
    )

    print(
        f"Split      : "
        f"{split}"
    )

    print(
        f"Checkpoint : "
        f"{checkpoint_path}"
    )

    print(
        "========================================"
    )

    if not checkpoint_path.exists():

        print(
            "[WARNING] "
            "Checkpoint not found."
        )

        print(
            f"Skip: "
            f"{experiment_name}"
        )

        return

    # --------------------------------------------------------
    # run_prediction should use:
    #
    # cfg["output"]["dir"]
    #
    # Therefore prediction results will be saved under
    # each experiment directory.
    # --------------------------------------------------------

    run_prediction(
        cfg=cfg,
        split=split,
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

    parser.add_argument(
        "--experiment",
        default=None,
        help=(
            "Synthetic ablation "
            "experiment name."
        ),
    )

    parser.add_argument(
        "--all-experiments",
        action="store_true",
        help=(
            "Run prediction for every "
            "synthetic ablation experiment."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Config
    # ========================================================

    base_cfg = load_config(
        args.config
    )

    # ========================================================
    # All synthetic experiments
    # ========================================================

    if args.all_experiments:

        if (
            "synthetic_ablation"
            not in base_cfg
        ):

            raise KeyError(
                "--all-experiments was used, "
                "but synthetic_ablation is "
                "not defined in the config."
            )

        experiments = (
            base_cfg[
                "synthetic_ablation"
            ][
                "experiments"
            ]
        )

        print(
            "\n"
            "Synthetic ablation predictions"
        )

        print(
            f"Experiments: "
            f"{len(experiments)}"
        )

        for experiment in experiments:

            run_one_experiment(
                base_cfg=
                    base_cfg,

                experiment=
                    experiment,

                split=
                    args.split,
            )

        return

    # ========================================================
    # One synthetic experiment
    # ========================================================

    if args.experiment is not None:

        if (
            "synthetic_ablation"
            not in base_cfg
        ):

            raise KeyError(
                "--experiment was used, "
                "but synthetic_ablation is "
                "not defined in the config."
            )

        experiment = find_experiment(
            cfg=base_cfg,
            experiment_name=
                args.experiment,
        )

        run_one_experiment(
            base_cfg=
                base_cfg,

            experiment=
                experiment,

            split=
                args.split,
        )

        return

    # ========================================================
    # Normal prediction
    #
    # Existing non-ablation behavior
    # ========================================================

    cfg = copy.deepcopy(
        base_cfg
    )

    run_prediction(
        cfg=cfg,
        split=args.split,
    )


if __name__ == "__main__":
    main()