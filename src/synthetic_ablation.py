import copy
import csv
from pathlib import Path

from src.evaluator import (
    evaluate_model_from_config,
)

from src.trainer import (
    train_from_config,
)


def get_data_root(
    cfg,
):

    data_cfg = cfg[
        "data"
    ]

    if "prepared_root" in data_cfg:

        return Path(
            data_cfg[
                "prepared_root"
            ]
        )

    return Path(
        data_cfg[
            "root"
        ]
    )


def count_real_train_images(
    cfg,
    category,
):

    root = get_data_root(
        cfg
    )

    image_dir = (
        root
        / category
        / "train"
        / "images"
    )

    if not image_dir.exists():

        return 0

    extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    return len([
        path
        for path
        in image_dir.iterdir()

        if (
            path.is_file()
            and
            path.suffix.lower()
            in extensions
        )
    ])


def build_experiment_config(
    base_cfg,
    experiment,
):
    """
    Build config for one synthetic-data experiment.

    experiment example:

        {
            "name": "synthetic_only_32",
            "synthetic_count": 32,
            "include_real_tissue": False,
        }
    """

    cfg = copy.deepcopy(
        base_cfg
    )

    experiment_name = (
        experiment[
            "name"
        ]
    )

    synthetic_count = int(
        experiment[
            "synthetic_count"
        ]
    )

    include_real_tissue = bool(
        experiment.get(
            "include_real_tissue",
            True,
        )
    )

    # ========================================================
    # Synthetic configuration
    # ========================================================

    cfg[
        "synthetic"
    ][
        "use_num"
    ] = synthetic_count

    cfg[
        "synthetic"
    ][
        "include_real_tissue"
    ] = include_real_tissue

    if synthetic_count > 0:

        cfg[
            "synthetic"
        ][
            "enabled"
        ] = True

    else:

        cfg[
            "synthetic"
        ][
            "enabled"
        ] = False

    # ========================================================
    # Oversampling OFF
    # ========================================================

    cfg.setdefault(
        "sampling",
        {},
    )

    cfg[
        "sampling"
    ][
        "enabled"
    ] = False

    # ========================================================
    # Output directory
    # ========================================================

    output_root = Path(
        cfg[
            "synthetic_ablation"
        ][
            "output_root"
        ]
    )

    experiment_dir = (
        output_root
        / experiment_name
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg[
        "output"
    ][
        "dir"
    ] = str(
        experiment_dir
    )

    return (
        cfg,
        experiment_name,
    )

def safe_category_metric(
    result,
    category,
    metric,
):

    value = (
        result
        .get(
            "categories",
            {},
        )
        .get(
            category,
            {},
        )
        .get(
            metric,
            None,
        )
    )

    if value is None:
        return ""

    return value


def save_summary(
    results,
    path,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "experiment",
    
        "include_real_tissue",
    
        "real_tissue",
        "synthetic_tissue",
        "total_tissue",
        "real_healthy",
    
        "best_epoch",
    
        "best_val_dice",
        "best_val_iou",
    
        "test_dice",
        "test_iou",
    
        "tissue_test_dice",
        "tissue_test_iou",
    
        "healthy_test_dice",
        "healthy_test_iou",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


def run_synthetic_ablation(
    base_cfg,
):

    ablation_cfg = (
        base_cfg[
            "synthetic_ablation"
        ]
    )
    
    experiments = (ablation_cfg["experiments"])

    real_tissue = (count_real_train_images(base_cfg,"Tissue",))

    real_healthy = (count_real_train_images(base_cfg,"Healthy",))

    print(
        "\n"
        "========================================"
    )

    print(
        "REAL TRAIN DATA"
    )

    print(
        "========================================"
    )

    print(
        f"Tissue  : "
        f"{real_tissue}"
    )

    print(
        f"Healthy : "
        f"{real_healthy}"
    )

    results = []

    # ========================================================
    # Experiments
    # ========================================================

    for index, experiment in enumerate(
        experiments,
        start=1,
    ):
        
        (cfg, experiment_name) = build_experiment_config(
            base_cfg=base_cfg,
            experiment=experiment,
        )

        synthetic_count = int(
            experiment["synthetic_count"]
        )
        
        include_real_tissue = bool(
            experiment.get("include_real_tissue",True)
        )

        used_real_tissue = (
            real_tissue
            if include_real_tissue
            else 0
        )
        
        total_tissue = (
            used_real_tissue
            + synthetic_count
        )
        

        print(
            "\n\n"
            "========================================"
        )

        print(
            f"{experiment_name}"
        )

        print(
            "========================================"
        )

        print(
            f"Real Tissue      : "
            f"{used_real_tissue}"
        )
        
        print(
            f"Synthetic Tissue : "
            f"{synthetic_count}"
        )
        
        print(
            f"Total Tissue     : "
            f"{total_tissue}"
        )
        
        print(
            f"Healthy          : "
            f"{real_healthy}"
        )

        # ====================================================
        # TRAIN
        # ====================================================

        print(
            "\n---------- TRAIN ----------"
        )

        training_result = (
            train_from_config(
                cfg
            )
        )

        # ====================================================
        # TEST
        #
        # Best validation checkpoint
        # ====================================================

        print(
            "\n---------- TEST -----------"
        )

        test_result = (
            evaluate_model_from_config(
                cfg=cfg,
                split="test",
            )
        )

        # ====================================================
        # Result
        # ====================================================

        result = {
            "experiment":
                experiment_name,
        
            "include_real_tissue":
                include_real_tissue,
        
            "real_tissue":
                used_real_tissue,
        
            "synthetic_tissue":
                synthetic_count,
        
            "total_tissue":
                total_tissue,
        
            "real_healthy":
                real_healthy,
        
            "best_epoch":
                training_result[
                    "best_epoch"
                ],
        
            "best_val_dice":
                training_result[
                    "best_val_dice"
                ],
        
            "best_val_iou":
                training_result[
                    "best_val_iou"
                ],
        
            "test_dice":
                test_result[
                    "dice"
                ],
        
            "test_iou":
                test_result[
                    "iou"
                ],
        
            "tissue_test_dice":
                safe_category_metric(
                    test_result,
                    "Tissue",
                    "dice",
                ),
        
            "tissue_test_iou":
                safe_category_metric(
                    test_result,
                    "Tissue",
                    "iou",
                ),
        
            "healthy_test_dice":
                safe_category_metric(
                    test_result,
                    "Healthy",
                    "dice",
                ),
        
            "healthy_test_iou":
                safe_category_metric(
                    test_result,
                    "Healthy",
                    "iou",
                ),
        }

        results.append(
            result
        )

        # Save after each experiment
        save_summary(
            results,
            ablation_cfg[
                "summary_csv"
            ],
        )

        print(
            "\nResult"
        )

        print(
            f"Best Val Dice : "
            f"{result['best_val_dice']:.4f}"
        )

        print(
            f"Test Dice     : "
            f"{result['test_dice']:.4f}"
        )

        print(
            f"Test IoU      : "
            f"{result['test_iou']:.4f}"
        )

    # ========================================================
    # Final table
    # ========================================================

    print(
        "\n\n"
        "============================================================"
    )

    print(
        "SYNTHETIC ABLATION RESULTS"
    )

    print(
        "============================================================"
    )

    print(
        f"{'Experiment':<18}"
        f"{'Tissue':>10}"
        f"{'Healthy':>10}"
        f"{'Val Dice':>12}"
        f"{'Test Dice':>12}"
        f"{'Test IoU':>12}"
    )

    print(
        "-" * 74
    )

    for result in results:

        print(
            f"{result['experiment']:<18}"
            f"{result['total_tissue']:>10}"
            f"{result['real_healthy']:>10}"
            f"{result['best_val_dice']:>12.4f}"
            f"{result['test_dice']:>12.4f}"
            f"{result['test_iou']:>12.4f}"
        )

    print(
        "\nSummary CSV:"
    )

    print(
        ablation_cfg[
            "summary_csv"
        ]
    )

    return results