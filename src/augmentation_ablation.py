import copy
import csv
from pathlib import Path

from src.evaluator import (
    evaluate_model_from_config,
)

from src.trainer import (
    train_from_config,
)


AUGMENTATION_METHODS = [
    "horizontal_flip",
    "rotation",
    "brightness",
    "contrast",
    "gamma",
    "gaussian_noise",
    "gaussian_blur",
]

def enable_one_augmentation(
    cfg,
    method,
):
    """
    Enable exactly one augmentation method.
    """

    if method not in AUGMENTATION_METHODS:

        raise ValueError(
            f"Unknown augmentation method: "
            f"{method}"
        )

    cfg[
        "augmentation"
    ][
        "enabled"
    ] = True

    categories = cfg[
        "data"
    ][
        "categories"
    ]

    for category in categories:

        category_cfg = (
            cfg[
                "augmentation"
            ].get(
                category
            )
        )

        if category_cfg is None:
            continue

        if method not in category_cfg:

            raise KeyError(
                f"Augmentation "
                f"'{method}' "
                f"not found for category "
                f"'{category}'."
            )

        category_cfg[
            method
        ][
            "enabled"
        ] = True

def build_experiment_config(
    base_cfg,
    method,
):
    """
    Build config for one ablation experiment.

    Exactly one augmentation method is enabled
    for augmentation experiments.
    """

    cfg = copy.deepcopy(
        base_cfg
    )

    # ========================================================
    # First:
    # disable ALL augmentation
    # disable oversampling
    # ========================================================

    disable_all_augmentations(
        cfg
    )

    cfg[
        "sampling"
    ][
        "enabled"
    ] = False

    # ========================================================
    # Baseline
    # ========================================================

    if method == "baseline":

        pass

    # ========================================================
    # Oversampling only
    # ========================================================

    elif method == "oversampling":

        cfg[
            "sampling"
        ][
            "enabled"
        ] = True

    # ========================================================
    # Augmentation only
    # ========================================================

    elif method in AUGMENTATION_METHODS:

        enable_one_augmentation(
            cfg,
            method,
        )

    # ========================================================
    # Augmentation + Oversampling
    # ========================================================

    elif method.endswith(
        "_oversampling"
    ):

        augmentation_method = (
            method[
                :-len(
                    "_oversampling"
                )
            ]
        )

        if (
            augmentation_method
            not in AUGMENTATION_METHODS
        ):

            raise ValueError(
                f"Unknown method: {method}"
            )

        enable_one_augmentation(
            cfg,
            augmentation_method,
        )

        cfg[
            "sampling"
        ][
            "enabled"
        ] = True

    else:

        raise ValueError(
            f"Unknown ablation method: "
            f"{method}"
        )

    # ========================================================
    # Output
    # ========================================================

    output_root = Path(
        "outputs/"
        "augmentation_ablation"
    )

    experiment_dir = (
        output_root
        / method
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

    return cfg

def disable_all_augmentations(
    cfg,
):
    """
    Disable all augmentation methods.

    Augmentation parameters such as
    probability, limit, sigma are preserved.
    """

    augmentation_cfg = cfg[
        "augmentation"
    ]

    # Master switch
    augmentation_cfg[
        "enabled"
    ] = False

    categories = cfg[
        "data"
    ][
        "categories"
    ]

    for category in categories:

        if category not in augmentation_cfg:
            continue

        for method in AUGMENTATION_METHODS:

            if (
                method
                not in augmentation_cfg[
                    category
                ]
            ):
                continue

            augmentation_cfg[
                category
            ][
                method
            ][
                "enabled"
            ] = False


def build_ablation_config(
    base_cfg,
    method,
):
    """
    Build config for one ablation experiment.

    Supported:
        baseline

        horizontal_flip
        rotation
        brightness
        contrast
        gamma
        gaussian_noise
        gaussian_blur

        oversampling

        horizontal_flip_oversampling
        rotation_oversampling
        brightness_oversampling
        contrast_oversampling
        gamma_oversampling
        gaussian_noise_oversampling
        gaussian_blur_oversampling
    """

    import copy
    from pathlib import Path

    cfg = copy.deepcopy(
        base_cfg
    )

    # ========================================================
    # Reset
    #
    # Every experiment starts from:
    #
    # augmentation = OFF
    # oversampling = OFF
    # ========================================================

    disable_all_augmentations(
        cfg
    )

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
    # 1. Baseline
    # ========================================================

    if method == "baseline":

        pass

    # ========================================================
    # 2. Oversampling only
    #
    # IMPORTANT:
    # Must be checked BEFORE treating method
    # as an augmentation.
    # ========================================================

    elif method == "oversampling":

        cfg[
            "sampling"
        ][
            "enabled"
        ] = True

    # ========================================================
    # 3. Augmentation only
    # ========================================================

    elif method in AUGMENTATION_METHODS:

        enable_one_augmentation(
            cfg,
            method,
        )

    # ========================================================
    # 4. One Augmentation + Oversampling
    # ========================================================

    elif method.endswith(
        "_oversampling"
    ):

        augmentation_method = (
            method[
                :-len(
                    "_oversampling"
                )
            ]
        )

        if (
            augmentation_method
            not in AUGMENTATION_METHODS
        ):

            raise ValueError(
                f"Unknown augmentation "
                f"with oversampling: "
                f"{augmentation_method}"
            )

        # Exactly one augmentation
        enable_one_augmentation(
            cfg,
            augmentation_method,
        )

        # Oversampling ON
        cfg[
            "sampling"
        ][
            "enabled"
        ] = True

    # ========================================================
    # Unknown
    # ========================================================

    else:

        raise ValueError(
            f"Unknown ablation method: "
            f"{method}"
        )

    # ========================================================
    # Output
    # ========================================================

    output_root = Path(
        "outputs/"
        "augmentation_ablation"
    )

    experiment_dir = (
        output_root
        / method
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

    return cfg


def set_experiment_output(
    cfg,
    method,
):
    """
    Each augmentation gets its own
    output directory.
    """

    output_root = Path(
        cfg[
            "output"
        ][
            "dir"
        ]
    )

    experiment_dir = (
        output_root
        / method
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

    return experiment_dir


def safe_category_metric(
    evaluation_result,
    category,
    metric,
):
    """
    Return category metric safely.

    If category has no samples,
    return empty string for CSV.
    """

    values = (
        evaluation_result
        .get(
            "categories",
            {},
        )
        .get(
            category,
            {},
        )
    )

    value = values.get(
        metric,
        None,
    )

    if value is None:
        return ""

    return value


def save_summary(
    results,
    output_path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "experiment",

        "best_epoch",

        "best_val_dice",
        "best_val_iou",

        "test_dice",
        "test_iou",
        "test_num_images",

        "geometry_test_dice",
        "geometry_test_iou",

        "tissue_test_dice",
        "tissue_test_iou",

        "healthy_test_dice",
        "healthy_test_iou",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                result
            )


def run_augmentation_ablation(
    base_cfg,
):
    """
    For each augmentation:

    1. Train on train split
    2. Select checkpoint using validation Dice
    3. Evaluate best checkpoint on test split
    4. Save Val + Test metrics to CSV
    """

    ablation_cfg = (
        base_cfg[
            "ablation"
        ]
    )

    output_dir = Path(base_cfg["output"]["dir"])

    methods = (
        ablation_cfg[
            "methods"
        ]
    )

    results = []

    total_experiments = len(
        methods
    )

    # ========================================================
    # Run experiments
    # ========================================================

    for experiment_index, method in enumerate(
        methods,
        start=1,
    ):

        print(
            "\n"
            "========================================"
        )

        print(
            f"Experiment "
            f"{experiment_index}/"
            f"{total_experiments}"
        )

        print(
            f"Augmentation: "
            f"{method}"
        )

        print(
            "========================================"
        )

        # ====================================================
        # Config
        # ====================================================

        cfg = build_ablation_config(
            base_cfg=base_cfg,
            method=method,
        )

        experiment_dir = (
            set_experiment_output(
                cfg=cfg,
                method=method,
            )
        )

        print(
            f"Output directory: "
            f"{experiment_dir}"
        )

        # ====================================================
        # 1. TRAIN
        # ====================================================

        print(
            "\n"
            "---------- TRAINING ----------"
        )

        training_result = (
            train_from_config(
                cfg
            )
        )

        # ====================================================
        # 2. TEST
        #
        # Best validation checkpoint is loaded automatically.
        # ====================================================

        print(
            "\n"
            "---------- TESTING -----------"
        )

        test_result = (
            evaluate_model_from_config(
                cfg=cfg,
                split="test",
            )
        )

        # ====================================================
        # 3. Result
        # ====================================================

        result = {
            "experiment":
                method,

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

            "test_num_images":
                test_result[
                    "num_images"
                ],

            "geometry_test_dice":
                safe_category_metric(
                    test_result,
                    "Geometry",
                    "dice",
                ),

            "geometry_test_iou":
                safe_category_metric(
                    test_result,
                    "Geometry",
                    "iou",
                ),

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

        # ====================================================
        # Save immediately
        # ====================================================

        summary_csv_path = output_dir / ablation_cfg["summary_csv"]
        save_summary(
            results=results,
            output_path= summary_csv_path)

        # ====================================================
        # Print experiment result
        # ====================================================

        print(
            "\n"
            "========================================"
        )

        print(
            f"RESULT: {method}"
        )

        print(
            "========================================"
        )

        print(
            f"Best Epoch    : "
            f"{result['best_epoch']}"
        )

        print(
            f"Best Val Dice : "
            f"{result['best_val_dice']:.4f}"
        )

        print(
            f"Best Val IoU  : "
            f"{result['best_val_iou']:.4f}"
        )

        print(
            f"Test Dice     : "
            f"{result['test_dice']:.4f}"
        )

        print(
            f"Test IoU      : "
            f"{result['test_iou']:.4f}"
        )

        # ====================================================
        # Category metrics
        # ====================================================

        print(
            "\nTest category-wise:"
        )

        for category in [
            "Geometry",
            "Tissue",
            "Healthy",
        ]:

            values = (
                test_result[
                    "categories"
                ].get(
                    category,
                    {},
                )
            )

            count = values.get(
                "count",
                0,
            )

            if count == 0:

                print(
                    f"{category:<10} "
                    f"n=0"
                )

                continue

            print(
                f"{category:<10} "
                f"n={count:<3} "
                f"Dice="
                f"{values['dice']:.4f} "
                f"IoU="
                f"{values['iou']:.4f}"
            )

    # ========================================================
    # Final summary
    # ========================================================

    print(
        "\n\n"
        "========================================"
    )

    print(
        "AUGMENTATION ABLATION RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"{'Experiment':<20} "
        f"{'Val Dice':>10} "
        f"{'Test Dice':>10} "
        f"{'Test IoU':>10}"
    )

    print(
        "-" * 55
    )

    for result in results:

        print(
            f"{result['experiment']:<20} "
            f"{result['best_val_dice']:>10.4f} "
            f"{result['test_dice']:>10.4f} "
            f"{result['test_iou']:>10.4f}"
        )

    print("\nSummary CSV:")
    print(summary_csv_path)

    return results