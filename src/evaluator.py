from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import IrisSegmentationDataset
from src.metrics import (
    calculate_dice,
    calculate_iou,
)
from src.model import build_model


def get_dataset_root(
    data_cfg,
):
    """
    Support both:
        data.root
        data.prepared_root
    """

    if "prepared_root" in data_cfg:
        return Path(
            data_cfg["prepared_root"]
        )

    if "root" in data_cfg:
        return Path(
            data_cfg["root"]
        )

    raise KeyError(
        "Neither 'prepared_root' nor "
        "'root' was found in data config."
    )


def get_available_categories(
    root,
    categories,
    split,
):
    """
    Use only categories whose image/mask
    directories exist for the requested split.
    """

    available = []

    for category in categories:

        image_dir = (
            root
            / category
            / split
            / "images"
        )

        mask_dir = (
            root
            / category
            / split
            / "masks"
        )

        if (
            image_dir.exists()
            and mask_dir.exists()
        ):
            available.append(
                category
            )

        else:
            print(
                f"[WARNING] "
                f"Skipping {category} "
                f"for split={split}: "
                f"directory not found."
            )

    return available


def load_checkpoint_model(
    cfg,
    device,
):
    """
    Load the best checkpoint produced
    by train_from_config().
    """

    model_cfg = cfg["model"]
    output_cfg = cfg["output"]

    checkpoint_path = (
        Path(
            output_cfg["dir"]
        )
        / output_cfg["checkpoint"]
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    print(
        f"Loading checkpoint: "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = build_model(
        model_cfg
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


def evaluate_model_from_config(
    cfg,
    split="test",
):
    """
    Evaluate the best checkpoint
    on validation or test data.

    Augmentation and oversampling are NOT
    applied to val/test.
    """

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    root = get_dataset_root(
        data_cfg
    )

    categories = data_cfg.get(
        "categories",
        [
            "Geometry",
            "Tissue",
            "Healthy",
        ],
    )

    available_categories = (
        get_available_categories(
            root=root,
            categories=categories,
            split=split,
        )
    )

    if not available_categories:

        raise RuntimeError(
            f"No categories available "
            f"for split='{split}'."
        )

    # ========================================================
    # Dataset
    #
    # augmentation_config=None:
    # val/test never receive augmentation
    # ========================================================

    dataset = IrisSegmentationDataset(
        root=root,
        split=split,
        image_size=int(
            train_cfg["image_size"]
        ),
        categories=
            available_categories,
        augmentation_config=None,
    )

    if len(dataset) == 0:

        raise RuntimeError(
            f"No images found "
            f"for split='{split}'."
        )

    loader = DataLoader(
        dataset,
        batch_size=int(
            train_cfg[
                "batch_size"
            ]
        ),
        shuffle=False,
        num_workers=int(
            train_cfg[
                "num_workers"
            ]
        ),
        pin_memory=
            torch.cuda.is_available(),
    )

    # ========================================================
    # Device + model
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, checkpoint = (
        load_checkpoint_model(
            cfg=cfg,
            device=device,
        )
    )

    threshold = float(
        train_cfg.get(
            "threshold",
            0.5,
        )
    )

    # ========================================================
    # Metrics
    # ========================================================

    all_dice = []
    all_iou = []

    category_dice = defaultdict(
        list
    )

    category_iou = defaultdict(
        list
    )

    category_count = defaultdict(
        int
    )

    model.eval()

    with torch.no_grad():

        for batch in loader:

            images = (
                batch["image"]
                .to(device)
            )

            masks = (
                batch["mask"]
                .to(device)
            )

            categories_batch = (
                batch["category"]
            )

            logits = model(
                images
            )

            probabilities = (
                torch.sigmoid(
                    logits
                )
            )

            predictions = (
                probabilities
                >= threshold
            ).float()

            batch_size = (
                images.size(0)
            )

            for i in range(
                batch_size
            ):

                prediction = (
                    predictions[
                        i,
                        0,
                    ]
                    .cpu()
                    .numpy()
                    .astype(
                        np.uint8
                    )
                )

                target = (
                    masks[
                        i,
                        0,
                    ]
                    .cpu()
                    .numpy()
                    .astype(
                        np.uint8
                    )
                )

                category = (
                    categories_batch[
                        i
                    ]
                )

                dice = calculate_dice(
                    prediction,
                    target,
                )

                iou = calculate_iou(
                    prediction,
                    target,
                )

                all_dice.append(
                    dice
                )

                all_iou.append(
                    iou
                )

                category_dice[
                    category
                ].append(
                    dice
                )

                category_iou[
                    category
                ].append(
                    iou
                )

                category_count[
                    category
                ] += 1

    # ========================================================
    # Results
    # ========================================================

    result = {
        "split":
            split,

        "checkpoint_epoch":
            checkpoint.get(
                "epoch",
                None,
            ),

        "dice":
            float(
                np.mean(
                    all_dice
                )
            ),

        "iou":
            float(
                np.mean(
                    all_iou
                )
            ),

        "num_images":
            len(all_dice),

        "categories":
            {},
    }

    for category in categories:

        if (
            category
            not in category_dice
        ):

            result[
                "categories"
            ][
                category
            ] = {
                "count": 0,
                "dice": None,
                "iou": None,
            }

            continue

        result[
            "categories"
        ][
            category
        ] = {
            "count":
                category_count[
                    category
                ],

            "dice":
                float(
                    np.mean(
                        category_dice[
                            category
                        ]
                    )
                ),

            "iou":
                float(
                    np.mean(
                        category_iou[
                            category
                        ]
                    )
                ),
        }

    # ========================================================
    # Print
    # ========================================================

    print(
        "\n"
        "========================================"
    )

    print(
        f"{split.upper()} EVALUATION"
    )

    print(
        "========================================"
    )

    print(
        f"Images : "
        f"{result['num_images']}"
    )

    print(
        f"Dice   : "
        f"{result['dice']:.4f}"
    )

    print(
        f"IoU    : "
        f"{result['iou']:.4f}"
    )

    print(
        "\nCategory-wise:"
    )

    for category in categories:

        values = (
            result[
                "categories"
            ][
                category
            ]
        )

        if values["count"] == 0:

            print(
                f"{category:<10} "
                f"n=0"
            )

            continue

        print(
            f"{category:<10} "
            f"n={values['count']:<3} "
            f"Dice="
            f"{values['dice']:.4f} "
            f"IoU="
            f"{values['iou']:.4f}"
        )

    return result