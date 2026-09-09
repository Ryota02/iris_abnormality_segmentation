import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.dataset import IrisSegmentationDataset
from src.metrics import (
    calculate_dice,
    calculate_iou,
    calculate_pixel_metrics,
)
from src.model import build_model


def nanmean(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        return np.nan

    if np.all(
        np.isnan(values)
    ):
        return np.nan

    return float(
        np.nanmean(values)
    )


def save_csv(
    records,
    output_path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not records:
        return

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                records[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            records
        )


def print_summary(records):
    grouped = defaultdict(list)

    for record in records:
        grouped[
            record["category"]
        ].append(record)

    print(
        "\n============================="
    )
    print(
        "Test results"
    )
    print(
        "============================="
    )

    metric_names = [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "pixel_accuracy",
        "false_positive_rate",
        "predicted_positive_fraction",
    ]

    for name, group in [
        ("Overall", records),
        *[
            (category, grouped[category])
            for category in [
                "Geometry",
                "Tissue",
                "Healthy",
            ]
        ],
    ]:
        if len(group) == 0:
            continue

        print(
            f"\n{name} "
            f"(n={len(group)})"
        )

        for metric_name in metric_names:
            value = nanmean([
                row[metric_name]
                for row in group
            ])

            print(
                f"  {metric_name:<27}: "
                f"{value:.4f}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/unet.yaml",
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    output_cfg = cfg["output"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint_path = (
        Path(output_cfg["dir"])
        / output_cfg["checkpoint"]
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

    categories = data_cfg["categories"]
    test_dataset = (
        IrisSegmentationDataset(
            root=data_cfg[
                "root"
            ],
            split="test",
            image_size=int(
                train_cfg["image_size"]
            ),
            categories=categories, 
            augmentation_config=None,
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(
            train_cfg["batch_size"]
        ),
        shuffle=False,
        num_workers=int(
            train_cfg["num_workers"]
        ),
        pin_memory=torch.cuda.is_available(),
    )

    threshold = float(
        train_cfg["threshold"]
    )

    records = []

    with torch.no_grad():
        for batch in tqdm(
            test_loader,
            desc="Evaluating",
        ):
            images = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            masks = batch[
                "mask"
            ].to(
                device,
                non_blocking=True,
            )

            logits = model(
                images
            )

            probs = torch.sigmoid(
                logits
            )

            preds = (
                probs >= threshold
            ).float()

            batch_size = (
                images.size(0)
            )

            for i in range(
                batch_size
            ):
                pred = (
                    preds[i, 0]
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                target = (
                    masks[i, 0]
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                metrics = (
                    calculate_pixel_metrics(
                        pred,
                        target,
                    )
                )

                record = {
                    "filename":
                        batch[
                            "filename"
                        ][i],

                    "category":
                        batch[
                            "category"
                        ][i],

                    "dice":
                        calculate_dice(
                            pred,
                            target,
                        ),

                    "iou":
                        calculate_iou(
                            pred,
                            target,
                        ),

                    **metrics,

                    "predicted_positive_fraction":
                        float(
                            pred.mean()
                        ),
                }

                records.append(
                    record
                )

    evaluation_dir = Path(
        output_cfg[
            "evaluation_dir"
        ]
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_csv(
        records,
        evaluation_dir
        / "per_image_metrics.csv",
    )

    print_summary(
        records
    )

    print(
        f"\nSaved:"
        f"\n{evaluation_dir / 'per_image_metrics.csv'}"
    )


if __name__ == "__main__":
    main()
