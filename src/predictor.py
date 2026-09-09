from pathlib import Path

import cv2
import numpy as np
import torch

from tqdm import tqdm

from src.inference import (
    load_model,
    predict_image,
)

from src.metrics import (
    calculate_dice,
    calculate_iou,
)

from src.prediction_io import (
    save_prediction_results,
)

from src.visualization import (
    create_probability_heatmap,
    create_segmentation_error_map,
)


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def process_one_image(
    model,
    image_path,
    gt_path,
    output_dir,
    image_size,
    threshold,
    device,
):
    # ========================================================
    # Load image
    # ========================================================

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read image: "
            f"{image_path}"
        )

    # ========================================================
    # Load GT
    # ========================================================

    gt = cv2.imread(
        str(gt_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if gt is None:
        raise RuntimeError(
            f"Cannot read GT: "
            f"{gt_path}"
        )

    if gt.shape != image.shape:

        print(
            f"[WARNING] "
            f"GT shape differs: "
            f"{image_path.name}"
        )

        gt = cv2.resize(
            gt,
            (
                image.shape[1],
                image.shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    gt_binary = (
        gt > 0
    ).astype(
        np.uint8
    )

    # ========================================================
    # Inference
    # ========================================================

    probability, prediction = (
        predict_image(
            model=model,
            image=image,
            image_size=image_size,
            threshold=threshold,
            device=device,
        )
    )

    # ========================================================
    # Visualization
    # ========================================================

    error_map = (
        create_segmentation_error_map(
            gt=gt_binary,
            prediction=prediction,
        )
    )

    probability_heatmap = (
        create_probability_heatmap(
            probability
        )
    )

    # ========================================================
    # Save
    # ========================================================

    save_prediction_results(
        output_dir=output_dir,
        image=image,
        gt=gt_binary,
        prediction=prediction,
        error_map=error_map,
        probability_heatmap=
            probability_heatmap,
    )

    # ========================================================
    # Metrics
    # ========================================================

    dice = calculate_dice(
        prediction,
        gt_binary,
    )

    iou = calculate_iou(
        prediction,
        gt_binary,
    )

    return {
        "dice":
            dice,

        "iou":
            iou,

        "gt_ratio":
            float(
                gt_binary.mean()
            ),

        "pred_ratio":
            float(
                prediction.mean()
            ),

        "probability_max":
            float(
                probability.max()
            ),
    }


def run_prediction(
    cfg,
    split="test",
):
    data_cfg = cfg[
        "data"
    ]

    train_cfg = cfg[
        "training"
    ]

    output_cfg = cfg[
        "output"
    ]

    categories = data_cfg.get(
        "categories",
        [
            "Geometry",
            "Tissue",
            "Healthy",
        ],
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # Model
    # ========================================================

    model = load_model(
        cfg,
        device,
    )

    # ========================================================
    # Paths / settings
    # ========================================================

    prepared_root = Path(
        data_cfg["root"]
    )

    prediction_root = (
        Path(
            output_cfg[
                "prediction_dir"
            ]
        )
        / split
    )

    image_size = int(
        train_cfg[
            "image_size"
        ]
    )

    threshold = float(
        train_cfg[
            "threshold"
        ]
    )

    print(
        f"Split: {split}"
    )

    print(
        f"Threshold: {threshold}"
    )

    total_images = 0

    # ========================================================
    # Categories
    # ========================================================

    for category in categories:

        image_dir = (
            prepared_root
            / category
            / split
            / "images"
        )

        gt_dir = (
            prepared_root
            / category
            / split
            / "masks"
        )

        if not image_dir.exists():

            print(
                f"[WARNING] "
                f"Image directory not found: "
                f"{image_dir}"
            )

            continue

        if not gt_dir.exists():

            print(
                f"[WARNING] "
                f"GT directory not found: "
                f"{gt_dir}"
            )

            continue

        image_paths = sorted([
            path
            for path in image_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ])

        print(
            f"\n{category}: "
            f"{len(image_paths)} images"
        )

        for image_path in tqdm(
            image_paths,
            desc=category,
        ):

            gt_path = (
                gt_dir
                / f"{image_path.stem}.png"
            )

            if not gt_path.exists():

                tqdm.write(
                    f"[WARNING] "
                    f"GT not found: "
                    f"{gt_path}"
                )

                continue

            image_output_dir = (
                prediction_root
                / category
                / image_path.stem
            )

            metrics = process_one_image(
                model=model,
                image_path=image_path,
                gt_path=gt_path,
                output_dir=image_output_dir,
                image_size=image_size,
                threshold=threshold,
                device=device,
            )

            total_images += 1

            tqdm.write(
                f"{category:<8} | "
                f"{image_path.name} | "
                f"Dice="
                f"{metrics['dice']:.4f} | "
                f"IoU="
                f"{metrics['iou']:.4f} | "
                f"GT="
                f"{metrics['gt_ratio']:.4f} | "
                f"Pred="
                f"{metrics['pred_ratio']:.4f} | "
                f"Pmax="
                f"{metrics['probability_max']:.4f}"
            )

    print(
        "\n=================================="
    )

    print(
        "Prediction finished"
    )

    print(
        "=================================="
    )

    print(
        f"Processed images: "
        f"{total_images}"
    )

    print(
        f"Results saved to:"
        f"\n{prediction_root}"
    )