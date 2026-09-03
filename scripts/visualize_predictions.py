import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.model import build_model


def load_model(
    cfg,
    device,
):
    model_cfg = cfg["model"]
    output_cfg = cfg["output"]

    checkpoint_path = (
        Path(output_cfg["dir"])
        / output_cfg["checkpoint"]
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
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def predict_mask(
    model,
    image,
    image_size,
    threshold,
    device,
):
    """
    Returns:
        probability map
        binary prediction mask
    """

    original_h, original_w = (
        image.shape
    )

    # -----------------------------
    # Resize
    # -----------------------------

    resized = cv2.resize(
        image,
        (
            image_size,
            image_size,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    # -----------------------------
    # Normalize
    # -----------------------------

    tensor = (
        resized.astype(
            np.float32
        )
        / 255.0
    )

    tensor = (
        torch.from_numpy(
            tensor
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    # -----------------------------
    # Prediction
    # -----------------------------

    with torch.no_grad():

        logits = model(
            tensor
        )

        probs = torch.sigmoid(
            logits
        )

    probability = (
        probs[0, 0]
        .cpu()
        .numpy()
    )

    # -----------------------------
    # Return to original size
    # -----------------------------

    probability = cv2.resize(
        probability,
        (
            original_w,
            original_h,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    prediction = (
        probability
        >= threshold
    ).astype(
        np.uint8
    )

    return (
        probability,
        prediction,
    )


def calculate_dice(
    prediction,
    target,
    smooth=1e-6,
):
    prediction = (
        prediction
        .astype(np.float32)
        .reshape(-1)
    )

    target = (
        target
        .astype(np.float32)
        .reshape(-1)
    )

    pred_sum = (
        prediction.sum()
    )

    target_sum = (
        target.sum()
    )

    if (
        pred_sum == 0
        and target_sum == 0
    ):
        return 1.0

    intersection = (
        prediction
        * target
    ).sum()

    dice = (
        2.0 * intersection
        + smooth
    ) / (
        pred_sum
        + target_sum
        + smooth
    )

    return float(
        dice
    )


def calculate_iou(
    prediction,
    target,
    smooth=1e-6,
):
    prediction = (
        prediction
        .astype(np.float32)
        .reshape(-1)
    )

    target = (
        target
        .astype(np.float32)
        .reshape(-1)
    )

    intersection = (
        prediction
        * target
    ).sum()

    union = (
        prediction.sum()
        + target.sum()
        - intersection
    )

    if union == 0:
        return 1.0

    return float(
        (
            intersection
            + smooth
        )
        / (
            union
            + smooth
        )
    )


def create_overlay(
    image,
    gt,
    prediction,
):
    """
    Overlay:
        GT only         -> red
        Prediction only -> blue
        Overlap         -> green
    """

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2RGB,
    )

    overlay = (
        image_rgb.copy()
    )

    # GT only
    gt_only = (
        (gt == 1)
        & (prediction == 0)
    )

    # Prediction only
    pred_only = (
        (gt == 0)
        & (prediction == 1)
    )

    # Correct overlap
    overlap = (
        (gt == 1)
        & (prediction == 1)
    )

    overlay[
        gt_only
    ] = [
        255,
        0,
        0,
    ]

    overlay[
        pred_only
    ] = [
        0,
        0,
        255,
    ]

    overlay[
        overlap
    ] = [
        0,
        255,
        0,
    ]

    # Blend with original
    result = cv2.addWeighted(
        image_rgb,
        0.6,
        overlay,
        0.4,
        0,
    )

    return result


def visualize_one(
    model,
    image_path,
    mask_path,
    category,
    output_path,
    image_size,
    threshold,
    device,
):
    # =============================
    # Load image
    # =============================

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    gt = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read image: "
            f"{image_path}"
        )

    if gt is None:
        raise RuntimeError(
            f"Cannot read mask: "
            f"{mask_path}"
        )

    # Binary GT
    gt = (
        gt > 0
    ).astype(
        np.uint8
    )

    # =============================
    # Prediction
    # =============================

    probability, prediction = (
        predict_mask(
            model=model,
            image=image,
            image_size=image_size,
            threshold=threshold,
            device=device,
        )
    )

    # =============================
    # Metrics
    # =============================

    dice = calculate_dice(
        prediction,
        gt,
    )

    iou = calculate_iou(
        prediction,
        gt,
    )

    gt_ratio = (
        gt.mean()
    )

    pred_ratio = (
        prediction.mean()
    )

    prob_min = (
        probability.min()
    )

    prob_max = (
        probability.max()
    )

    prob_mean = (
        probability.mean()
    )

    # =============================
    # Overlay
    # =============================

    overlay = create_overlay(
        image,
        gt,
        prediction,
    )

    # =============================
    # Plot
    # =============================

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(
            20,
            4,
        ),
    )

    # Original
    axes[0].imshow(
        image,
        cmap="gray",
    )

    axes[0].set_title(
        "Original"
    )

    # GT
    axes[1].imshow(
        gt,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        f"Ground Truth\n"
        f"positive={gt_ratio:.4f}"
    )

    # Probability
    im = axes[2].imshow(
        probability,
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        f"Probability\n"
        f"min={prob_min:.3f} "
        f"max={prob_max:.3f}"
    )

    fig.colorbar(
        im,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )

    # Prediction
    axes[3].imshow(
        prediction,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[3].set_title(
        f"Prediction\n"
        f"positive={pred_ratio:.4f}"
    )

    # Overlay
    axes[4].imshow(
        overlay
    )

    axes[4].set_title(
        f"Overlay\n"
        f"Dice={dice:.4f} "
        f"IoU={iou:.4f}"
    )

    for ax in axes:
        ax.axis(
            "off"
        )

    fig.suptitle(
        f"{category} | "
        f"{image_path.name}\n"
        f"Threshold={threshold:.2f} | "
        f"Probability mean={prob_mean:.4f}",
        fontsize=12,
    )

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"{category:<8} "
        f"{image_path.name:<30} "
        f"Dice={dice:.4f} "
        f"IoU={iou:.4f} "
        f"GT={gt_ratio:.4f} "
        f"Pred={pred_ratio:.4f} "
        f"Pmax={prob_max:.4f}"
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

    parser.add_argument(
        "--split",
        default="val",
        choices=[
            "train",
            "val",
            "test",
        ],
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    # =============================
    # Config
    # =============================

    cfg = load_config(
        args.config
    )

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    output_cfg = cfg["output"]

    prepared_root = Path(
        data_cfg["root"]
    )

    image_size = int(
        train_cfg["image_size"]
    )

    threshold = float(
        train_cfg["threshold"]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # =============================
    # Model
    # =============================

    model = load_model(
        cfg,
        device,
    )

    # =============================
    # Output directory
    # =============================

    output_dir = (
        Path(
            output_cfg["dir"]
        )
        / "visualizations"
        / args.split
    )

    # =============================
    # Each category
    # =============================
    categories = data_cfg["category"]
    for category in categories:

        image_dir = (
            prepared_root
            / category
            / args.split
            / "images"
        )

        mask_dir = (
            prepared_root
            / category
            / args.split
            / "masks"
        )

        if not image_dir.exists():

            print(
                f"[WARNING] Not found: "
                f"{image_dir}"
            )

            continue

        image_paths = sorted(
            image_dir.glob(
                "*.png"
            )
        )

        image_paths = (
            image_paths[
                :args.num_images
            ]
        )

        for image_path in image_paths:

            mask_path = (
                mask_dir
                / (
                    image_path.stem
                    + ".png"
                )
            )

            if not mask_path.exists():

                print(
                    f"[WARNING] "
                    f"Mask not found: "
                    f"{mask_path}"
                )

                continue

            output_path = (
                output_dir
                / category
                / (
                    image_path.stem
                    + "_comparison.png"
                )
            )

            visualize_one(
                model=model,
                image_path=image_path,
                mask_path=mask_path,
                category=category,
                output_path=output_path,
                image_size=image_size,
                threshold=threshold,
                device=device,
            )


if __name__ == "__main__":
    main()