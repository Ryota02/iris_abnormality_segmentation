import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.model import build_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/unet.yaml",
    )

    parser.add_argument(
        "--image",
        required=True,
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

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

    model = build_model(model_cfg)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    image_path = Path(
        args.image
    )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image: "
            f"{image_path}"
        )

    original_height, original_width = (
        image.shape
    )

    image_size = int(
        train_cfg["image_size"]
    )

    resized = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )

    tensor = (
        torch.from_numpy(
            resized.astype(
                np.float32
            ) / 255.0
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(
            tensor
        )

        probs = torch.sigmoid(
            logits
        )

        pred = (
            probs
            >= float(
                train_cfg["threshold"]
            )
        ).float()

    mask = (
        pred[0, 0]
        .cpu()
        .numpy()
        * 255
    ).astype(np.uint8)

    mask = cv2.resize(
        mask,
        (
            original_width,
            original_height,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    prediction_dir = Path(
        output_cfg[
            "prediction_dir"
        ]
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_path = (
        prediction_dir
        / f"{image_path.stem}_mask.png"
    )

    overlay_path = (
        prediction_dir
        / f"{image_path.stem}_overlay.png"
    )

    cv2.imwrite(
        str(mask_path),
        mask,
    )

    # grayscale -> BGR for visualization
    overlay = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR,
    )

    # 予測領域を白で重ねる
    white = np.full_like(
        overlay,
        255,
    )

    region = (
        mask > 0
    )

    overlay[
        region
    ] = (
        0.6
        * overlay[region]
        + 0.4
        * white[region]
    ).astype(np.uint8)

    cv2.imwrite(
        str(overlay_path),
        overlay,
    )

    print(
        f"Saved mask: "
        f"{mask_path}"
    )

    print(
        f"Saved overlay: "
        f"{overlay_path}"
    )


if __name__ == "__main__":
    main()
