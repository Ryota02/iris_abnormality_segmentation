from pathlib import Path

import cv2
import numpy as np
import torch

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

    return model


def predict_image(
    model,
    image,
    image_size,
    threshold,
    device,
):
    """
    Predict one grayscale iris image.

    Returns
    -------
    probability : np.ndarray
        Pixel-wise abnormal probability [0, 1].

    prediction : np.ndarray
        Binary segmentation mask {0, 1}.
    """

    original_height, original_width = (
        image.shape
    )

    resized = cv2.resize(
        image,
        (
            image_size,
            image_size,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

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

    with torch.no_grad():

        logits = model(
            tensor
        )

        probability = torch.sigmoid(
            logits
        )

    probability = (
        probability[
            0,
            0,
        ]
        .cpu()
        .numpy()
    )

    probability = cv2.resize(
        probability,
        (
            original_width,
            original_height,
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