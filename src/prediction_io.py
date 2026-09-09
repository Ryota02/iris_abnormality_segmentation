from pathlib import Path

import cv2
import numpy as np


def save_prediction_results(
    output_dir,
    image,
    gt,
    prediction,
    error_map,
    probability_heatmap,
):
    """
    Save visualization results for one image.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt_image = (
        (gt > 0)
        .astype(np.uint8)
        * 255
    )

    prediction_image = (
        (prediction > 0)
        .astype(np.uint8)
        * 255
    )

    cv2.imwrite(
        str(
            output_dir
            / "original.png"
        ),
        image,
    )

    cv2.imwrite(
        str(
            output_dir
            / "ground_truth.png"
        ),
        gt_image,
    )

    cv2.imwrite(
        str(
            output_dir
            / "segmentation.png"
        ),
        prediction_image,
    )

    cv2.imwrite(
        str(
            output_dir
            / "color_map.png"
        ),
        error_map,
    )

    cv2.imwrite(
        str(
            output_dir
            / "heat_map.png"
        ),
        probability_heatmap,
    )