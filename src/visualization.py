import cv2
import numpy as np


def create_segmentation_error_map(
    gt,
    prediction,
):
    """
    Segmentation error map.

    Green:
        Correct segmentation
        TP: GT=1, Pred=1

    Blue:
        Extra segmentation
        FP: GT=0, Pred=1

    Red:
        Missed segmentation
        FN: GT=1, Pred=0

    Black:
        Background
        TN: GT=0, Pred=0
    """

    gt = (
        gt > 0
    ).astype(
        np.uint8
    )

    prediction = (
        prediction > 0
    ).astype(
        np.uint8
    )

    well_area = (
        (gt == 1)
        & (prediction == 1)
    )

    extra_area = (
        (gt == 0)
        & (prediction == 1)
    )

    missed_area = (
        (gt == 1)
        & (prediction == 0)
    )

    color_map = np.zeros(
        (
            gt.shape[0],
            gt.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    # OpenCV uses BGR

    # Green = TP
    color_map[
        well_area
    ] = (
        0,
        255,
        0,
    )

    # Blue = FP
    color_map[
        extra_area
    ] = (
        255,
        0,
        0,
    )

    # Red = FN
    color_map[
        missed_area
    ] = (
        0,
        0,
        255,
    )

    return color_map


def create_probability_heatmap(
    probability,
):
    """
    Convert pixel-wise abnormal probability
    into a colored heat map.

    This is NOT Grad-CAM.

    probability:
        [H, W], range [0, 1]
    """

    probability_uint8 = (
        np.clip(
            probability * 255.0,
            0,
            255,
        )
        .astype(
            np.uint8
        )
    )

    heatmap = cv2.applyColorMap(
        probability_uint8,
        cv2.COLORMAP_JET,
    )

    return heatmap