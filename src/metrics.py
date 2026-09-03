import numpy as np


def _safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def calculate_dice(pred, target, smooth=1e-6):
    pred = pred.astype(np.float32).reshape(-1)
    target = target.astype(np.float32).reshape(-1)

    pred_sum = pred.sum()
    target_sum = target.sum()

    # Healthy GT かつ予測も空なら完全一致とする
    if pred_sum == 0 and target_sum == 0:
        return 1.0

    intersection = (pred * target).sum()

    return float(
        (2.0 * intersection + smooth)
        / (pred_sum + target_sum + smooth)
    )


def calculate_iou(pred, target, smooth=1e-6):
    pred = pred.astype(np.float32).reshape(-1)
    target = target.astype(np.float32).reshape(-1)

    pred_sum = pred.sum()
    target_sum = target.sum()

    if pred_sum == 0 and target_sum == 0:
        return 1.0

    intersection = (pred * target).sum()
    union = pred_sum + target_sum - intersection

    return float(
        (intersection + smooth)
        / (union + smooth)
    )


def calculate_pixel_metrics(pred, target):
    pred = pred.astype(np.uint8).reshape(-1)
    target = target.astype(np.uint8).reshape(-1)

    tp = np.logical_and(pred == 1, target == 1).sum()
    tn = np.logical_and(pred == 0, target == 0).sum()
    fp = np.logical_and(pred == 1, target == 0).sum()
    fn = np.logical_and(pred == 0, target == 1).sum()

    return {
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, tp + fn),
        "specificity": _safe_divide(tn, tn + fp),
        "pixel_accuracy": _safe_divide(
            tp + tn,
            tp + tn + fp + fn,
        ),
        "false_positive_rate": _safe_divide(fp, fp + tn),
    }
