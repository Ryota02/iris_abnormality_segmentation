import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/unet_resnet34.yaml",
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    root = Path(
        cfg["data"]["root"]
    )

    total = 0
    errors = 0

    for category in [
        "Geometry",
        "Tissue",
        "Healthy",
    ]:
        for split in [
            "train",
            "val",
            "test",
        ]:
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

            count = 0

            for image_path in sorted(
                image_dir.iterdir()
            ):
                if not image_path.is_file():
                    continue

                mask_path = (
                    mask_dir
                    / f"{image_path.stem}.png"
                )

                image = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_GRAYSCALE,
                )

                mask = cv2.imread(
                    str(mask_path),
                    cv2.IMREAD_GRAYSCALE,
                )

                if image is None:
                    print(
                        f"[ERROR] Cannot read image: "
                        f"{image_path}"
                    )
                    errors += 1
                    continue

                if mask is None:
                    print(
                        f"[ERROR] Missing/unreadable mask: "
                        f"{mask_path}"
                    )
                    errors += 1
                    continue

                if image.shape != mask.shape:
                    print(
                        f"[ERROR] Shape mismatch: "
                        f"{image_path.name}: "
                        f"image={image.shape}, "
                        f"mask={mask.shape}"
                    )
                    errors += 1

                unique_values = set(
                    np.unique(mask).tolist()
                )

                if not unique_values.issubset(
                    {0, 255}
                ):
                    print(
                        f"[ERROR] Non-binary mask: "
                        f"{mask_path} "
                        f"{sorted(unique_values)}"
                    )
                    errors += 1

                if (
                    category == "Healthy"
                    and np.any(mask != 0)
                ):
                    print(
                        f"[ERROR] Healthy mask "
                        f"contains positive pixels: "
                        f"{mask_path}"
                    )
                    errors += 1

                count += 1
                total += 1

            print(
                f"{category}/{split}: "
                f"{count} images"
            )

    print(
        f"\nChecked: {total} images"
    )

    if errors == 0:
        print(
            "Dataset check passed."
        )
    else:
        print(
            f"Dataset check failed: "
            f"{errors} error(s)"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
