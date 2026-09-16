import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.config import load_config
from src.synthetic import generate_synthetic_dataset


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/unet_resnet34.yaml",
    )

    args = parser.parse_args()

    # ========================================================
    # Load config
    # ========================================================

    cfg = load_config(
        args.config
    )

    synthetic_cfg = cfg[
        "synthetic"
    ]

    # ========================================================
    # Generate synthetic dataset
    # ========================================================

    generate_synthetic_dataset(

        tissue_image_dir=
            synthetic_cfg[
                "tissue_image_dir"
            ],

        tissue_mask_dir=
            synthetic_cfg[
                "tissue_mask_dir"
            ],

        healthy_image_dir=
            synthetic_cfg[
                "healthy_image_dir"
            ],

        output_image_dir=
            synthetic_cfg[
                "output_image_dir"
            ],

        output_mask_dir=
            synthetic_cfg[
                "output_mask_dir"
            ],

        num_images=int(
            synthetic_cfg[
                "num_images"
            ]
        ),

        seed=int(
            cfg.get(
                "seed",
                42,
            )
        ),

        maximum_attempts=int(
            synthetic_cfg.get(
                "maximum_attempts",
                1000,
            )
        ),

        pupil_margin=int(
            synthetic_cfg.get(
                "pupil_margin",
                5,
            )
        ),

        iris_margin=int(
            synthetic_cfg.get(
                "iris_margin",
                5,
            )
        ),

        feather_size=int(
            synthetic_cfg.get(
                "feather_size",
                5,
            )
        ),
    )


if __name__ == "__main__":
    main()