import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def get_subject_id(filename):
    """
    Example:

    0011_R_IG_1_1.png
    0011_R_IG_2_1.png

    -> 0011
    """

    stem = Path(filename).stem

    return stem.split("_")[0]


def get_images(directory):
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    return sorted([
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    ])


def create_mask_lookup(mask_dir):
    masks = get_images(mask_dir)

    return {
        path.stem: path
        for path in masks
    }


def collect_dataset(
    geometry_images,
    geometry_masks,
    # tissue_images,
    # tissue_masks,
    # healthy_images,
):
    """
    Collect all images.

    Structure:

    subject_id
        -> list of samples

    Each sample:
        category
        image_path
        mask_path
    """

    subjects = defaultdict(list)

    # =====================================
    # Geometry
    # =====================================

    geometry_mask_lookup = create_mask_lookup(
        geometry_masks
    )

    for image_path in get_images(
        geometry_images
    ):
        stem = image_path.stem

        if stem not in geometry_mask_lookup:
            print(
                f"[WARNING] Geometry mask missing: "
                f"{image_path.name}"
            )
            continue

        subject_id = get_subject_id(
            image_path.name
        )

        subjects[subject_id].append({
            "category": "Geometry",
            "image_path": image_path,
            "mask_path":
                geometry_mask_lookup[stem],
        })

    # # =====================================
    # # Tissue
    # # =====================================

    # tissue_mask_lookup = create_mask_lookup(
    #     tissue_masks
    # )

    # for image_path in get_images(
    #     tissue_images
    # ):
    #     stem = image_path.stem

    #     if stem not in tissue_mask_lookup:
    #         print(
    #             f"[WARNING] Tissue mask missing: "
    #             f"{image_path.name}"
    #         )
    #         continue

    #     subject_id = get_subject_id(
    #         image_path.name
    #     )

    #     subjects[subject_id].append({
    #         "category": "Tissue",
    #         "image_path": image_path,
    #         "mask_path":
    #             tissue_mask_lookup[stem],
    #     })

    # # =====================================
    # # Healthy
    # # =====================================

    # for image_path in get_images(
    #     healthy_images
    # ):
    #     subject_id = get_subject_id(
    #         image_path.name
    #     )

    #     subjects[subject_id].append({
    #         "category": "Healthy",
    #         "image_path": image_path,
    #         "mask_path": None,
    #     })

    return subjects


def get_subject_category(samples):
    """
    Usually a subject belongs to one category.

    If multiple categories are present,
    return a combined category string.
    """

    categories = sorted(
        set(
            sample["category"]
            for sample in samples
        )
    )

    return "+".join(categories)


def split_subjects(
    subjects,
    train_ratio,
    val_ratio,
    test_ratio,
    seed,
):
    """
    Subject-level split.

    Same subject can NEVER appear
    in multiple splits.
    """

    if not np.isclose(
        train_ratio
        + val_ratio
        + test_ratio,
        1.0,
    ):
        raise ValueError(
            "train_ratio + val_ratio + "
            "test_ratio must equal 1.0"
        )

    random.seed(seed)

    # -------------------------------------
    # Group subjects by category
    # -------------------------------------

    category_subjects = defaultdict(list)

    for subject_id, samples in subjects.items():

        category = get_subject_category(
            samples
        )

        category_subjects[
            category
        ].append(
            subject_id
        )

    split_mapping = {}

    # -------------------------------------
    # Split each category independently
    # -------------------------------------

    for category, subject_ids in (
        category_subjects.items()
    ):

        random.shuffle(
            subject_ids
        )

        n = len(
            subject_ids
        )

        n_train = int(
            n * train_ratio
        )

        n_val = int(
            n * val_ratio
        )

        train_ids = (
            subject_ids[
                :n_train
            ]
        )

        val_ids = (
            subject_ids[
                n_train:
                n_train + n_val
            ]
        )

        test_ids = (
            subject_ids[
                n_train + n_val:
            ]
        )

        for subject_id in train_ids:
            split_mapping[
                subject_id
            ] = "train"

        for subject_id in val_ids:
            split_mapping[
                subject_id
            ] = "val"

        for subject_id in test_ids:
            split_mapping[
                subject_id
            ] = "test"

        print(
            f"\n{category}"
        )

        print(
            f"Subjects: {n}"
        )

        print(
            f"Train: {len(train_ids)}"
        )

        print(
            f"Val:   {len(val_ids)}"
        )

        print(
            f"Test:  {len(test_ids)}"
        )

    return split_mapping


def create_output_directories(
    output_dir,
):
    categories = [
        "Geometry",
        # "Tissue",
        # "Healthy",
    ]

    splits = [
        "train",
        "val",
        "test",
    ]

    for category in categories:

        for split in splits:

            (
                output_dir
                / category
                / split
                / "images"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                output_dir
                / category
                / split
                / "masks"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )


def save_abnormal_mask(
    mask_path,
    output_path,
):
    """
    Convert CVAT mask to:

    0   = background
    255 = abnormal
    """

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise RuntimeError(
            f"Cannot read mask: "
            f"{mask_path}"
        )

    binary_mask = np.where(
        mask > 0,
        255,
        0,
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        str(output_path),
        binary_mask,
    )


def create_healthy_mask(
    image_path,
    output_path,
):
    """
    Healthy GT:

    all pixels = 0
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read image: "
            f"{image_path}"
        )

    mask = np.zeros(
        image.shape,
        dtype=np.uint8,
    )

    cv2.imwrite(
        str(output_path),
        mask,
    )


def copy_dataset(
    subjects,
    split_mapping,
    output_dir,
):
    image_counts = defaultdict(
        int
    )

    for subject_id, samples in (
        subjects.items()
    ):

        split = split_mapping[
            subject_id
        ]

        for sample in samples:

            category = sample[
                "category"
            ]

            image_path = sample[
                "image_path"
            ]

            mask_path = sample[
                "mask_path"
            ]

            # =================================
            # Image output
            # =================================

            output_image_path = (
                output_dir
                / category
                / split
                / "images"
                / image_path.name
            )

            shutil.copy2(
                image_path,
                output_image_path,
            )

            # =================================
            # Mask output
            # =================================

            output_mask_path = (
                output_dir
                / category
                / split
                / "masks"
                / (
                    image_path.stem
                    + ".png"
                )
            )

            if category == "Healthy":

                create_healthy_mask(
                    image_path,
                    output_mask_path,
                )

            else:

                save_abnormal_mask(
                    mask_path,
                    output_mask_path,
                )

            image_counts[
                (
                    category,
                    split,
                )
            ] += 1

    return image_counts


def check_subject_leakage(
    subjects,
    split_mapping,
):
    """
    Sanity check.

    A subject should have exactly
    one split.
    """

    subject_splits = defaultdict(
        set
    )

    for subject_id in subjects:

        subject_splits[
            subject_id
        ].add(
            split_mapping[
                subject_id
            ]
        )

    leaked = {
        subject_id: splits
        for subject_id, splits
        in subject_splits.items()
        if len(splits) > 1
    }

    if leaked:
        raise RuntimeError(
            "Subject leakage detected: "
            f"{leaked}"
        )

    print(
        "\nNo subject leakage detected."
    )


def print_summary(
    image_counts,
):
    print(
        "\n"
        "=================================="
    )

    print(
        "Final image counts"
    )

    print(
        "=================================="
    )

    for category in [
        "Geometry",
        # "Tissue",
        # "Healthy",
    ]:

        print(
            f"\n{category}"
        )

        for split in [
            "train",
            "val",
            "test",
        ]:

            count = image_counts.get(
                (
                    category,
                    split,
                ),
                0,
            )

            print(
                f"  {split:<5}: "
                f"{count}"
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

    data_cfg = cfg[
        "data"
    ]

    raw_cfg = data_cfg[
        "raw"
    ]

    split_cfg = data_cfg[
        "split"
    ]

    seed = cfg.get(
        "seed",
        42,
    )

    output_dir = Path(
        data_cfg[
            "output_dir"
        ]
    )

    # =====================================
    # Create folders
    # =====================================

    create_output_directories(
        output_dir
    )

    # =====================================
    # Collect all data
    # =====================================

    subjects = collect_dataset(

        geometry_images=
            raw_cfg[
                "geometry_images"
            ],

        geometry_masks=
            raw_cfg[
                "geometry_masks"
            ],

        # tissue_images=
        #     raw_cfg[
        #         "tissue_images"
        #     ],

        # tissue_masks=
        #     raw_cfg[
        #         "tissue_masks"
        #     ],

        # healthy_images=
        #     raw_cfg[
        #         "healthy_images"
        #     ],
    )

    print(
        f"\nTotal unique subjects: "
        f"{len(subjects)}"
    )

    # =====================================
    # Subject-level split
    # =====================================

    split_mapping = split_subjects(

        subjects,

        train_ratio=
            split_cfg[
                "train_ratio"
            ],

        val_ratio=
            split_cfg[
                "val_ratio"
            ],

        test_ratio=
            split_cfg[
                "test_ratio"
            ],

        seed=seed,
    )

    # =====================================
    # Check leakage
    # =====================================

    check_subject_leakage(
        subjects,
        split_mapping,
    )

    # =====================================
    # Copy
    # =====================================

    image_counts = copy_dataset(
        subjects,
        split_mapping,
        output_dir,
    )

    # =====================================
    # Summary
    # =====================================

    print_summary(
        image_counts
    )

    print(
        f"\nDataset created:"
        f"\n{output_dir}"
    )


if __name__ == "__main__":
    main()