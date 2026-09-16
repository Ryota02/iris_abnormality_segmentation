from pathlib import Path
import random

import cv2

from torch.utils.data import Dataset

from src.augmentation import (
    SegmentationAugmentation,
)


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def find_mask_path(
    mask_dir,
    image_path,
):
    """
    Support:

        image.png -> image.png

    and:

        image.png -> image_mask.png
    """

    mask_dir = Path(
        mask_dir
    )

    candidates = [
        mask_dir
        / f"{image_path.stem}.png",

        mask_dir
        / f"{image_path.stem}_mask.png",
    ]

    for path in candidates:

        if path.exists():
            return path

    return None


class IrisSegmentationDataset(
    Dataset
):

    def __init__(
        self,
        root,
        split,
        image_size,
        categories,
        augmentation_config=None,
        synthetic_config=None,
    ):

        self.root = Path(
            root
        )

        self.split = split

        self.categories = (
            categories
        )

        self.samples = []

        # ====================================================
        # Real dataset
        # ====================================================

        self._add_real_samples(
            synthetic_config=
                synthetic_config,
        )

        self.real_count = len(
            self.samples
        )

        # ====================================================
        # Synthetic Tissue
        #
        # ONLY TRAIN
        # ====================================================

        if (
            split == "train"
            and synthetic_config is not None
            and synthetic_config.get(
                "enabled",
                False,
            )
        ):

            self._add_synthetic_tissue(
                synthetic_config
            )

        # ====================================================
        # Transform
        # ====================================================

        self.transform = (
            SegmentationAugmentation(
                image_size=
                    image_size,

                augmentation_config=
                    augmentation_config,

                train=(
                    split == "train"
                ),
            )
        )

        # ====================================================
        # Summary
        # ====================================================

        self._print_summary()

    # ========================================================
    # Real data
    # ========================================================

    def _add_real_samples(
        self,
        synthetic_config=None,
    ):
    
        # ========================================================
        # Whether Real Tissue should be included
        #
        # IMPORTANT:
        # This affects TRAIN only.
        # Val/Test always use real data.
        # ========================================================
    
        include_real_tissue = True
    
        if (
            self.split == "train"
            and synthetic_config is not None
        ):
    
            include_real_tissue = bool(
                synthetic_config.get(
                    "include_real_tissue",
                    True,
                )
            )
    
        # ========================================================
        # Categories
        # ========================================================
    
        for category in (
            self.categories
        ):
    
            # ----------------------------------------------------
            # Synthetic-only experiment:
            # exclude REAL Tissue from TRAIN
            # ----------------------------------------------------
    
            if (
                self.split == "train"
                and category == "Tissue"
                and not include_real_tissue
            ):
    
                print(
                    "[INFO] "
                    "Real Tissue excluded "
                    "from training."
                )
    
                continue
    
            image_dir = (
                self.root
                / category
                / self.split
                / "images"
            )
    
            mask_dir = (
                self.root
                / category
                / self.split
                / "masks"
            )
    
            if not image_dir.exists():
    
                raise FileNotFoundError(
                    f"Image directory "
                    f"not found: "
                    f"{image_dir}"
                )
    
            if not mask_dir.exists():
    
                raise FileNotFoundError(
                    f"Mask directory "
                    f"not found: "
                    f"{mask_dir}"
                )
    
            image_paths = sorted([
                path
                for path
                in image_dir.iterdir()
    
                if (
                    path.is_file()
                    and
                    path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            ])
    
            for image_path in (
                image_paths
            ):
    
                mask_path = (
                    find_mask_path(
                        mask_dir,
                        image_path,
                    )
                )
    
                if mask_path is None:
    
                    raise FileNotFoundError(
                        f"Mask not found for: "
                        f"{image_path}"
                    )
    
                self.samples.append(
                    {
                        "image_path":
                            image_path,
    
                        "mask_path":
                            mask_path,
    
                        "category":
                            category,
    
                        "is_synthetic":
                            False,
                    }
                )

    # ========================================================
    # Synthetic Tissue
    # ========================================================

    def _add_synthetic_tissue(
        self,
        synthetic_config,
    ):

        if (
            "Tissue"
            not in self.categories
        ):

            print(
                "[WARNING] "
                "Synthetic Tissue enabled, "
                "but Tissue is not included "
                "in data.categories."
            )

            return

        image_dir = Path(
            synthetic_config[
                "image_dir"
            ]
        )

        mask_dir = Path(
            synthetic_config[
                "mask_dir"
            ]
        )

        if not image_dir.exists():

            raise FileNotFoundError(
                f"Synthetic image directory "
                f"not found: "
                f"{image_dir}"
            )

        if not mask_dir.exists():

            raise FileNotFoundError(
                f"Synthetic mask directory "
                f"not found: "
                f"{mask_dir}"
            )

        image_paths = sorted([
            path
            for path
            in image_dir.iterdir()

            if (
                path.is_file()
                and
                path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ])

        # ----------------------------------------------------
        # Deterministic shuffle
        #
        # 10枚 experiment:
        #   synthetic 1-10
        #
        # 20枚 experiment:
        #   same 1-10 + additional 10
        #
        # とするための固定seed
        # ----------------------------------------------------

        shuffle_seed = int(
            synthetic_config.get(
                "shuffle_seed",
                42,
            )
        )

        rng = random.Random(
            shuffle_seed
        )

        rng.shuffle(
            image_paths
        )

        use_num = int(
            synthetic_config.get(
                "use_num",
                len(image_paths),
            )
        )

        if use_num > len(
            image_paths
        ):

            raise ValueError(
                f"Requested "
                f"{use_num} synthetic images, "
                f"but only "
                f"{len(image_paths)} "
                f"are available."
            )

        selected_paths = (
            image_paths[
                :use_num
            ]
        )

        for image_path in (
            selected_paths
        ):

            mask_path = (
                find_mask_path(
                    mask_dir,
                    image_path,
                )
            )

            if mask_path is None:

                raise FileNotFoundError(
                    f"Synthetic mask "
                    f"not found for: "
                    f"{image_path}"
                )

            self.samples.append(
                {
                    "image_path":
                        image_path,

                    "mask_path":
                        mask_path,

                    "category":
                        "Tissue",

                    "is_synthetic":
                        True,
                }
            )

    # ========================================================
    # Summary
    # ========================================================

    def _print_summary(
        self,
    ):

        print(
            f"\n"
            f"Dataset: "
            f"{self.split}"
        )

        for category in (
            self.categories
        ):

            real_count = sum(
                (
                    sample[
                        "category"
                    ]
                    == category

                    and
                    not sample[
                        "is_synthetic"
                    ]
                )

                for sample
                in self.samples
            )

            synthetic_count = sum(
                (
                    sample[
                        "category"
                    ]
                    == category

                    and
                    sample[
                        "is_synthetic"
                    ]
                )

                for sample
                in self.samples
            )

            total = (
                real_count
                + synthetic_count
            )

            print(
                f"{category:<10} "
                f"Real={real_count:<3} "
                f"Synthetic="
                f"{synthetic_count:<3} "
                f"Total={total:<3}"
            )

        print(
            f"Total: "
            f"{len(self.samples)}"
        )

    # ========================================================
    # Dataset
    # ========================================================

    def __len__(
        self,
    ):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        sample = (
            self.samples[
                index
            ]
        )

        image_path = (
            sample[
                "image_path"
            ]
        )

        mask_path = (
            sample[
                "mask_path"
            ]
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:

            raise RuntimeError(
                f"Cannot read image: "
                f"{image_path}"
            )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:

            raise RuntimeError(
                f"Cannot read mask: "
                f"{mask_path}"
            )

        image, mask = (
            self.transform(
                image,
                mask,
                sample[
                    "category"
                ],
            )
        )

        return {
            "image":
                image,

            "mask":
                mask,

            "category":
                sample[
                    "category"
                ],

            "filename":
                image_path.name,

            "is_synthetic":
                sample[
                    "is_synthetic"
                ],
        }