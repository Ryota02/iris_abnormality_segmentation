import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {
    ".png"
}


class SegmentationTransform:
    def __init__(
        self,
        image_size,
        train=False,
        augmentation_config=None,
    ):
        self.image_size = image_size
        self.train = train
        self.augmentation_config = augmentation_config or {}

    def __call__(self, image, mask):
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        if self.train:
            flip_cfg = self.augmentation_config.get(
                "horizontal_flip",
                {},
            )
            if flip_cfg.get("enabled", False):
                probability = flip_cfg.get("probability", 0.5)
                if random.random() < probability:
                    image = np.fliplr(image).copy()
                    mask = np.fliplr(mask).copy()

            rotation_cfg = self.augmentation_config.get(
                "rotation",
                {},
            )
            if rotation_cfg.get("enabled", False):
                probability = rotation_cfg.get("probability", 0.5)

                if random.random() < probability:
                    limit = rotation_cfg.get("limit", 10)
                    angle = random.uniform(-limit, limit)

                    center = (
                        self.image_size / 2.0,
                        self.image_size / 2.0,
                    )

                    matrix = cv2.getRotationMatrix2D(
                        center,
                        angle,
                        1.0,
                    )

                    image = cv2.warpAffine(
                        image,
                        matrix,
                        (self.image_size, self.image_size),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )

                    mask = cv2.warpAffine(
                        mask,
                        matrix,
                        (self.image_size, self.image_size),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


class IrisSegmentationDataset(Dataset):
    """
    Expected structure:

    root/
      Geometry/
        train/images/
        train/masks/
        val/images/
        val/masks/
        test/images/
        test/masks/
      Tissue/
        ...
      Healthy/
        ...
    """

    def __init__(
        self,
        root,
        split,
        image_size=512,
        categories=None,
        train=False,
        augmentation_config=None,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.train = train

        if categories is None:
            categories = [
                "Geometry",
                "Tissue",
                "Healthy",
            ]

        self.categories = categories
        self.samples = []

        for category in categories:
            image_dir = (
                self.root
                / category
                / split
                / "images"
            )

            mask_dir = (
                self.root
                / category
                / split
                / "masks"
            )

            if not image_dir.exists():
                raise FileNotFoundError(
                    f"Image directory not found: {image_dir}"
                )

            if not mask_dir.exists():
                raise FileNotFoundError(
                    f"Mask directory not found: {mask_dir}"
                )

            for image_path in sorted(image_dir.iterdir()):
                if (
                    not image_path.is_file()
                    or image_path.suffix.lower()
                    not in IMAGE_EXTENSIONS
                ):
                    continue

                mask_path = (
                    mask_dir
                    / f"{image_path.stem}.png"
                )

                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"Mask not found for {image_path.name}: "
                        f"{mask_path}"
                    )

                self.samples.append({
                    "image": image_path,
                    "mask": mask_path,
                    "category": category,
                })

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for split: {split}"
            )

        self.transform = SegmentationTransform(
            image_size=image_size,
            train=train,
            augmentation_config=augmentation_config,
        )

        print(f"\n{split}: {len(self.samples)} images")

        for category in categories:
            count = sum(
                sample["category"] == category
                for sample in self.samples
            )
            print(f"  {category}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = sample["image"]
        mask_path = sample["mask"]
        category = sample["category"]

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise RuntimeError(
                f"Cannot read image: {image_path}"
            )

        if mask is None:
            raise RuntimeError(
                f"Cannot read mask: {mask_path}"
            )

        image, mask = self.transform(
            image,
            mask,
        )

        return {
            "image": image,
            "mask": mask,
            "category": category,
            "filename": image_path.name,
        }
