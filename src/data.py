import torch

from torch.utils.data import (
    DataLoader,
)

from src.dataset import (
    IrisSegmentationDataset
)

from src.sampler import (
    CategoryMultiplierSampler
)


def build_train_sampler(
    dataset,
    cfg,
):
    """
    Build oversampling sampler.

    Returns:
        sampler or None
    """

    sampling_cfg = cfg.get(
        "sampling",
        {},
    )

    if not sampling_cfg.get(
        "enabled",
        False,
    ):

        print(
            "\nOversampling: disabled"
        )

        return None

    category_multiplier = (
        sampling_cfg.get(
            "category_multiplier",
            {},
        )
    )

    sampler = (
        CategoryMultiplierSampler(
            samples=dataset.samples,
            category_multiplier=
                category_multiplier,
            seed=int(
                cfg.get(
                    "seed",
                    42,
                )
            ),
        )
    )

    return sampler


def build_dataloaders(
    cfg,
):

    data_cfg = (
        cfg["data"]
    )

    train_cfg = (
        cfg["training"]
    )

    augmentation_cfg = (
        cfg.get(
            "augmentation",
            {},
        )
    )

    categories = (
        data_cfg.get(
            "categories",
            [
                "Geometry",
                "Tissue",
                "Healthy",
            ],
        )
    )

    root = (
        data_cfg[
            "root"
        ]
    )

    image_size = int(
        train_cfg[
            "image_size"
        ]
    )

    # ========================================================
    # Train Dataset
    # ========================================================

    train_dataset = (
        IrisSegmentationDataset(
            root=root,
            split="train",
            image_size=image_size,
            categories=categories,
            augmentation_config=
                augmentation_cfg,
        )
    )

    # ========================================================
    # Validation Dataset
    #
    # augmentationなし
    # oversamplingなし
    # ========================================================

    val_dataset = (
        IrisSegmentationDataset(
            root=root,
            split="val",
            image_size=image_size,
            categories=categories,
            augmentation_config=None,
        )
    )

    # ========================================================
    # Oversampling
    # ========================================================

    train_sampler = (
        build_train_sampler(
            train_dataset,
            cfg,
        )
    )

    # sampler使用時はshuffle=False
    shuffle = (
        train_sampler is None
    )

    # ========================================================
    # Train Loader
    # ========================================================

    train_loader = DataLoader(
        train_dataset,

        batch_size=int(
            train_cfg[
                "batch_size"
            ]
        ),

        shuffle=shuffle,

        sampler=train_sampler,

        num_workers=int(
            train_cfg[
                "num_workers"
            ]
        ),

        pin_memory=
            torch.cuda.is_available(),
    )

    # ========================================================
    # Validation Loader
    # ========================================================

    val_loader = DataLoader(
        val_dataset,

        batch_size=int(
            train_cfg[
                "batch_size"
            ]
        ),

        shuffle=False,

        num_workers=int(
            train_cfg[
                "num_workers"
            ]
        ),

        pin_memory=
            torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
    )