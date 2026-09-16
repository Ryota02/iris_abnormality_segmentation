import torch

from torch.utils.data import (
    DataLoader,
)

from src.dataset import (
    IrisSegmentationDataset,
)

from src.sampler import (
    CategoryMultiplierSampler,
)


def build_train_sampler(
    dataset,
    cfg,
):

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

    return CategoryMultiplierSampler(
        samples=
            dataset.samples,

        category_multiplier=
            category_multiplier,

        seed=int(
            cfg.get(
                "seed",
                42,
            )
        ),
    )


def get_data_root(
    data_cfg,
):

    if "prepared_root" in data_cfg:

        return data_cfg[
            "prepared_root"
        ]

    if "root" in data_cfg:

        return data_cfg[
            "root"
        ]

    raise KeyError(
        "data.root or "
        "data.prepared_root "
        "is required."
    )


def build_dataloaders(
    cfg,
):

    data_cfg = cfg[
        "data"
    ]

    train_cfg = cfg[
        "training"
    ]

    augmentation_cfg = (
        cfg.get(
            "augmentation",
            {},
        )
    )

    synthetic_cfg = (
        cfg.get(
            "synthetic",
            {},
        )
    )

    categories = (
        data_cfg.get(
            "categories",
            [
                "Tissue",
                "Healthy",
            ],
        )
    )

    root = get_data_root(
        data_cfg
    )

    image_size = int(
        train_cfg[
            "image_size"
        ]
    )

    # ========================================================
    # TRAIN
    #
    # Synthetic allowed
    # ========================================================

    train_dataset = (
        IrisSegmentationDataset(
            root=root,

            split="train",

            image_size=
                image_size,

            categories=
                categories,

            augmentation_config=
                augmentation_cfg,

            synthetic_config=
                synthetic_cfg,
        )
    )

    # ========================================================
    # VALIDATION
    #
    # NO Synthetic
    # NO Augmentation
    # ========================================================

    val_dataset = (
        IrisSegmentationDataset(
            root=root,

            split="val",

            image_size=
                image_size,

            categories=
                categories,

            augmentation_config=None,

            synthetic_config=None,
        )
    )

    train_sampler = (
        build_train_sampler(
            train_dataset,
            cfg,
        )
    )

    train_loader = DataLoader(
        train_dataset,

        batch_size=int(
            train_cfg[
                "batch_size"
            ]
        ),

        shuffle=(
            train_sampler
            is None
        ),

        sampler=
            train_sampler,

        num_workers=int(
            train_cfg[
                "num_workers"
            ]
        ),

        pin_memory=
            torch.cuda.is_available(),
    )

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