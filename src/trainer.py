import csv
import random
from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm

from src.data import build_dataloaders
from src.losses import BCEDiceLoss
from src.metrics import (
    calculate_dice,
    calculate_iou,
)
from src.model import build_model
from src.plots import (
    plot_training_history,
)
from src.utils import set_seed


def validate(
    model,
    loader,
    criterion,
    device,
    threshold,
):

    model.eval()

    losses = []
    dices = []
    ious = []

    with torch.no_grad():

        for batch in loader:

            images = (
                batch["image"]
                .to(device)
            )

            masks = (
                batch["mask"]
                .to(device)
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                masks,
            )

            probs = torch.sigmoid(
                logits
            )

            preds = (
                probs
                >= threshold
            ).float()

            losses.append(
                loss.item()
            )

            for i in range(
                images.size(0)
            ):

                pred = (
                    preds[
                        i,
                        0,
                    ]
                    .cpu()
                    .numpy()
                    .astype(
                        np.uint8
                    )
                )

                target = (
                    masks[
                        i,
                        0,
                    ]
                    .cpu()
                    .numpy()
                    .astype(
                        np.uint8
                    )
                )

                dices.append(
                    calculate_dice(
                        pred,
                        target,
                    )
                )

                ious.append(
                    calculate_iou(
                        pred,
                        target,
                    )
                )

    return {
        "loss":
            float(
                np.mean(
                    losses
                )
            ),

        "dice":
            float(
                np.mean(
                    dices
                )
            ),

        "iou":
            float(
                np.mean(
                    ious
                )
            ),
    }


def save_history(
    history,
    path,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    keys = list(
        history.keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            ["epoch"]
            + keys
        )

        for epoch in range(
            len(
                history[
                    keys[0]
                ]
            )
        ):

            writer.writerow(
                [epoch + 1]
                + [
                    history[key][
                        epoch
                    ]
                    for key
                    in keys
                ]
            )


def train_from_config(
    cfg,
):

    seed = int(
        cfg.get(
            "seed",
            42,
        )
    )

    set_seed(seed)

    train_cfg = (
        cfg["training"]
    )

    model_cfg = (
        cfg["model"]
    )

    output_cfg = (
        cfg["output"]
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # Data
    # ========================================================

    (
        train_loader,
        val_loader,
    ) = build_dataloaders(
        cfg
    )

    # ========================================================
    # Model
    # ========================================================

    model = (
        build_model(
            model_cfg
        )
        .to(device)
    )

    # ========================================================
    # Loss
    # ========================================================

    loss_cfg = (
        train_cfg[
            "loss"
        ]
    )

    criterion = (
        BCEDiceLoss(
            bce_weight=float(
                loss_cfg[
                    "bce_weight"
                ]
            ),

            dice_weight=float(
                loss_cfg[
                    "dice_weight"
                ]
            ),
        )
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),

            lr=float(
                train_cfg[
                    "learning_rate"
                ]
            ),

            weight_decay=float(
                train_cfg[
                    "weight_decay"
                ]
            ),
        )
    )

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler_cfg = (
        train_cfg.get(
            "scheduler",
            {},
        )
    )

    scheduler = None

    if scheduler_cfg.get(
        "enabled",
        False,
    ):

        scheduler = (
            torch.optim.lr_scheduler
            .ReduceLROnPlateau(
                optimizer,

                mode="max",

                factor=float(
                    scheduler_cfg[
                        "factor"
                    ]
                ),

                patience=int(
                    scheduler_cfg[
                        "patience"
                    ]
                ),
            )
        )

    # ========================================================
    # Output
    # ========================================================

    output_dir = Path(
        output_cfg[
            "dir"
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        / output_cfg[
            "checkpoint"
        ]
    )

    history_path = (
        output_dir
        / output_cfg[
            "history_csv"
        ]
    )

    loss_curve_path = (
        output_dir
        / output_cfg[
            "loss_curve"
        ]
    )

    metric_curve_path = (
        output_dir
        / output_cfg[
            "metric_curve"
        ]
    )

    # ========================================================
    # History
    # ========================================================

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "lr": [],
    }

    best_dice = -1.0

    epochs = int(
        train_cfg[
            "epochs"
        ]
    )

    threshold = float(
        train_cfg[
            "threshold"
        ]
    )

    # ========================================================
    # Epoch
    # ========================================================

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        train_losses = []

        progress = tqdm(
            train_loader,

            desc=(
                f"Epoch "
                f"{epoch}/"
                f"{epochs}"
            ),
        )

        for batch in progress:

            images = (
                batch["image"]
                .to(device)
            )

            masks = (
                batch["mask"]
                .to(device)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                masks,
            )

            loss.backward()

            optimizer.step()

            train_losses.append(
                loss.item()
            )

            progress.set_postfix(
                loss=(
                    f"{loss.item():.4f}"
                )
            )

        train_loss = float(
            np.mean(
                train_losses
            )
        )

        # ====================================================
        # Validation
        # ====================================================

        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
        )

        # ====================================================
        # Scheduler
        # ====================================================

        if scheduler is not None:

            scheduler.step(
                val_metrics[
                    "dice"
                ]
            )

        current_lr = float(
            optimizer.param_groups[
                0
            ]["lr"]
        )

        # ====================================================
        # History
        # ====================================================

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "val_loss"
        ].append(
            val_metrics[
                "loss"
            ]
        )

        history[
            "val_dice"
        ].append(
            val_metrics[
                "dice"
            ]
        )

        history[
            "val_iou"
        ].append(
            val_metrics[
                "iou"
            ]
        )

        history[
            "lr"
        ].append(
            current_lr
        )

        print(
            f"\nEpoch {epoch:03d} | "
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Val Loss: "
            f"{val_metrics['loss']:.4f} | "
            f"Val Dice: "
            f"{val_metrics['dice']:.4f} | "
            f"Val IoU: "
            f"{val_metrics['iou']:.4f} | "
            f"LR: "
            f"{current_lr:.6g}"
        )

        # ====================================================
        # Best model
        # ====================================================

        if (
            val_metrics[
                "dice"
            ]
            > best_dice
        ):

            best_dice = (
                val_metrics[
                    "dice"
                ]
            )

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_dice":
                        best_dice,

                    "config":
                        cfg,
                },

                checkpoint_path,
            )

            print(
                f"Saved best model: "
                f"{checkpoint_path}"
            )

        save_history(
            history,
            history_path,
        )

        plot_training_history(
            history,
            loss_curve_path,
            metric_curve_path,
        )

    print(
        "\nTraining completed."
    )

    print(
        f"Best Val Dice: "
        f"{best_dice:.4f}"
    )