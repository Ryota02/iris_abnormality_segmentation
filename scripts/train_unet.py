import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.dataset import IrisSegmentationDataset
from src.losses import BCEDiceLoss
from src.metrics import calculate_dice, calculate_iou
from src.model import build_model
from src.plots import plot_training_history


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_validation(
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
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

            probs = torch.sigmoid(
                logits
            )

            preds = (
                probs >= threshold
            ).float()

            losses.append(
                loss.item()
            )

            for i in range(
                images.size(0)
            ):
                pred = (
                    preds[i, 0]
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                target = (
                    masks[i, 0]
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
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
        "loss": float(
            np.mean(losses)
        ),
        "dice": float(
            np.mean(dices)
        ),
        "iou": float(
            np.mean(ious)
        ),
    }


def save_history_csv(
    history,
    output_path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    keys = list(
        history.keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            ["epoch"] + keys
        )

        for i in range(
            len(history[keys[0]])
        ):
            writer.writerow(
                [i + 1]
                + [
                    history[key][i]
                    for key in keys
                ]
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

    seed = int(
        cfg.get(
            "seed",
            42,
        )
    )

    set_seed(seed)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    aug_cfg = cfg.get(
        "augmentation",
        {},
    )
    output_cfg = cfg["output"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    prepared_root = data_cfg["root"]
    categories = data_cfg["category"]

    train_dataset = (
        IrisSegmentationDataset(
            root=prepared_root,
            split="train",
            image_size=int(
                train_cfg["image_size"]
            ),
            categories=categories,
            train=True,
            augmentation_config=aug_cfg,
        )
    )

    val_dataset = (
        IrisSegmentationDataset(
            root=prepared_root,
            split="val",
            image_size=int(
                train_cfg["image_size"]
            ),
            categories=categories,
            train=False,
            augmentation_config=None,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(
            train_cfg["batch_size"]
        ),
        shuffle=True,
        num_workers=int(
            train_cfg["num_workers"]
        ),
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(
            train_cfg["batch_size"]
        ),
        shuffle=False,
        num_workers=int(
            train_cfg["num_workers"]
        ),
        pin_memory=torch.cuda.is_available(),
    )
    
    model = build_model(
       model_cfg
    ).to(device)

    loss_cfg = train_cfg["loss"]

    criterion = BCEDiceLoss(
        bce_weight=float(
            loss_cfg["bce_weight"]
        ),
        dice_weight=float(
            loss_cfg["dice_weight"]
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            train_cfg["learning_rate"]
        ),
        weight_decay=float(
            train_cfg["weight_decay"]
        ),
    )

    scheduler_cfg = train_cfg[
        "scheduler"
    ]

    scheduler = None
    
    if scheduler_cfg.get("enabled",False):
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=float(
                    scheduler_cfg["factor"]
                ),
                patience=int(
                    scheduler_cfg["patience"]
                ),
            )
    )

    output_dir = Path(
        output_cfg["dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        / output_cfg["checkpoint"]
    )

    history_path = (
        output_dir
        / output_cfg["history"]
    )

    loss_curve_path = (
        output_dir
        / output_cfg["loss_curve"]
    )

    metric_curve_path = (
        output_dir
        / output_cfg["metric_curve"]
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "lr": [],
    }

    best_dice = -1.0

    epochs = int(
        train_cfg["epochs"]
    )

    threshold = float(
        train_cfg["threshold"]
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        train_losses = []

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch}/{epochs}"
            ),
        )

        for batch in progress:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
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
                loss=f"{loss.item():.4f}"
            )

        train_loss = float(
            np.mean(train_losses)
        )

        val_metrics = (
            evaluate_validation(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                threshold=threshold,
            )
        )
        if scheduler is not None: 
            scheduler.step(
                val_metrics["dice"]
            )

        current_lr = float(
            optimizer.param_groups[0][
                "lr"
            ]
        )

        history[
            "train_loss"
        ].append(train_loss)

        history[
            "val_loss"
        ].append(
            val_metrics["loss"]
        )

        history[
            "val_dice"
        ].append(
            val_metrics["dice"]
        )

        history[
            "val_iou"
        ].append(
            val_metrics["iou"]
        )

        history[
            "lr"
        ].append(
            current_lr
        )

        print(
            f"\nEpoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"LR: {current_lr:.6g}"
        )

        if (
            val_metrics["dice"]
            > best_dice
        ):
            best_dice = (
                val_metrics["dice"]
            )

            torch.save(
                {
                    "epoch": epoch,
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

        save_history_csv(
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
        f"Best validation Dice: "
        f"{best_dice:.4f}"
    )


if __name__ == "__main__":
    main()
