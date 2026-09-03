from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(
    history,
    loss_path,
    metric_path,
):
    loss_path = Path(loss_path)
    metric_path = Path(metric_path)

    loss_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        history["val_loss"],
        label="Val Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["val_dice"],
        label="Val Dice",
    )
    plt.plot(
        epochs,
        history["val_iou"],
        label="Val IoU",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(metric_path, dpi=200)
    plt.close()
