from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages


AUGMENTATION_METHODS = [
    "horizontal_flip",
    "rotation",
    "brightness",
    "contrast",
    "gamma",
    "gaussian_noise",
    "gaussian_blur",
]


# ============================================================
# Load CSV
# ============================================================


def load_ablation_results(
    csv_path,
):
    csv_path = Path(
        csv_path
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Summary CSV not found: "
            f"{csv_path}"
        )

    df = pd.read_csv(
        csv_path
    )

    required_columns = {
        "experiment",
        "best_epoch",
        "best_val_dice",
        "best_val_iou",
        "test_dice",
        "test_iou",
        "tissue_test_dice",
        "tissue_test_iou",
    }

    missing = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing columns in summary CSV: "
            f"{sorted(missing)}"
        )

    return df


# ============================================================
# Experiment classification
# ============================================================


def classify_experiment(
    experiment,
):
    """
    Return:
        baseline
        augmentation
        oversampling
        augmentation_oversampling
    """

    experiment = str(
        experiment
    )

    if experiment == "baseline":
        return "baseline"

    if experiment == "oversampling":
        return "oversampling"

    if experiment.endswith(
        "_oversampling"
    ):
        return (
            "augmentation_oversampling"
        )

    if (
        experiment
        in AUGMENTATION_METHODS
    ):
        return "augmentation"

    return "other"


def add_experiment_group(
    df,
):
    df = df.copy()

    df[
        "group"
    ] = df[
        "experiment"
    ].apply(
        classify_experiment
    )

    return df


# ============================================================
# Formatting
# ============================================================


def pretty_name(
    name,
):
    replacements = {
        "baseline":
            "Baseline",

        "oversampling":
            "Oversampling",

        "horizontal_flip":
            "Horizontal Flip",

        "rotation":
            "Rotation",

        "brightness":
            "Brightness",

        "contrast":
            "Contrast",

        "gamma":
            "Gamma",

        "gaussian_noise":
            "Gaussian Noise",

        "gaussian_blur":
            "Gaussian Blur",
    }

    if name in replacements:
        return replacements[
            name
        ]

    if name.endswith(
        "_oversampling"
    ):

        base_name = name[
            :-len(
                "_oversampling"
            )
        ]

        return (
            f"{pretty_name(base_name)} "
            f"+ Oversampling"
        )

    return str(
        name
    )


def metric_text(
    value,
):
    if pd.isna(
        value
    ):
        return "-"

    return f"{float(value):.4f}"


# ============================================================
# Page 1: experiment overview
# ============================================================


def draw_overview_page(
    pdf,
    df,
    model_name,
):
    fig = plt.figure(
        figsize=(
            11.69,
            8.27,
        )
    )

    fig.suptitle(
        "Augmentation / Oversampling Ablation Study",
        fontsize=18,
        y=0.96,
    )

    ax = fig.add_axes(
        [
            0.05,
            0.08,
            0.90,
            0.80,
        ]
    )

    ax.axis(
        "off"
    )

    lines = [
        f"Model: {model_name}",
        "",
        "Experimental conditions:",
        "",
        "1. Baseline",
        "   - Augmentation: OFF",
        "   - Oversampling: OFF",
        "",
        "2. Augmentation only",
        "   - Exactly one augmentation method is enabled per experiment.",
        "   - Oversampling: OFF",
        "",
        "3. Oversampling only",
        "   - Augmentation: OFF",
        "   - Oversampling: ON",
        "",
        "4. Augmentation + Oversampling",
        "   - Exactly one augmentation method is enabled per experiment.",
        "   - Oversampling: ON",
        "",
        "Augmentation methods:",
    ]

    for method in AUGMENTATION_METHODS:

        lines.append(
            f"   - {pretty_name(method)}"
        )

    lines.extend(
        [
            "",
            (
                "Validation and test sets must not "
                "use augmentation or oversampling."
            ),
            (
                "Test metrics are for final evaluation; "
                "do not use them to choose the best model."
            ),
        ]
    )

    ax.text(
        0.02,
        0.98,
        "\n".join(
            lines
        ),
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.45,
    )

    pdf.savefig(
        fig,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Page 2: quantitative table
# ============================================================


def draw_results_table(
    pdf,
    df,
):
    fig, ax = plt.subplots(
        figsize=(
            16,
            9,
        )
    )

    ax.axis(
        "off"
    )

    ax.set_title(
        "Quantitative Results",
        fontsize=18,
        pad=20,
    )

    headers = [
        "Experiment",
        "Best\nEpoch",
        "Val\nDice",
        "Val\nIoU",
        "Test\nDice",
        "Test\nIoU",
        "Tissue\nDice",
        "Tissue\nIoU",
    ]

    table_data = []

    for _, row in (
        df.iterrows()
    ):

        table_data.append(
            [
                pretty_name(
                    row[
                        "experiment"
                    ]
                ),

                (
                    "-"
                    if pd.isna(
                        row[
                            "best_epoch"
                        ]
                    )
                    else str(
                        int(
                            row[
                                "best_epoch"
                            ]
                        )
                    )
                ),

                metric_text(
                    row[
                        "best_val_dice"
                    ]
                ),

                metric_text(
                    row[
                        "best_val_iou"
                    ]
                ),

                metric_text(
                    row[
                        "test_dice"
                    ]
                ),

                metric_text(
                    row[
                        "test_iou"
                    ]
                ),

                metric_text(
                    row[
                        "tissue_test_dice"
                    ]
                ),

                metric_text(
                    row[
                        "tissue_test_iou"
                    ]
                ),
            ]
        )

    table = ax.table(
        cellText=
            table_data,

        colLabels=
            headers,

        cellLoc=
            "center",

        colLoc=
            "center",

        loc=
            "center",

        colWidths=[
            0.24,
            0.08,
            0.10,
            0.10,
            0.10,
            0.10,
            0.11,
            0.11,
        ],
    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        9
    )

    table.scale(
        1.0,
        1.6,
    )

    # Header
    for column in range(
        len(
            headers
        )
    ):

        table[
            (
                0,
                column,
            )
        ].set_text_props(
            weight="bold"
        )

    fig.text(
        0.5,
        0.04,
        (
            "Tissue-specific metrics are especially important "
            "because Healthy masks contain no abnormal region."
        ),
        ha="center",
        fontsize=10,
    )

    pdf.savefig(
        fig,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Page 3: all Tissue Dice
# ============================================================


def draw_tissue_metric_bar(
    pdf,
    df,
    metric,
    title,
    ylabel,
):
    working_df = (
        df.copy()
    )

    working_df[
        metric
    ] = pd.to_numeric(
        working_df[
            metric
        ],
        errors="coerce",
    )

    working_df = (
        working_df.dropna(
            subset=[
                metric
            ]
        )
    )

    labels = [
        pretty_name(
            name
        )
        for name in
        working_df[
            "experiment"
        ]
    ]

    values = (
        working_df[
            metric
        ].values
    )

    fig, ax = plt.subplots(
        figsize=(
            15,
            7,
        )
    )

    positions = np.arange(
        len(
            values
        )
    )

    bars = ax.bar(
        positions,
        values,
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.set_ylim(
        bottom=0
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width()
            / 2,

            bar.get_height()
            + 0.01,

            f"{value:.3f}",

            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()

    pdf.savefig(
        fig,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Paired augmentation comparison
# ============================================================


def get_experiment_metric(
    df,
    experiment,
    metric,
):
    rows = df[
        df[
            "experiment"
        ]
        == experiment
    ]

    if len(
        rows
    ) == 0:
        return np.nan

    return pd.to_numeric(
        rows.iloc[
            0
        ][
            metric
        ],
        errors="coerce",
    )


def draw_paired_comparison(
    pdf,
    df,
    metric,
    title,
    ylabel,
):
    augmentation_values = []

    augmentation_os_values = []

    labels = []

    for method in (
        AUGMENTATION_METHODS
    ):

        labels.append(
            pretty_name(
                method
            )
        )

        augmentation_values.append(
            get_experiment_metric(
                df,
                method,
                metric,
            )
        )

        augmentation_os_values.append(
            get_experiment_metric(
                df,
                (
                    f"{method}"
                    f"_oversampling"
                ),
                metric,
            )
        )

    x = np.arange(
        len(
            AUGMENTATION_METHODS
        )
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(
            13,
            7,
        )
    )

    bars_aug = ax.bar(
        x - width / 2,
        augmentation_values,
        width,
        label="Augmentation only",
    )

    bars_aug_os = ax.bar(
        x + width / 2,
        augmentation_os_values,
        width,
        label=(
            "Augmentation + Oversampling"
        ),
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.set_ylim(
        bottom=0
    )

    for bars in [
        bars_aug,
        bars_aug_os,
    ]:

        for bar in bars:

            height = (
                bar.get_height()
            )

            if np.isnan(
                height
            ):
                continue

            ax.text(
                bar.get_x()
                + bar.get_width()
                / 2,

                height
                + 0.01,

                f"{height:.3f}",

                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()

    pdf.savefig(
        fig,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Baseline / Oversampling summary
# ============================================================


def draw_main_comparison(
    pdf,
    df,
):
    experiments = [
        "baseline",
        "oversampling",
    ]

    labels = [
        "Baseline",
        "Oversampling",
    ]

    dice_values = [
        get_experiment_metric(
            df,
            experiment,
            "tissue_test_dice",
        )
        for experiment
        in experiments
    ]

    iou_values = [
        get_experiment_metric(
            df,
            experiment,
            "tissue_test_iou",
        )
        for experiment
        in experiments
    ]

    x = np.arange(
        len(
            labels
        )
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    bars_dice = ax.bar(
        x - width / 2,
        dice_values,
        width,
        label="Tissue Dice",
    )

    bars_iou = ax.bar(
        x + width / 2,
        iou_values,
        width,
        label="Tissue IoU",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Baseline vs Oversampling"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.set_ylim(
        bottom=0
    )

    for bars in [
        bars_dice,
        bars_iou,
    ]:

        for bar in bars:

            value = (
                bar.get_height()
            )

            if np.isnan(
                value
            ):
                continue

            ax.text(
                bar.get_x()
                + bar.get_width()
                / 2,

                value
                + 0.01,

                f"{value:.3f}",

                ha="center",
                fontsize=9,
            )

    fig.tight_layout()

    pdf.savefig(
        fig,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Final summary page
# ============================================================


def draw_summary_page(
    pdf,
    df,
):
    fig, ax = plt.subplots(
        figsize=(
            11.69,
            8.27,
        )
    )

    ax.axis(
        "off"
    )

    tissue_dice = pd.to_numeric(
        df[
            "tissue_test_dice"
        ],
        errors="coerce",
    )

    tissue_iou = pd.to_numeric(
        df[
            "tissue_test_iou"
        ],
        errors="coerce",
    )

    lines = [
        "Result Summary",
        "",
    ]

    if tissue_dice.notna().any():

        best_dice_index = (
            tissue_dice.idxmax()
        )

        best_row = (
            df.loc[
                best_dice_index
            ]
        )

        lines.extend(
            [
                (
                    "Highest observed "
                    "Tissue Test Dice:"
                ),
                (
                    f"  {pretty_name(best_row['experiment'])}"
                    f" = "
                    f"{best_row['tissue_test_dice']:.4f}"
                ),
                "",
            ]
        )

    if tissue_iou.notna().any():

        best_iou_index = (
            tissue_iou.idxmax()
        )

        best_row = (
            df.loc[
                best_iou_index
            ]
        )

        lines.extend(
            [
                (
                    "Highest observed "
                    "Tissue Test IoU:"
                ),
                (
                    f"  {pretty_name(best_row['experiment'])}"
                    f" = "
                    f"{best_row['tissue_test_iou']:.4f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            (
                "These test-set maxima are shown "
                "for descriptive reporting only."
            ),
            (
                "The test set should not be used "
                "to select the final training method."
            ),
            "",
            (
                "Compare validation results first, "
                "then use the test set for final evaluation."
            ),
        ]
    )

    ax.text(
        0.08,
        0.90,
        "\n".join(
            lines
        ),
        va="top",
        fontsize=14,
        linespacing=1.6,
    )

    pdf.savefig(
        fig,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Main
# ============================================================


def create_augmentation_ablation_report(
    summary_csv,
    output_pdf,
    model_name="ResNet34 U-Net",
):
    df = load_ablation_results(
        summary_csv
    )

    df = add_experiment_group(
        df
    )

    output_pdf = Path(
        output_pdf
    )

    output_pdf.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loaded experiments: "
        f"{len(df)}"
    )

    print(
        f"Output PDF: "
        f"{output_pdf}"
    )

    with PdfPages(
        output_pdf
    ) as pdf:

        # Page 1
        draw_overview_page(
            pdf=pdf,
            df=df,
            model_name=
                model_name,
        )

        # Page 2
        draw_results_table(
            pdf=pdf,
            df=df,
        )

        # Page 3
        draw_tissue_metric_bar(
            pdf=pdf,
            df=df,
            metric=
                "tissue_test_dice",
            title=(
                "Tissue Test Dice - "
                "All Experiments"
            ),
            ylabel=
                "Tissue Test Dice",
        )

        # Page 4
        draw_tissue_metric_bar(
            pdf=pdf,
            df=df,
            metric=
                "tissue_test_iou",
            title=(
                "Tissue Test IoU - "
                "All Experiments"
            ),
            ylabel=
                "Tissue Test IoU",
        )

        # Page 5
        draw_paired_comparison(
            pdf=pdf,
            df=df,
            metric=
                "tissue_test_dice",
            title=(
                "Augmentation vs "
                "Augmentation + Oversampling"
            ),
            ylabel=
                "Tissue Test Dice",
        )

        # Page 6
        draw_paired_comparison(
            pdf=pdf,
            df=df,
            metric=
                "tissue_test_iou",
            title=(
                "Augmentation vs "
                "Augmentation + Oversampling"
            ),
            ylabel=
                "Tissue Test IoU",
        )

        # Page 7
        draw_main_comparison(
            pdf=pdf,
            df=df,
        )

        # Page 8
        draw_summary_page(
            pdf=pdf,
            df=df,
        )

    print(
        "\n========================================"
    )

    print(
        "Augmentation ablation report created"
    )

    print(
        "========================================"
    )

    print(
        output_pdf
    )

    return output_pdf