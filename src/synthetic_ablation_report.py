from pathlib import Path
import shutil
import tempfile

import matplotlib.pyplot as plt
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# CSV
# ============================================================


def load_ablation_results(
    csv_path,
):
    """
    Load synthetic ablation summary CSV.
    """

    csv_path = Path(
        csv_path
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Ablation CSV not found: "
            f"{csv_path}"
        )

    dataframe = pd.read_csv(
        csv_path
    )

    required_columns = {
        "experiment",
        "real_tissue",
        "synthetic_tissue",
        "total_tissue",
        "real_healthy",
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
            dataframe.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing columns in "
            "synthetic ablation CSV: "
            f"{sorted(missing)}"
        )

    return dataframe


# ============================================================
# Utility
# ============================================================


def format_metric(
    value,
    digits=4,
):
    """
    Format metric safely.
    """

    if pd.isna(value):
        return "-"

    try:
        return (
            f"{float(value):.{digits}f}"
        )

    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )


def format_integer(
    value,
):
    if pd.isna(value):
        return "-"

    return str(
        int(value)
    )


def find_best_experiment(
    dataframe,
    metric="tissue_test_dice",
):
    """
    Used only for report highlighting.

    IMPORTANT:
    This must NOT be used for model selection.
    Model selection should be based on validation.
    """

    valid = dataframe.dropna(
        subset=[
            metric
        ]
    )

    if len(valid) == 0:
        return None

    index = (
        valid[
            metric
        ]
        .astype(float)
        .idxmax()
    )

    return dataframe.loc[
        index
    ]


# ============================================================
# Matplotlib figures
# ============================================================


def get_real_plus_synthetic_rows(
    dataframe,
):
    """
    Exclude synthetic-only experiments from
    the line plot.

    Example:
        synthetic_31
            -> include

        synthetic_only_31
            -> exclude
    """

    if (
        "include_real_tissue"
        in dataframe.columns
    ):

        mask = (
            dataframe[
                "include_real_tissue"
            ]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        )

        result = dataframe[
            mask
        ].copy()

    else:

        result = dataframe[
            ~dataframe[
                "experiment"
            ]
            .astype(str)
            .str.contains(
                "only",
                case=False,
                na=False,
            )
        ].copy()

    result = result.sort_values(
        "synthetic_tissue"
    )

    return result


def create_metric_plot(
    dataframe,
    metric,
    ylabel,
    output_path,
):
    """
    Create:
        synthetic image count vs metric

    Synthetic-only is shown as an
    independent point.
    """

    output_path = Path(
        output_path
    )

    normal_rows = (
        get_real_plus_synthetic_rows(
            dataframe
        )
    )

    synthetic_only_rows = dataframe[
        dataframe[
            "experiment"
        ]
        .astype(str)
        .str.contains(
            "synthetic_only",
            case=False,
            na=False,
        )
    ].copy()

    fig, ax = plt.subplots(
        figsize=(
            7.5,
            4.8,
        )
    )

    # --------------------------------------------------------
    # Real + Synthetic
    # --------------------------------------------------------

    if len(
        normal_rows
    ) > 0:

        x = normal_rows[
            "synthetic_tissue"
        ].astype(float)

        y = normal_rows[
            metric
        ].astype(float)

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=(
                "Real Tissue "
                "+ Synthetic Tissue"
            ),
        )

        for _, row in (
            normal_rows.iterrows()
        ):

            if pd.isna(
                row[
                    metric
                ]
            ):
                continue

            ax.annotate(
                (
                    f"{float(row[metric]):.3f}"
                ),
                (
                    float(
                        row[
                            "synthetic_tissue"
                        ]
                    ),
                    float(
                        row[
                            metric
                        ]
                    ),
                ),
                xytext=(
                    0,
                    8,
                ),
                textcoords=(
                    "offset points"
                ),
                ha="center",
                fontsize=8,
            )

    # --------------------------------------------------------
    # Synthetic only
    # --------------------------------------------------------

    if len(
        synthetic_only_rows
    ) > 0:

        x = synthetic_only_rows[
            "synthetic_tissue"
        ].astype(float)

        y = synthetic_only_rows[
            metric
        ].astype(float)

        ax.scatter(
            x,
            y,
            marker="X",
            s=100,
            label="Synthetic Tissue only",
        )

        for _, row in (
            synthetic_only_rows.iterrows()
        ):

            if pd.isna(
                row[
                    metric
                ]
            ):
                continue

            ax.annotate(
                (
                    f"{row['experiment']}\n"
                    f"{float(row[metric]):.3f}"
                ),
                (
                    float(
                        row[
                            "synthetic_tissue"
                        ]
                    ),
                    float(
                        row[
                            metric
                        ]
                    ),
                ),
                xytext=(
                    8,
                    -5,
                ),
                textcoords=(
                    "offset points"
                ),
                fontsize=8,
            )

    ax.set_xlabel(
        "Number of Synthetic Tissue Images"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        f"Synthetic Data Ablation - {ylabel}"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Report styles
# ============================================================


def build_styles():
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "SyntheticTitle",
        parent=styles[
            "Title"
        ],
        alignment=TA_CENTER,
        fontSize=22,
        leading=27,
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        "SyntheticSubtitle",
        parent=styles[
            "Normal"
        ],
        alignment=TA_CENTER,
        fontSize=11,
        leading=15,
        textColor=colors.HexColor(
            "#555555"
        ),
        spaceAfter=16,
    )

    heading_style = ParagraphStyle(
        "SyntheticHeading",
        parent=styles[
            "Heading2"
        ],
        fontSize=15,
        leading=19,
        spaceBefore=5,
        spaceAfter=9,
    )

    body_style = ParagraphStyle(
        "SyntheticBody",
        parent=styles[
            "BodyText"
        ],
        fontSize=10,
        leading=15,
        spaceAfter=7,
    )

    note_style = ParagraphStyle(
        "SyntheticNote",
        parent=styles[
            "BodyText"
        ],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor(
            "#555555"
        ),
        leftIndent=6,
        rightIndent=6,
        spaceAfter=8,
    )

    return {
        "title":
            title_style,

        "subtitle":
            subtitle_style,

        "heading":
            heading_style,

        "body":
            body_style,

        "note":
            note_style,
    }


# ============================================================
# Tables
# ============================================================


def build_experiment_table(
    dataframe,
):
    """
    Experimental configuration table.
    """

    data = [
        [
            "Experiment",
            "Real\nTissue",
            "Synthetic\nTissue",
            "Total\nTissue",
            "Real\nHealthy",
        ]
    ]

    for _, row in (
        dataframe.iterrows()
    ):

        data.append(
            [
                str(
                    row[
                        "experiment"
                    ]
                ),

                format_integer(
                    row[
                        "real_tissue"
                    ]
                ),

                format_integer(
                    row[
                        "synthetic_tissue"
                    ]
                ),

                format_integer(
                    row[
                        "total_tissue"
                    ]
                ),

                format_integer(
                    row[
                        "real_healthy"
                    ]
                ),
            ]
        )

    table = Table(
        data,
        colWidths=[
            55 * mm,
            27 * mm,
            30 * mm,
            27 * mm,
            30 * mm,
        ],
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    colors.HexColor(
                        "#E8EEF7"
                    ),
                ),

                (
                    "TEXTCOLOR",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    colors.HexColor(
                        "#1F2937"
                    ),
                ),

                (
                    "FONTNAME",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    "Helvetica-Bold",
                ),

                (
                    "ALIGN",
                    (
                        1,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    "CENTER",
                ),

                (
                    "VALIGN",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    "MIDDLE",
                ),

                (
                    "GRID",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.4,
                    colors.HexColor(
                        "#B8C2CC"
                    ),
                ),

                (
                    "ROWBACKGROUNDS",
                    (
                        0,
                        1,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    [
                        colors.white,
                        colors.HexColor(
                            "#F8FAFC"
                        ),
                    ],
                ),

                (
                    "TOPPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    6,
                ),

                (
                    "BOTTOMPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    6,
                ),
            ]
        )
    )

    return table


def build_results_table(
    dataframe,
):
    """
    Main result table.
    """

    data = [
        [
            "Experiment",
            "Best\nEpoch",
            "Val\nDice",
            "Val\nIoU",
            "Test\nDice",
            "Test\nIoU",
            "Tissue\nDice",
            "Tissue\nIoU",
        ]
    ]

    for _, row in (
        dataframe.iterrows()
    ):

        data.append(
            [
                str(
                    row[
                        "experiment"
                    ]
                ),

                format_integer(
                    row[
                        "best_epoch"
                    ]
                ),

                format_metric(
                    row[
                        "best_val_dice"
                    ]
                ),

                format_metric(
                    row[
                        "best_val_iou"
                    ]
                ),

                format_metric(
                    row[
                        "test_dice"
                    ]
                ),

                format_metric(
                    row[
                        "test_iou"
                    ]
                ),

                format_metric(
                    row[
                        "tissue_test_dice"
                    ]
                ),

                format_metric(
                    row[
                        "tissue_test_iou"
                    ]
                ),
            ]
        )

    table = Table(
        data,
        colWidths=[
            52 * mm,
            18 * mm,
            22 * mm,
            22 * mm,
            22 * mm,
            22 * mm,
            25 * mm,
            25 * mm,
        ],
        repeatRows=1,
    )

    commands = [
        (
            "BACKGROUND",
            (
                0,
                0,
            ),
            (
                -1,
                0,
            ),
            colors.HexColor(
                "#E8EEF7"
            ),
        ),

        (
            "FONTNAME",
            (
                0,
                0,
            ),
            (
                -1,
                0,
            ),
            "Helvetica-Bold",
        ),

        (
            "ALIGN",
            (
                1,
                0,
            ),
            (
                -1,
                -1,
            ),
            "CENTER",
        ),

        (
            "VALIGN",
            (
                0,
                0,
            ),
            (
                -1,
                -1,
            ),
            "MIDDLE",
        ),

        (
            "GRID",
            (
                0,
                0,
            ),
            (
                -1,
                -1,
            ),
            0.4,
            colors.HexColor(
                "#B8C2CC"
            ),
        ),

        (
            "ROWBACKGROUNDS",
            (
                0,
                1,
            ),
            (
                -1,
                -1,
            ),
            [
                colors.white,
                colors.HexColor(
                    "#F8FAFC"
                ),
            ],
        ),

        (
            "TOPPADDING",
            (
                0,
                0,
            ),
            (
                -1,
                -1,
            ),
            5,
        ),

        (
            "BOTTOMPADDING",
            (
                0,
                0,
            ),
            (
                -1,
                -1,
            ),
            5,
        ),
    ]

    # --------------------------------------------------------
    # Highlight highest Tissue Test Dice
    # For visualization only.
    # Do NOT use test metric for model selection.
    # --------------------------------------------------------

    numeric_values = pd.to_numeric(
        dataframe[
            "tissue_test_dice"
        ],
        errors="coerce",
    )

    if numeric_values.notna().any():

        best_index = (
            numeric_values.idxmax()
        )

        row_position = (
            list(
                dataframe.index
            ).index(
                best_index
            )
            + 1
        )

        commands.append(
            (
                "BACKGROUND",
                (
                    6,
                    row_position,
                ),
                (
                    6,
                    row_position,
                ),
                colors.HexColor(
                    "#DDF3E4"
                ),
            )
        )

        commands.append(
            (
                "FONTNAME",
                (
                    6,
                    row_position,
                ),
                (
                    6,
                    row_position,
                ),
                "Helvetica-Bold",
            )
        )

    table.setStyle(
        TableStyle(
            commands
        )
    )

    return table


# ============================================================
# Optional qualitative comparison
# ============================================================


def find_prediction_image(
    prediction_root,
    experiment,
    category,
    sample,
    filename,
):
    """
    Expected structure:

    prediction_root/
        baseline/
            test/
                Tissue/
                    sample_name/
                        segmentation.png

    If your directory differs, only modify
    this function.
    """

    path = (
        Path(
            prediction_root
        )
        / experiment
        / "test"
        / category
        / sample
        / filename
    )

    if path.exists():
        return path

    return None


def build_qualitative_pages(
    story,
    styles,
    report_cfg,
):
    """
    Optional prediction comparison.

    Example:
        same Tissue image across
        baseline / synthetic_26 /
        synthetic_31 / synthetic_only_31
    """

    qualitative_cfg = (
        report_cfg.get(
            "qualitative",
            {}
        )
    )

    if not qualitative_cfg.get(
        "enabled",
        False,
    ):
        return

    prediction_root = (
        qualitative_cfg.get(
            "prediction_root"
        )
    )

    experiments = (
        qualitative_cfg.get(
            "experiments",
            []
        )
    )

    samples = (
        qualitative_cfg.get(
            "samples",
            []
        )
    )

    if not prediction_root:
        return

    if not experiments:
        return

    if not samples:
        return

    for sample_cfg in samples:

        category = sample_cfg.get(
            "category",
            "Tissue",
        )

        sample = sample_cfg[
            "sample"
        ]

        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                (
                    "Qualitative Comparison - "
                    f"{sample}"
                ),
                styles[
                    "heading"
                ],
            )
        )

        rows = [
            [
                "Experiment",
                "Segmentation",
                "Error Map",
                "Probability Map",
            ]
        ]

        for experiment in experiments:

            segmentation = (
                find_prediction_image(
                    prediction_root,
                    experiment,
                    category,
                    sample,
                    "segmentation.png",
                )
            )

            color_map = (
                find_prediction_image(
                    prediction_root,
                    experiment,
                    category,
                    sample,
                    "color_map.png",
                )
            )

            heat_map = (
                find_prediction_image(
                    prediction_root,
                    experiment,
                    category,
                    sample,
                    "heat_map.png",
                )
            )

            if (
                segmentation is None
                and color_map is None
                and heat_map is None
            ):
                continue

            row = [
                experiment
            ]

            for path in [
                segmentation,
                color_map,
                heat_map,
            ]:

                if path is None:

                    row.append(
                        "Not found"
                    )

                else:

                    row.append(
                        Image(
                            str(path),
                            width=48 * mm,
                            height=48 * mm,
                        )
                    )

            rows.append(
                row
            )

        if len(
            rows
        ) == 1:
            continue

        table = Table(
            rows,
            colWidths=[
                45 * mm,
                52 * mm,
                52 * mm,
                52 * mm,
            ],
            repeatRows=1,
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            0,
                        ),
                        colors.HexColor(
                            "#E8EEF7"
                        ),
                    ),

                    (
                        "FONTNAME",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            0,
                        ),
                        "Helvetica-Bold",
                    ),

                    (
                        "ALIGN",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        "CENTER",
                    ),

                    (
                        "VALIGN",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        "MIDDLE",
                    ),

                    (
                        "GRID",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        0.4,
                        colors.HexColor(
                            "#B8C2CC"
                        ),
                    ),

                    (
                        "TOPPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        5,
                    ),

                    (
                        "BOTTOMPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        5,
                    ),
                ]
            )
        )

        story.append(
            table
        )


# ============================================================
# Page number
# ============================================================


def draw_page_number(
    canvas,
    doc,
):
    canvas.saveState()

    canvas.setFont(
        "Helvetica",
        8,
    )

    canvas.setFillColor(
        colors.HexColor(
            "#666666"
        )
    )

    canvas.drawRightString(
        doc.pagesize[0]
        - 15 * mm,
        10 * mm,
        f"Page {doc.page}",
    )

    canvas.restoreState()


# ============================================================
# PDF
# ============================================================


def create_synthetic_ablation_report(
    csv_path,
    output_pdf,
    report_cfg=None,
):
    """
    Create PDF report for synthetic data ablation.
    """

    if report_cfg is None:
        report_cfg = {}

    dataframe = (
        load_ablation_results(
            csv_path
        )
    )

    output_pdf = Path(
        output_pdf
    )

    output_pdf.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    styles = build_styles()

    # Temporary files for plots
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=(
                "synthetic_report_"
            )
        )
    )

    try:

        dice_plot = (
            temp_dir
            / "tissue_dice.png"
        )

        iou_plot = (
            temp_dir
            / "tissue_iou.png"
        )

        create_metric_plot(
            dataframe=dataframe,
            metric=(
                "tissue_test_dice"
            ),
            ylabel=(
                "Tissue Test Dice"
            ),
            output_path=dice_plot,
        )

        create_metric_plot(
            dataframe=dataframe,
            metric=(
                "tissue_test_iou"
            ),
            ylabel=(
                "Tissue Test IoU"
            ),
            output_path=iou_plot,
        )

        # ----------------------------------------------------
        # Landscape A4 because results table is wide
        # ----------------------------------------------------

        doc = SimpleDocTemplate(
            str(
                output_pdf
            ),
            pagesize=landscape(
                A4
            ),
            rightMargin=15 * mm,
            leftMargin=15 * mm,
            topMargin=15 * mm,
            bottomMargin=17 * mm,
            title=(
                "Synthetic Data "
                "Ablation Study"
            ),
        )

        story = []

        # ====================================================
        # PAGE 1
        # ====================================================

        story.append(
            Spacer(
                1,
                15 * mm,
            )
        )

        story.append(
            Paragraph(
                "Synthetic Data Ablation Study",
                styles[
                    "title"
                ],
            )
        )

        model_name = (
            report_cfg.get(
                "model_name",
                "ResNet34 U-Net",
            )
        )

        story.append(
            Paragraph(
                (
                    f"Model: {model_name}"
                ),
                styles[
                    "subtitle"
                ],
            )
        )

        story.append(
            Spacer(
                1,
                4 * mm,
            )
        )

        story.append(
            Paragraph(
                "Experimental Design",
                styles[
                    "heading"
                ],
            )
        )

        story.append(
            Paragraph(
                (
                    "Synthetic Tissue images are added "
                    "only to the training set. "
                    "Validation and test sets contain "
                    "real images only."
                ),
                styles[
                    "body"
                ],
            )
        )

        story.append(
            Paragraph(
                (
                    "The synthetic-only condition "
                    "uses no real Tissue images during "
                    "training, while real Healthy images "
                    "remain in the training set."
                ),
                styles[
                    "body"
                ],
            )
        )

        story.append(
            Spacer(
                1,
                4 * mm,
            )
        )

        story.append(
            build_experiment_table(
                dataframe
            )
        )

        story.append(
            Spacer(
                1,
                8 * mm,
            )
        )

        story.append(
            Paragraph(
                (
                    "Important: test-set metrics are "
                    "reported for final evaluation and "
                    "should not be used to select the "
                    "training condition."
                ),
                styles[
                    "note"
                ],
            )
        )

        # ====================================================
        # PAGE 2
        # ====================================================

        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                "Quantitative Results",
                styles[
                    "heading"
                ],
            )
        )

        story.append(
            Paragraph(
                (
                    "Tissue-specific Dice and IoU "
                    "are shown separately because "
                    "overall metrics can be strongly "
                    "affected by Healthy images with "
                    "empty ground-truth masks."
                ),
                styles[
                    "body"
                ],
            )
        )

        story.append(
            Spacer(
                1,
                3 * mm,
            )
        )

        story.append(
            build_results_table(
                dataframe
            )
        )

        best_row = (
            find_best_experiment(
                dataframe,
                metric=(
                    "tissue_test_dice"
                ),
            )
        )

        if best_row is not None:

            story.append(
                Spacer(
                    1,
                    8 * mm,
                )
            )

            story.append(
                Paragraph(
                    (
                        "Highest observed Tissue "
                        "Test Dice: "
                        f"<b>{best_row['experiment']}</b> "
                        f"({float(best_row['tissue_test_dice']):.4f}). "
                        "This highlight is descriptive "
                        "only and must not be used for "
                        "model selection."
                    ),
                    styles[
                        "note"
                    ],
                )
            )

        # ====================================================
        # PAGE 3
        # ====================================================

        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                "Effect of Synthetic Data Volume",
                styles[
                    "heading"
                ],
            )
        )

        story.append(
            Paragraph(
                (
                    "The line represents experiments "
                    "that use real Tissue together with "
                    "Synthetic Tissue. "
                    "Synthetic-only is plotted "
                    "separately because it represents "
                    "a different training condition."
                ),
                styles[
                    "body"
                ],
            )
        )

        chart_table = Table(
            [
                [
                    Image(
                        str(
                            dice_plot
                        ),
                        width=125 * mm,
                        height=80 * mm,
                    ),
                    Image(
                        str(
                            iou_plot
                        ),
                        width=125 * mm,
                        height=80 * mm,
                    ),
                ]
            ],
            colWidths=[
                130 * mm,
                130 * mm,
            ],
        )

        chart_table.setStyle(
            TableStyle(
                [
                    (
                        "VALIGN",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        "TOP",
                    ),

                    (
                        "ALIGN",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        "CENTER",
                    ),
                ]
            )
        )

        story.append(
            chart_table
        )

        # ====================================================
        # Optional qualitative pages
        # ====================================================

        build_qualitative_pages(
            story=story,
            styles=styles,
            report_cfg=report_cfg,
        )

        # ====================================================
        # Build
        # ====================================================

        doc.build(
            story,
            onFirstPage=
                draw_page_number,
            onLaterPages=
                draw_page_number,
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )

    print(
        "\nSynthetic ablation report created:"
    )

    print(
        output_pdf
    )

    return output_pdf