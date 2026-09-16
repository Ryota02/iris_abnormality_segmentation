from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


DEFAULT_CATEGORIES = [
    # "Geometry",
    "Tissue",
    "Healthy",
]

def collect_prediction_samples(
    prediction_root,
    categories=None,
    experiment_name=None,
):
    """
    Collect prediction result folders.
    """

    prediction_root = Path(
        prediction_root
    )

    if categories is None:
        categories = (
            DEFAULT_CATEGORIES
        )

    samples = []

    for category in categories:

        category_dir = (
            prediction_root
            / category
        )

        if not category_dir.exists():

            print(
                f"[WARNING] "
                f"Not found: "
                f"{category_dir}"
            )

            continue

        for sample_dir in sorted(
            category_dir.iterdir()
        ):

            if not sample_dir.is_dir():
                continue

            paths = {
                "original":
                    sample_dir
                    / "original.png",

                "gt":
                    sample_dir
                    / "ground_truth.png",

                "segmentation":
                    sample_dir
                    / "segmentation.png",

                "color_map":
                    sample_dir
                    / "color_map.png",

                "heat_map":
                    sample_dir
                    / "heat_map.png",
            }

            missing_files = [
                path
                for path in paths.values()
                if not path.exists()
            ]

            if missing_files:

                print(
                    f"[WARNING] "
                    f"Missing files in: "
                    f"{sample_dir}"
                )

                continue

            samples.append(
                {
                    "experiment":
                        experiment_name,

                    "category":
                        category,

                    "name":
                        sample_dir.name,

                    **paths,
                }
            )

    return samples


def load_sample_images(
    sample,
):
    """
    Load all visualization images
    for one sample.
    """

    return {
        "original":
            Image.open(
                sample["original"]
            ).convert("L"),

        "gt":
            Image.open(
                sample["gt"]
            ).convert("L"),

        "segmentation":
            Image.open(
                sample["segmentation"]
            ).convert("L"),

        "color_map":
            Image.open(
                sample["color_map"]
            ).convert("RGB"),

        "heat_map":
            Image.open(
                sample["heat_map"]
            ).convert("RGB"),
    }


def draw_sample_row(
    axes,
    row,
    sample,
):
    """
    Draw one sample in one PDF row.

    Columns:
        1. Original
        2. Ground Truth
        3. Prediction
        4. Segmentation Error Map
        5. Probability Heat Map
    """

    images = load_sample_images(
        sample
    )

    # ========================================================
    # Original
    # ========================================================

    axes[
        row,
        0
    ].imshow(
        images["original"],
        cmap="gray",
    )

    axes[
        row,
        0
    ].set_title(
        "Original"
    )

    # ========================================================
    # Ground Truth
    # ========================================================

    axes[
        row,
        1
    ].imshow(
        images["gt"],
        cmap="gray",
        vmin=0,
        vmax=255,
    )

    axes[
        row,
        1
    ].set_title(
        "Ground Truth"
    )

    # ========================================================
    # Prediction
    # ========================================================

    axes[
        row,
        2
    ].imshow(
        images["segmentation"],
        cmap="gray",
        vmin=0,
        vmax=255,
    )

    axes[
        row,
        2
    ].set_title(
        "Prediction"
    )

    # ========================================================
    # Segmentation Error Map
    # ========================================================

    axes[
        row,
        3
    ].imshow(
        images["color_map"]
    )

    axes[
        row,
        3
    ].set_title(
        "Segmentation Error Map"
    )

    # ========================================================
    # Probability Heat Map
    # ========================================================

    axes[
        row,
        4
    ].imshow(
        images["heat_map"]
    )

    axes[
        row,
        4
    ].set_title(
        "Probability Heat Map"
    )

    # ========================================================
    # Category / filename
    # ========================================================

    experiment = sample.get(
        "experiment",
        None,
    )
    
    if experiment is None:
    
        label = (
            f"{sample['category']}\n"
            f"{sample['name']}"
        )
    
    else:
    
        label = (
            f"{experiment}\n"
            f"{sample['category']}\n"
            f"{sample['name']}"
        )
    
    axes[
        row,
        0
    ].set_ylabel(
        label,
        fontsize=9,
    )

    # ========================================================
    # Hide axes
    # ========================================================

    for col in range(5):

        axes[
            row,
            col
        ].axis(
            "off"
        )


def create_prediction_pdf(
    samples,
    output_pdf,
    rows_per_page=4,
    title=None,
):
    """
    Create multi-page PDF
    containing prediction results.
    """

    output_pdf = Path(
        output_pdf
    )

    output_pdf.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if len(samples) == 0:

        raise ValueError(
            "No samples provided."
        )

    with PdfPages(
        output_pdf
    ) as pdf:

        for start in range(
            0,
            len(samples),
            rows_per_page,
        ):

            batch = samples[
                start:
                start
                + rows_per_page
            ]

            n_rows = len(
                batch
            )

            fig, axes = plt.subplots(
                n_rows,
                5,
                figsize=(
                    20,
                    4 * n_rows,
                ),
            )

            # 1 row only
            if n_rows == 1:

                axes = axes.reshape(
                    1,
                    -1,
                )

            # =================================================
            # Draw rows
            # =================================================

            for row, sample in enumerate(
                batch
            ):

                draw_sample_row(
                    axes=axes,
                    row=row,
                    sample=sample,
                )

            page_start = (
                start + 1
            )

            page_end = (
                start
                + len(batch)
            )

            # =================================================
            # Page title
            # =================================================

            if title is None:
                title = (
                    "Iris Abnormality "
                    "Segmentation Results"
                )
            
            fig.suptitle(
                f"{title}\n"
                "Original | Ground Truth | "
                "Prediction | "
                "Segmentation Error Map | "
                "Probability Heat Map\n"
                "Error Map: "
                "Green=Correct (TP), "
                "Blue=Extra (FP), "
                "Red=Missed (FN)",
                fontsize=14,
            )
            plt.tight_layout(
                rect=[
                    0,
                    0,
                    1,
                    0.92,
                ]
            )

            # =================================================
            # Save PDF page
            # =================================================

            pdf.savefig(
                fig,
                bbox_inches="tight",
            )

            plt.close(
                fig
            )

            print(
                f"Saved page: "
                f"{page_start}-"
                f"{page_end}"
            )

    print(
        f"\nSaved PDF:"
        f"\n{output_pdf}"
    )


def generate_prediction_report(
    prediction_root,
    output_pdf,
    categories=None,
    rows_per_page=4,
    title=None,
):
    """
    High-level function called from script.
    """

    samples = (
        collect_prediction_samples(
            prediction_root=
                prediction_root,
            categories=
                categories,
        )
    )

    print(
        f"Total samples: "
        f"{len(samples)}"
    )

    if len(samples) == 0:

        print(
            "[WARNING] "
            "No valid samples found."
        )

        return

    create_prediction_pdf(
        samples=samples,
        output_pdf=output_pdf,
        rows_per_page=
            rows_per_page,
        title=title,
    )