import random

import cv2
import numpy as np

from src.iris_geometry import (
    create_valid_iris_mask,
    detect_iris,
    detect_pupil,
)

from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def get_image_paths(
    directory,
):
    """
    Get image file paths from directory.
    """

    directory = Path(
        directory
    )

    if not directory.exists():

        raise FileNotFoundError(
            f"Directory not found: "
            f"{directory}"
        )

    image_paths = sorted([
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ])

    return image_paths


def load_grayscale(
    path,
):
    """
    Load grayscale image.
    """

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:

        raise RuntimeError(
            f"Cannot read image: "
            f"{path}"
        )

    return image


def extract_lesion_patch(
    tissue_image,
    tissue_mask,
    padding=3,
):
    """
    Extract lesion image and binary mask
    using Tissue GT.

    Returns:
        lesion_image
        lesion_mask
    """

    mask_binary = (
        tissue_mask > 0
    ).astype(
        np.uint8
    )

    ys, xs = np.where(
        mask_binary > 0
    )

    if len(xs) == 0:

        raise RuntimeError(
            "Tissue mask is empty."
        )

    height, width = (
        tissue_image.shape
    )

    x1 = max(
        int(xs.min()) - padding,
        0,
    )

    x2 = min(
        int(xs.max()) + padding + 1,
        width,
    )

    y1 = max(
        int(ys.min()) - padding,
        0,
    )

    y2 = min(
        int(ys.max()) + padding + 1,
        height,
    )

    lesion_image = (
        tissue_image[
            y1:y2,
            x1:x2,
        ].copy()
    )

    lesion_mask = (
        mask_binary[
            y1:y2,
            x1:x2,
        ].copy()
    )

    return (
        lesion_image,
        lesion_mask,
    )


def find_valid_location(
    lesion_mask,
    valid_iris_mask,
    pupil_center,
    maximum_attempts=2000,
    num_valid_candidates=100,
    center_bias_strength=4.0,
):
    """
    Find a valid lesion position with preference
    toward the central / inner iris region.

    Conditions:
        - lesion must remain outside pupil
        - lesion must remain inside iris
        - positions closer to the pupil are preferred

    Parameters
    ----------
    center_bias_strength:
        0.0 -> almost uniform
        2.0 -> mild center preference
        4.0 -> recommended
        6.0+ -> strong center preference
    """

    image_height, image_width = (
        valid_iris_mask.shape
    )

    patch_height, patch_width = (
        lesion_mask.shape
    )

    if (
        patch_height > image_height
        or patch_width > image_width
    ):
        raise RuntimeError(
            "Lesion patch is larger "
            "than target image."
        )

    lesion_pixels = (
        lesion_mask > 0
    )

    # ========================================================
    # Lesion centroid inside its cropped patch
    # ========================================================

    ys, xs = np.where(
        lesion_pixels
    )

    if len(xs) == 0:
        raise RuntimeError(
            "Lesion mask is empty."
        )

    lesion_center_x = float(
        xs.mean()
    )

    lesion_center_y = float(
        ys.mean()
    )

    # ========================================================
    # Collect valid candidate locations
    # ========================================================

    candidates = []

    for _ in range(
        maximum_attempts
    ):

        x = random.randint(
            0,
            image_width
            - patch_width,
        )

        y = random.randint(
            0,
            image_height
            - patch_height,
        )

        valid_region = (
            valid_iris_mask[
                y:
                y + patch_height,

                x:
                x + patch_width,
            ]
        )

        # ----------------------------------------------------
        # Entire lesion must be inside valid iris region
        # ----------------------------------------------------

        if not np.all(
            valid_region[
                lesion_pixels
            ] > 0
        ):
            continue

        # ----------------------------------------------------
        # Actual lesion centroid after placement
        # ----------------------------------------------------

        placed_center_x = (
            x
            + lesion_center_x
        )

        placed_center_y = (
            y
            + lesion_center_y
        )

        # Distance from pupil center
        distance = np.sqrt(
            (
                placed_center_x
                - pupil_center[0]
            ) ** 2
            +
            (
                placed_center_y
                - pupil_center[1]
            ) ** 2
        )

        candidates.append(
            (
                x,
                y,
                distance,
            )
        )

        if (
            len(candidates)
            >= num_valid_candidates
        ):
            break

    if len(candidates) == 0:
        raise RuntimeError(
            "Could not find a valid "
            "lesion placement."
        )

    # ========================================================
    # Center-biased selection
    #
    # Shorter distance from pupil -> larger weight
    # ========================================================

    distances = np.array(
        [
            candidate[2]
            for candidate
            in candidates
        ],
        dtype=np.float64,
    )

    min_distance = (
        distances.min()
    )

    max_distance = (
        distances.max()
    )

    if (
        max_distance
        - min_distance
        < 1e-8
    ):

        selected_index = (
            random.randrange(
                len(candidates)
            )
        )

    else:

        normalized_distance = (
            (
                distances
                - min_distance
            )
            /
            (
                max_distance
                - min_distance
            )
        )

        # Distance 0 -> high probability
        # Distance 1 -> low probability
        weights = np.exp(
            -float(
                center_bias_strength
            )
            * normalized_distance
        )

        weights = (
            weights
            / weights.sum()
        )

        selected_index = (
            np.random.choice(
                len(candidates),
                p=weights,
            )
        )

    x, y, _ = (
        candidates[
            selected_index
        ]
    )

    return (
        x,
        y,
    )

def paste_lesion(
    healthy_image,
    lesion_image,
    lesion_mask,
    valid_iris_mask,
    x,
    y,
    feather_size=5,
):
    """
    Paste lesion into Healthy image.

    Important:
        Blending is also restricted to
        the valid iris region.
    """

    synthetic = (
        healthy_image.copy()
    )

    patch_height, patch_width = (
        lesion_mask.shape
    )

    target_region = synthetic[
        y:y + patch_height,
        x:x + patch_width,
    ]

    valid_region = valid_iris_mask[
        y:y + patch_height,
        x:x + patch_width,
    ]

    # ========================================================
    # Original lesion alpha
    # ========================================================

    binary_alpha = (
        lesion_mask > 0
    ).astype(
        np.float32
    )

    # ========================================================
    # Feather
    # ========================================================

    if feather_size > 1:

        if feather_size % 2 == 0:
            feather_size += 1

        alpha = cv2.GaussianBlur(
            binary_alpha,
            (
                feather_size,
                feather_size,
            ),
            0,
        )

    else:

        alpha = binary_alpha

    alpha = np.clip(
        alpha,
        0.0,
        1.0,
    )

    # ========================================================
    # CRITICAL:
    # Never allow blending outside valid iris
    # ========================================================

    alpha = (
        alpha
        * (
            valid_region > 0
        ).astype(
            np.float32
        )
    )

    # ========================================================
    # Blend
    # ========================================================

    donor = lesion_image.astype(
        np.float32
    )

    target = target_region.astype(
        np.float32
    )

    blended = (
        alpha * donor
        +
        (
            1.0 - alpha
        ) * target
    )

    synthetic[
        y:y + patch_height,
        x:x + patch_width,
    ] = np.clip(
        blended,
        0,
        255,
    ).astype(
        np.uint8
    )

    # ========================================================
    # Synthetic GT
    # ========================================================

    synthetic_mask = np.zeros(
        healthy_image.shape,
        dtype=np.uint8,
    )

    final_mask = (
        (lesion_mask > 0)
        &
        (valid_region > 0)
    )

    target_mask = synthetic_mask[
        y:y + patch_height,
        x:x + patch_width,
    ]

    target_mask[
        final_mask
    ] = 255

    return (
        synthetic,
        synthetic_mask,
    )

def create_synthetic_tissue(
    tissue_image,
    tissue_mask,
    healthy_image,
    maximum_attempts=1000,
    pupil_margin=5,
    iris_margin=5,
    feather_size=5,
):
    """
    Simple constrained copy-paste.

    Rules:
        - lesion must be outside pupil
        - lesion must be inside iris
        - entire lesion must satisfy both
    """

    # ========================================================
    # 1. Extract lesion from Tissue
    # ========================================================

    lesion_image, lesion_mask = (
        extract_lesion_patch(
            tissue_image,
            tissue_mask,
        )
    )

    # ========================================================
    # 2. Detect Healthy pupil
    # ========================================================

    (
        pupil_center,
        pupil_radius,
    ) = detect_pupil(
        healthy_image
    )

    # ========================================================
    # 3. Detect Healthy iris
    # ========================================================

    (
        iris_center,
        iris_radius,
    ) = detect_iris(
        healthy_image,
        pupil_center,
        pupil_radius,
    )

    # ========================================================
    # 4. Valid placement region
    # ========================================================

    valid_iris_mask = create_valid_iris_mask(
        image_shape=healthy_image.shape,
    
        pupil_center=pupil_center,
        pupil_radius=pupil_radius,
    
        iris_center=iris_center,
        iris_radius=iris_radius,
    
        pupil_margin_ratio=0.15,
        iris_margin_ratio=0.12,
    )

    # ========================================================
    # 5. Find valid random location
    # ========================================================

    x, y = find_valid_location(
        lesion_mask=lesion_mask,
        valid_iris_mask=valid_iris_mask,
    
        pupil_center=pupil_center,
    
        maximum_attempts=
            maximum_attempts,
    
        num_valid_candidates=100,
    
        center_bias_strength=7.0,
    )
    # ========================================================
    # 6. Copy-paste
    # ========================================================

    synthetic, synthetic_mask = paste_lesion(
        healthy_image=healthy_image,
        lesion_image=lesion_image,
        lesion_mask=lesion_mask,
    
        valid_iris_mask=valid_iris_mask,
    
        x=x,
        y=y,
    
        feather_size=feather_size,
    )

    metadata = {
        "x":
            x,

        "y":
            y,

        "pupil_center":
            pupil_center,

        "pupil_radius":
            pupil_radius,

        "iris_center":
            iris_center,

        "iris_radius":
            iris_radius,
    }

    return (
        synthetic,
        synthetic_mask,
        metadata,
    )

from pathlib import Path
import random

import cv2


def generate_synthetic_dataset(
    tissue_image_dir,
    tissue_mask_dir,
    healthy_image_dir,
    output_image_dir,
    output_mask_dir,
    num_images,
    seed=42,
    maximum_attempts=1000,
    pupil_margin=5,
    iris_margin=5,
    feather_size=5,
):
    """
    Generate synthetic Tissue images.

    Rules:
        - Tissue donor can be reused.
        - Each Healthy image is used at most once.
        - Lesion must be outside pupil.
        - Lesion must be inside iris.
    """

    random.seed(seed)

    tissue_image_dir = Path(
        tissue_image_dir
    )

    tissue_mask_dir = Path(
        tissue_mask_dir
    )

    healthy_image_dir = Path(
        healthy_image_dir
    )

    output_image_dir = Path(
        output_image_dir
    )

    output_mask_dir = Path(
        output_mask_dir
    )

    output_image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Get images
    # ========================================================

    tissue_paths = get_image_paths(
        tissue_image_dir
    )

    healthy_paths = get_image_paths(
        healthy_image_dir
    )

    if len(tissue_paths) == 0:
        raise RuntimeError(
            "No Tissue images found."
        )

    if len(healthy_paths) == 0:
        raise RuntimeError(
            "No Healthy images found."
        )

    # ========================================================
    # One Healthy image can be used only once
    # ========================================================

    if num_images > len(
        healthy_paths
    ):
        raise ValueError(
            f"Requested num_images={num_images}, "
            f"but only {len(healthy_paths)} "
            f"unique Healthy images are available."
        )

    # Randomize Healthy order,
    # but do not reuse the same image
    random.shuffle(
        healthy_paths
    )

    generated = 0
    skipped = 0

    # ========================================================
    # Process Healthy images one by one
    # ========================================================

    for healthy_path in healthy_paths:

        if generated >= num_images:
            break

        healthy_image = load_grayscale(
            healthy_path
        )

        success = False

        # ----------------------------------------------------
        # Try several Tissue donors for this Healthy image
        # ----------------------------------------------------

        tissue_candidates = (
            tissue_paths.copy()
        )

        random.shuffle(
            tissue_candidates
        )

        for tissue_path in tissue_candidates:

            mask_path = (
                tissue_mask_dir
                / f"{tissue_path.stem}.png"
            )

            if not mask_path.exists():

                print(
                    f"[WARNING] "
                    f"Mask not found: "
                    f"{mask_path}"
                )

                continue

            tissue_image = load_grayscale(
                tissue_path
            )

            tissue_mask = load_grayscale(
                mask_path
            )

            # ------------------------------------------------
            # Create constrained copy-paste
            # ------------------------------------------------

            try:

                (
                    synthetic,
                    synthetic_mask,
                    metadata,
                ) = create_synthetic_tissue(
                    tissue_image=
                        tissue_image,

                    tissue_mask=
                        tissue_mask,

                    healthy_image=
                        healthy_image,

                    maximum_attempts=
                        maximum_attempts,

                    pupil_margin=
                        pupil_margin,

                    iris_margin=
                        iris_margin,

                    feather_size=
                        feather_size,
                )

            except RuntimeError as error:

                print(
                    f"[WARNING] "
                    f"Healthy="
                    f"{healthy_path.name} | "
                    f"Tissue="
                    f"{tissue_path.name} | "
                    f"{error}"
                )

                continue

            # =================================================
            # Save
            # =================================================

            filename = (
                f"synthetic_tissue_"
                f"{generated:04d}.png"
            )

            image_output_path = (
                output_image_dir
                / filename
            )

            mask_output_path = (
                output_mask_dir
                / filename
            )

            cv2.imwrite(
                str(
                    image_output_path
                ),
                synthetic,
            )

            cv2.imwrite(
                str(
                    mask_output_path
                ),
                synthetic_mask,
            )

            print(
                f"[{generated + 1}/"
                f"{num_images}] "
                f"{filename} | "
                f"Healthy="
                f"{healthy_path.name} | "
                f"Tissue="
                f"{tissue_path.name} | "
                f"pupil="
                f"{metadata['pupil_center']} | "
                f"iris="
                f"{metadata['iris_center']} | "
                f"position="
                f"({metadata['x']}, "
                f"{metadata['y']})"
            )

            generated += 1

            success = True

            # IMPORTANT:
            # This Healthy image is finished.
            # Do not generate another image from it.
            break

        # ====================================================
        # No Tissue donor worked for this Healthy image
        # ====================================================

        if not success:

            skipped += 1

            print(
                f"[SKIP] "
                f"Could not place lesion on: "
                f"{healthy_path.name}"
            )

    # ========================================================
    # Result
    # ========================================================

    print(
        "\n"
        "======================================"
    )

    print(
        "Synthetic generation finished"
    )

    print(
        "======================================"
    )

    print(
        f"Requested             : "
        f"{num_images}"
    )

    print(
        f"Generated             : "
        f"{generated}"
    )

    print(
        f"Unique Healthy used   : "
        f"{generated}"
    )

    print(
        f"Skipped Healthy       : "
        f"{skipped}"
    )

    print(
        f"Healthy available     : "
        f"{len(healthy_paths)}"
    )

    if generated < num_images:

        print(
            "\n[WARNING] "
            f"Only {generated}/"
            f"{num_images} images "
            f"were generated."
        )