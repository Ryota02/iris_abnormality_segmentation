import cv2
import numpy as np


def detect_pupil(
    image,
):
    """
    Detect pupil from grayscale iris image.

    Returns:
        center: (x, y)
        radius: int
    """

    if image.ndim != 2:
        raise ValueError(
            "Grayscale image is required."
        )

    height, width = image.shape

    blurred = cv2.GaussianBlur(
        image,
        (9, 9),
        2,
    )

    # ========================================================
    # Search only around central region
    # ========================================================

    y1 = int(
        height * 0.20
    )

    y2 = int(
        height * 0.80
    )

    x1 = int(
        width * 0.20
    )

    x2 = int(
        width * 0.80
    )

    center_region = blurred[
        y1:y2,
        x1:x2,
    ]

    threshold_value = np.percentile(
        center_region,
        20,
    )

    binary = (
        blurred
        <= threshold_value
    ).astype(
        np.uint8
    ) * 255

    # ========================================================
    # Morphology
    # ========================================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    # ========================================================
    # Find pupil candidate
    # ========================================================

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_center = np.array(
        [
            width / 2.0,
            height / 2.0,
        ],
        dtype=np.float32,
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area <= 0:
            continue

        (
            (x, y),
            radius,
        ) = cv2.minEnclosingCircle(
            contour
        )

        min_dimension = min(
            height,
            width,
        )

        if radius < (
            min_dimension * 0.03
        ):
            continue

        if radius > (
            min_dimension * 0.25
        ):
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter <= 0:
            continue

        circularity = (
            4.0
            * np.pi
            * area
            / (
                perimeter ** 2
            )
        )

        distance = np.linalg.norm(
            np.array(
                [x, y]
            )
            - image_center
        )

        # Circular + near center + reasonably large
        score = (
            circularity
            * np.sqrt(area)
            / (
                1.0
                + distance
                / min_dimension
            )
        )

        candidates.append(
            (
                score,
                x,
                y,
                radius,
            )
        )

    if not candidates:

        raise RuntimeError(
            "Pupil detection failed."
        )

    candidates.sort(
        key=lambda item:
            item[0],
        reverse=True,
    )

    _, x, y, radius = (
        candidates[0]
    )

    return (
        (
            int(round(x)),
            int(round(y)),
        ),
        int(round(radius)),
    )


def detect_iris(
    image,
    pupil_center,
    pupil_radius,
):
    """
    Detect outer iris boundary.

    Returns:
        iris_center: (x, y)
        iris_radius: int
    """

    height, width = image.shape

    blurred = cv2.GaussianBlur(
        image,
        (9, 9),
        2,
    )

    min_radius = int(
        pupil_radius * 1.8
    )

    max_radius = int(
        min(
            height,
            width,
        ) * 0.48
    )

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    # ========================================================
    # Select circle near pupil
    # ========================================================

    if circles is not None:

        circles = np.round(
            circles[0]
        ).astype(
            int
        )

        candidates = []

        for x, y, radius in circles:

            distance = np.hypot(
                x
                - pupil_center[0],

                y
                - pupil_center[1],
            )

            # Iris center should be
            # reasonably close to pupil
            if distance > (
                pupil_radius * 1.5
            ):
                continue

            candidates.append(
                (
                    distance,
                    x,
                    y,
                    radius,
                )
            )

        if candidates:

            candidates.sort(
                key=lambda item:
                    item[0]
            )

            (
                _,
                x,
                y,
                radius,
            ) = candidates[0]

            return (
                (
                    int(x),
                    int(y),
                ),
                int(radius),
            )

    # ========================================================
    # Fallback
    # ========================================================

    max_possible_radius = int(
        min(
            pupil_center[0],
            width
            - pupil_center[0],

            pupil_center[1],
            height
            - pupil_center[1],
        )
    )

    iris_radius = min(
        int(
            pupil_radius * 3.0
        ),
        max_possible_radius,
    )

    return (
        pupil_center,
        iris_radius,
    )


def create_valid_iris_mask(
    image_shape,
    pupil_center,
    pupil_radius,
    iris_center,
    iris_radius,
    pupil_margin_ratio=0.15,
    iris_margin_ratio=0.12,
):
    """
    Conservative valid region for lesion placement.

    Lesion must be:
        - sufficiently outside pupil
        - sufficiently inside iris

    Returns:
        uint8 mask
        1 = valid
        0 = invalid
    """

    height, width = image_shape

    yy, xx = np.ogrid[
        :height,
        :width,
    ]

    # ========================================================
    # Distance from pupil center
    # ========================================================

    pupil_distance = np.sqrt(
        (xx - pupil_center[0]) ** 2
        +
        (yy - pupil_center[1]) ** 2
    )

    # ========================================================
    # Distance from iris center
    # ========================================================

    iris_distance = np.sqrt(
        (xx - iris_center[0]) ** 2
        +
        (yy - iris_center[1]) ** 2
    )

    # ========================================================
    # Conservative boundaries
    # ========================================================

    pupil_safe_radius = (
        pupil_radius
        * (
            1.0
            + pupil_margin_ratio
        )
    )

    iris_safe_radius = (
        iris_radius
        * (
            1.0
            - iris_margin_ratio
        )
    )

    outside_pupil = (
        pupil_distance
        >= pupil_safe_radius
    )

    inside_iris = (
        iris_distance
        <= iris_safe_radius
    )

    valid_mask = (
        outside_pupil
        &
        inside_iris
    )

    return valid_mask.astype(
        np.uint8
    )