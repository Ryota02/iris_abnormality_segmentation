import random

import cv2
import numpy as np
import torch


class SegmentationAugmentation:

    def __init__(
        self,
        image_size,
        augmentation_config=None,
        train=False,
    ):
        self.image_size = int(
            image_size
        )

        self.augmentation_config = (
            augmentation_config or {}
        )

        self.train = train

        self.enabled = (
            self.augmentation_config.get(
                "enabled",
                False,
            )
        )

    # ========================================================
    # Probability
    # ========================================================

    @staticmethod
    def _apply(probability):

        return (
            random.random()
            < float(probability)
        )

    # ========================================================
    # Horizontal flip
    # ========================================================

    def _horizontal_flip(
        self,
        image,
        mask,
    ):

        image = np.fliplr(
            image
        ).copy()

        mask = np.fliplr(
            mask
        ).copy()

        return image, mask

    # ========================================================
    # Rotation
    # ========================================================

    def _rotation(
        self,
        image,
        mask,
        limit,
    ):

        angle = random.uniform(
            -float(limit),
            float(limit),
        )

        height, width = (
            image.shape
        )

        center = (
            width / 2.0,
            height / 2.0,
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0,
        )

        image = cv2.warpAffine(
            image,
            matrix,
            (
                width,
                height,
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        mask = cv2.warpAffine(
            mask,
            matrix,
            (
                width,
                height,
            ),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return image, mask

    # ========================================================
    # Brightness
    # ========================================================

    def _brightness(
        self,
        image,
        limit,
    ):

        factor = random.uniform(
            -float(limit),
            float(limit),
        )

        image = image.astype(
            np.float32
        )

        image = (
            image
            + factor * 255.0
        )

        image = np.clip(
            image,
            0,
            255,
        )

        return image.astype(
            np.uint8
        )

    # ========================================================
    # Contrast
    # ========================================================

    def _contrast(
        self,
        image,
        limit,
    ):

        factor = random.uniform(
            1.0 - float(limit),
            1.0 + float(limit),
        )

        image = image.astype(
            np.float32
        )

        mean = image.mean()

        image = (
            (image - mean)
            * factor
            + mean
        )

        image = np.clip(
            image,
            0,
            255,
        )

        return image.astype(
            np.uint8
        )

    # ========================================================
    # Gamma
    # ========================================================

    def _gamma(
        self,
        image,
        limit,
    ):

        gamma = random.uniform(
            1.0 - float(limit),
            1.0 + float(limit),
        )

        image_float = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        image_float = np.power(
            image_float,
            gamma,
        )

        image = np.clip(
            image_float * 255.0,
            0,
            255,
        )

        return image.astype(
            np.uint8
        )

    # ========================================================
    # Gaussian noise
    # ========================================================

    def _gaussian_noise(
        self,
        image,
        sigma,
    ):

        image_float = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        noise = np.random.normal(
            loc=0.0,
            scale=float(sigma),
            size=image.shape,
        ).astype(
            np.float32
        )

        image_float = (
            image_float
            + noise
        )

        image_float = np.clip(
            image_float,
            0.0,
            1.0,
        )

        return (
            image_float
            * 255.0
        ).astype(
            np.uint8
        )

    # ========================================================
    # Gaussian blur
    # ========================================================

    @staticmethod
    def _gaussian_blur(
        image,
    ):

        return cv2.GaussianBlur(
            image,
            (3, 3),
            0,
        )

    # ========================================================
    # Category-specific augmentation
    # ========================================================

    def _augment(
        self,
        image,
        mask,
        category,
    ):

        if (
            not self.train
            or not self.enabled
        ):
            return image, mask

        category_cfg = (
            self.augmentation_config.get(
                category,
                {},
            )
        )

        # ----------------------------------------------------
        # Horizontal flip
        # image + mask
        # ----------------------------------------------------

        cfg = category_cfg.get(
            "horizontal_flip",
            {},
        )

        if (
            cfg.get(
                "enabled",
                False,
            )
            and self._apply(
                cfg.get(
                    "probability",
                    0.5,
                )
            )
        ):

            image, mask = (
                self._horizontal_flip(
                    image,
                    mask,
                )
            )

        # ----------------------------------------------------
        # Rotation
        # image + mask
        # ----------------------------------------------------

        cfg = category_cfg.get(
            "rotation",
            {},
        )

        if (
            cfg.get(
                "enabled",
                False,
            )
            and self._apply(
                cfg.get(
                    "probability",
                    0.5,
                )
            )
        ):

            image, mask = (
                self._rotation(
                    image,
                    mask,
                    cfg.get(
                        "limit",
                        10,
                    ),
                )
            )

        # ----------------------------------------------------
        # Brightness
        # image only
        # ----------------------------------------------------

        cfg = category_cfg.get(
            "brightness",
            {},
        )

        if (
            cfg.get(
                "enabled",
                False,
            )
            and self._apply(
                cfg.get(
                    "probability",
                    0.3,
                )
            )
        ):

            image = self._brightness(
                image,
                cfg.get(
                    "limit",
                    0.1,
                ),
            )

        # ----------------------------------------------------
        # Contrast
        # image only
        # ----------------------------------------------------

        cfg = category_cfg.get(
            "contrast",
            {},
        )

        if (
            cfg.get(
                "enabled",
                False,
            )
            and self._apply(
                cfg.get(
                    "probability",
                    0.3,
                )
            )
        ):

            image = self._contrast(
                image,
                cfg.get(
                    "limit",
                    0.1,
                ),
            )

        # ----------------------------------------------------
        # Gamma
        # image only
        # ----------------------------------------------------

        cfg = category_cfg.get(
            "gamma",
            {},
        )

        if (
            cfg.get(
                "enabled",
                False,
            )
            and self._apply(
                cfg.get(
                    "probability",
                    0.2,
                )
            )
        ):

            image = self._gamma(
                image,
                cfg.get(
                    "limit",
                    0.1,
                ),
            )

        # ----------------------------------------------------
        # Gaussian noise
        # image only
        # ----------------------------------------------------

        cfg = category_cfg.get(
            "gaussian_noise",
            {},
        )

        if (
            cfg.get(
                "enabled",
                False,
            )
            and self._apply(
                cfg.get(
                    "probability",
                    0.2,
                )
            )
        ):

            image = (
                self._gaussian_noise(
                    image,
                    cfg.get(
                        "sigma",
                        0.01,
                    ),
                )
            )

        # ----------------------------------------------------
        # Gaussian blur
        # image only
        # ----------------------------------------------------

        cfg = category_cfg.get(
            "gaussian_blur",
            {},
        )

        if (
            cfg.get(
                "enabled",
                False,
            )
            and self._apply(
                cfg.get(
                    "probability",
                    0.1,
                )
            )
        ):

            image = (
                self._gaussian_blur(
                    image
                )
            )

        return image, mask

    # ========================================================
    # Main
    # ========================================================

    def __call__(
        self,
        image,
        mask,
        category,
    ):

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = cv2.resize(
            image,
            (
                self.image_size,
                self.image_size,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (
                self.image_size,
                self.image_size,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        image, mask = self._augment(
            image,
            mask,
            category,
        )

        # ----------------------------------------------------
        # Normalize image
        # ----------------------------------------------------

        image = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        # ----------------------------------------------------
        # Binary GT
        # ----------------------------------------------------

        mask = (
            mask > 0
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        image = (
            torch.from_numpy(
                image
            )
            .unsqueeze(0)
        )

        mask = (
            torch.from_numpy(
                mask
            )
            .unsqueeze(0)
        )

        return image, mask