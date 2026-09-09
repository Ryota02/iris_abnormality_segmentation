from collections import Counter

import torch
from torch.utils.data import Sampler


class CategoryMultiplierSampler(Sampler):
    """
    Categoryごとに指定した倍率でsampleを繰り返すSampler。

    Example:

        Geometry: 43 images × 1
        Healthy : 32 images × 1
        Tissue  :  6 images × 7

    -> 1 epoch = 117 samples

    各epochでindex順序をshuffleする。
    """

    def __init__(
        self,
        samples,
        category_multiplier,
        seed=42,
    ):
        self.samples = samples

        self.category_multiplier = {
            category: int(multiplier)
            for category, multiplier
            in category_multiplier.items()
        }

        self.seed = int(seed)

        self.epoch = 0

        self.indices = self._build_indices()

        self._print_summary()

    # ========================================================
    # Build repeated indices
    # ========================================================

    def _build_indices(self):

        indices = []

        for index, sample in enumerate(
            self.samples
        ):

            category = sample[
                "category"
            ]

            multiplier = (
                self.category_multiplier.get(
                    category,
                    1,
                )
            )

            if multiplier < 1:
                raise ValueError(
                    f"Multiplier must be >= 1: "
                    f"{category}={multiplier}"
                )

            indices.extend(
                [index] * multiplier
            )

        return indices

    # ========================================================
    # Iterator
    # ========================================================

    def __iter__(self):

        generator = torch.Generator()

        generator.manual_seed(
            self.seed
            + self.epoch
        )

        permutation = torch.randperm(
            len(self.indices),
            generator=generator,
        ).tolist()

        shuffled_indices = [
            self.indices[i]
            for i in permutation
        ]

        self.epoch += 1

        return iter(
            shuffled_indices
        )

    # ========================================================
    # Length
    # ========================================================

    def __len__(self):

        return len(
            self.indices
        )

    # ========================================================
    # Summary
    # ========================================================

    def _print_summary(self):

        original_counts = Counter(
            sample["category"]
            for sample in self.samples
        )

        effective_counts = {}

        for category, count in (
            original_counts.items()
        ):

            multiplier = (
                self.category_multiplier.get(
                    category,
                    1,
                )
            )

            effective_counts[
                category
            ] = (
                count
                * multiplier
            )

        print(
            "\n"
            "=================================="
        )

        print(
            "Oversampling"
        )

        print(
            "=================================="
        )

        for category in sorted(
            original_counts.keys()
        ):

            original = (
                original_counts[
                    category
                ]
            )

            multiplier = (
                self.category_multiplier.get(
                    category,
                    1,
                )
            )

            effective = (
                effective_counts[
                    category
                ]
            )

            print(
                f"{category:<10} "
                f"{original:>4} "
                f"x {multiplier:<2} "
                f"= {effective:>4}"
            )

        print(
            "----------------------------------"
        )

        print(
            f"Total samples per epoch: "
            f"{len(self.indices)}"
        )