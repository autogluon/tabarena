from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class GroundTruthMixin:
    def labels_test(self, dataset: str, fold: int) -> np.array:
        return self._ground_truth.labels_test(dataset=dataset, fold=fold)

    def labels_val(self, dataset: str, fold: int) -> np.array:
        return self._ground_truth.labels_val(dataset=dataset, fold=fold)
