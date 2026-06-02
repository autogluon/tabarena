from __future__ import annotations

import numpy as np

from tabarena.portfolio.zeroshot_selection import zeroshot_configs


def test_zeroshot_configs():
    val_scores = np.array(
        [
            [0.1, 0.2, 0.0, 1.0, 0.2],
            [0.0, 0.2, 1.0, 1.0, 0.2],
            [0.1, 0.2, 1.0, 0.0, 0.2],
        ]
    )
    assert zeroshot_configs(val_scores, 3) == [0, 2, 3]
    assert zeroshot_configs(val_scores, 4) == [0, 2, 3]
