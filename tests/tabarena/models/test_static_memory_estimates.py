from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.models.tabdpt.model import TabDPTv13Model
from tabarena.models.tabpfnv2_5.model import RealTabPFNv25Model

VRAM_96_GB = 96 * 1e9


@pytest.mark.parametrize("model_cls", [TabDPTv13Model, RealTabPFNv25Model])
def test_medium_wide_table_stays_under_a_96_gb_budget(model_cls):
    """A 100k x 500 table (the largest of BeyondArena's medium bucket) must not be refused by the estimate."""
    X = pd.DataFrame(np.zeros((100_000, 500), dtype=np.float32))
    estimate = model_cls._estimate_memory_usage_static(X=X, hyperparameters={})
    assert 10 * 1e9 <= estimate < VRAM_96_GB


def test_estimate_grows_with_the_frame():
    small = pd.DataFrame(np.zeros((1_000, 10), dtype=np.float32))
    large = pd.DataFrame(np.zeros((100_000, 500), dtype=np.float32))
    assert RealTabPFNv25Model._estimate_memory_usage_static(X=large) > RealTabPFNv25Model._estimate_memory_usage_static(
        X=small
    )


def test_realtabpfn_keeps_the_row_cap_and_drops_the_feature_cap():
    assert RealTabPFNv25Model._default_auxiliary_params_extra == {"max_rows": 100_000}
