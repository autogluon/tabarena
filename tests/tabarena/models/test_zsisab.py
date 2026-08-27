from __future__ import annotations

import numpy as np
import pandas as pd

from tabarena.models.zsisab.model import ZSISABModel


def test_zsisab_instantiation():
    model = ZSISABModel(problem_type="binary")
    assert model.ag_key == "ZSISAB"
    assert model.ag_name == "ZS-ISAB"


def test_zsisab_preprocessing():
    df_raw = pd.DataFrame({
        "feat_num": [1.0, np.nan, 3.0],
        "feat_cat": ["A", "B", "A"],
    })
    model = ZSISABModel()
    processed = model._preprocess(df_raw, is_train=True)
    assert processed.shape == (3, 2)
    assert not np.isnan(processed).any()
    assert processed.dtype == np.float32

    # Test double preprocessing safety on np.ndarray
    reprocessed = model._preprocess(processed, is_train=False)
    assert isinstance(reprocessed, np.ndarray)
    assert reprocessed.shape == (3, 2)


def test_zsisab_memory_estimate():
    df_raw = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    model = ZSISABModel()
    mem = model._estimate_memory_usage(df_raw)
    assert isinstance(mem, int) and mem > 0
