from __future__ import annotations

import numpy as np
import pandas as pd

from tabarena.models.zsisab.model import ZSISABModel
from tabarena.models.zsisab.info import zsisab_info, zsisab_method_metadata
from tabarena.models.zsisab.hpo import gen_zsisab


def test_zsisab_instantiation():
    model = ZSISABModel(problem_type="binary")
    assert model.ag_key == "ZSISAB"
    assert model.ag_name == "ZS-ISAB"


def test_zsisab_metadata():
    if zsisab_method_metadata is not None:
        assert zsisab_method_metadata.method == "ZS-ISAB"
        assert zsisab_method_metadata.ag_key == "ZSISAB"
        assert zsisab_info.model_cls == ZSISABModel
        assert gen_zsisab is not None


def test_zsisab_preprocessing_dataframe():
    df_raw = pd.DataFrame({
        "feat_num": [1.0, np.nan, 3.0],
        "feat_cat": ["A", "B", "A"],
    })
    model = ZSISABModel()
    processed = model._preprocess(df_raw, is_train=True)
    assert processed.shape == (3, 2)
    assert not np.isnan(processed).any()
    assert processed.dtype == np.float32


def test_zsisab_preprocessing_ndarray():
    arr_raw = np.array([[1.0, np.nan], [4.0, 5.0]], dtype=np.float64)
    model = ZSISABModel()
    processed = model._preprocess(arr_raw, is_train=False)
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (2, 2)
    assert processed.dtype == np.float32
    assert not np.isnan(processed).any()


def test_zsisab_memory_estimate():
    df_raw = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    model = ZSISABModel()
    mem = model._estimate_memory_usage(df_raw)
    assert isinstance(mem, int) and mem > 0
