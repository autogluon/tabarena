from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from tabarena.models.zsisab.hpo import gen_zsisab
    from tabarena.models.zsisab.info import zsisab_info, zsisab_method_metadata
    from tabarena.models.zsisab.model import ZSISABModel
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False


def test_zsisab_instantiation():
    if not AUTOGLUON_AVAILABLE:
        return
    model = ZSISABModel(problem_type="binary")
    assert model.ag_key == "ZSISAB"
    assert model.ag_name == "ZS-ISAB"


def test_zsisab_metadata():
    if not AUTOGLUON_AVAILABLE:
        return
    assert zsisab_method_metadata.method == "ZS-ISAB"
    assert zsisab_method_metadata.ag_key == "ZSISAB"
    assert zsisab_info.model_cls == ZSISABModel
    assert gen_zsisab is not None


def test_zsisab_preprocessing_dataframe():
    if not AUTOGLUON_AVAILABLE:
        return
    df_raw = pd.DataFrame({
        "feat_num": [1.0, np.nan, 3.0],
        "feat_cat": ["A", "B", "A"],
    })
    model = ZSISABModel()
    processed = model._preprocess(df_raw, is_train=True)
    assert isinstance(processed, pd.DataFrame)
    assert processed.shape == (3, 2)


def test_zsisab_memory_estimate():
    if not AUTOGLUON_AVAILABLE:
        return
    df_raw = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    model = ZSISABModel()
    mem = model._estimate_memory_usage(df_raw)
    assert isinstance(mem, int) and mem > 0
