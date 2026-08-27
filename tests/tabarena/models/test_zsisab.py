from __future__ import annotations

import pandas as pd
import numpy as np

from tabarena.models.zsisab.model import ZSISABModel
from tabarena.models.zsisab.info import zsisab_info, zsisab_method_metadata


def test_zsisab_metadata():
    assert zsisab_method_metadata.method == "ZS-ISAB"
    assert zsisab_method_metadata.ag_key == "ZSISAB"
    assert zsisab_info.model_cls == ZSISABModel


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
