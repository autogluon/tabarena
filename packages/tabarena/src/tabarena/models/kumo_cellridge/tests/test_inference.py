from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sdm")
from tabarena.models.kumo_cellridge.model import KumoCellRidgeModel


def test_missing_class_and_reload(tmp_path):
    if not os.environ.get("KUMO_CELLRIDGE_WEIGHTS"):
        pytest.skip("Set KUMO_CELLRIDGE_WEIGHTS to the v1.0 release directory")
    x = pd.DataFrame({"num": np.linspace(-3, 3, 60), "cat": pd.Categorical(["a", "b", None] * 20)})
    y = pd.Series(np.tile([0, 2], 30))
    model = KumoCellRidgeModel(path=str(tmp_path / "model"), problem_type="multiclass")
    model.fit(X=x, y=y, num_cpus=1, num_gpus=0, num_classes=3)
    q = x.iloc[[4, 5]].copy()
    before = model.predict_proba(q)
    assert before.shape == (2, 3)
    assert np.all(model._predict_proba(q)[:, 1] == 0)
    assert np.all(before[:, 1] <= 1.1e-7)
    np.testing.assert_allclose(before.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(model.predict_proba(q.iloc[:1]), before[:1], atol=1e-5)
    path = model.save()
    restored = KumoCellRidgeModel.load(str(Path(path)))
    np.testing.assert_allclose(restored.predict_proba(q), before, atol=1e-5)
