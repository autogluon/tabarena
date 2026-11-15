"""The linear probe reads its training rows through the same preprocessing as prediction.

On TabArena-Lite's Bioresponse (fintfm 2026-11-11) AutoGluon dropped columns for the model
before ``_fit``. The probe re-read the training rows through the private ``_preprocess``,
which keeps them, so the fitted classifier received a wider table than it was fitted on and
refused it. These tests stand in for the network and check the width the probe passes on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fintfm")

from tabarena.models.fintfm.model import FinTFMModel
from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel


class _Classifier:
    """A fitted classifier that refuses a query wider than its training table, as fintfm's does."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.seen: list[int] = []

    def transform_representation(self, X: np.ndarray, stage: str) -> np.ndarray:
        """Return two columns per row after checking ``X``'s width.

        Args:
            X: Query rows.
            stage: Ignored; accepted for signature compatibility.

        Returns:
            A ``(n, 2)`` stand-in representation.

        Raises:
            ValueError: If ``X`` is not as wide as the training table.
        """
        self.seen.append(X.shape[1])
        if X.shape[1] != self.width:
            raise ValueError("feature width at predict time must match fit time")
        return np.column_stack([X[:, 0], np.ones(len(X))])


def _model_dropping_one_column(monkeypatch: pytest.MonkeyPatch) -> tuple[FinTFMLinearProbeModel, _Classifier]:
    """A probe model whose AutoGluon preprocessing drops column ``drop``, with the parent fit stubbed."""
    model = FinTFMLinearProbeModel(path="", name="fintfm_probe_test", problem_type="binary", eval_metric="roc_auc")
    model.random_seed = 0
    clf = _Classifier(width=2)

    def parent_fit(self, X, y, **kwargs) -> None:
        self.model = clf

    def public_preprocess(self, X, **kwargs) -> np.ndarray:
        # AutoGluon's non-adaptive step removes the column before the model's own _preprocess.
        return X.drop(columns=["drop"]).to_numpy(dtype=np.float32)

    def private_preprocess(self, X, **kwargs) -> np.ndarray:
        return X.to_numpy(dtype=np.float32)

    monkeypatch.setattr(FinTFMModel, "_fit", parent_fit)
    monkeypatch.setattr(FinTFMLinearProbeModel, "preprocess", public_preprocess)
    monkeypatch.setattr(FinTFMLinearProbeModel, "_preprocess", private_preprocess)
    monkeypatch.delenv("FINTFM_PARTITION", raising=False)
    return model, clf


def test_probe_rows_lose_the_columns_autogluon_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe is fitted on the two columns the classifier saw, not the three that arrived."""
    model, clf = _model_dropping_one_column(monkeypatch)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "drop": np.zeros(200)})
    y = pd.Series((X["a"] > 0).astype(int))
    model._fit(X, y)
    assert clf.seen == [2]
    assert model._linear_probe is not None
