"""``FINTFM_MINORITY_POSITIVE=1`` codes the minority class as 1 and maps the probabilities back."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.models.fintfm.model import FinTFMModel


class _StubClassifier:
    """Returns fixed two-column probabilities, so the adapter's column handling is all that varies."""

    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.proba


def _model(problem_type: str = "binary", eval_metric: str = "roc_auc") -> FinTFMModel:
    model = FinTFMModel(path="", name="fintfm_test", problem_type=problem_type, eval_metric=eval_metric)
    model.model = _StubClassifier()
    model.preprocess = lambda X, **kwargs: np.asarray(X)  # the stub ignores features
    return model


@pytest.mark.parametrize(
    ("flag", "share_of_ones", "expected"),
    [(None, 0.9, False), ("0", 0.9, False), ("1", 0.9, True), ("1", 0.5, False), ("1", 0.1, False)],
)
def test_recodes_only_when_opted_in_and_class_one_is_the_majority(
    monkeypatch: pytest.MonkeyPatch, flag: str | None, share_of_ones: float, expected: bool
) -> None:
    """Off by default; on, only a strict class-1 majority is recoded (an exact half is not)."""
    monkeypatch.delenv("FINTFM_MINORITY_POSITIVE", raising=False)
    if flag is not None:
        monkeypatch.setenv("FINTFM_MINORITY_POSITIVE", flag)
    y = np.zeros(100, dtype=np.int64)
    y[: round(100 * share_of_ones)] = 1
    assert _model()._flip_to_minority(y) is expected


def test_regression_is_never_recoded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINTFM_MINORITY_POSITIVE", "1")
    assert _model("regression", "rmse")._flip_to_minority(np.ones(10)) is False


def test_predict_proba_maps_the_columns_back() -> None:
    """A recoded fit's column 0 is AutoGluon's class 1, so that is what the adapter returns;
    without a recoding the adapter returns column 1 exactly as AutoGluon's own path does.
    """
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    plain = _model()
    flipped = _model()
    flipped._minority_flipped = True
    np.testing.assert_array_equal(plain._predict_proba(X), _StubClassifier.proba[:, 1])
    np.testing.assert_array_equal(flipped._predict_proba(X), _StubClassifier.proba[:, 0])
