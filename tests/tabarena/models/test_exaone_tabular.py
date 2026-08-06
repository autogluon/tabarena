"""Fit tests for the EXAONE Tabular wrapper, runnable without a GPU.

``test_all_models.py`` also covers this model, but skips it on a machine without CUDA
(its ``MethodMetadata.compute`` is ``"gpu"``). EXAONE derives its device from the
allocated resources, so with no GPU allocated it runs on CPU — which is what makes a
dedicated file useful while iterating on the wrapper.

Marked ``models`` + ``network`` (a real fit, and the checkpoints are downloaded from
Hugging Face), both of which the default ``addopts`` skips. Cases whose checkpoint cannot
be resolved skip (see :func:`_require_checkpoint`). Run it with::

    pytest tests/tabarena/models/test_exaone_tabular.py -m models -s
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from tabarena.models.exaone_tabular.model import EXAONETabularModel

#: A single ensemble member keeps the toy fits fast; everything else stays at the
#: released checkpoint's defaults.
SMOKE_HYPERPARAMETERS = {"ensemble_count": 1}

pytestmark = [
    pytest.mark.models,
    pytest.mark.network,
    pytest.mark.skipif(
        importlib.util.find_spec("exaonetabular") is None,
        reason="exaonetabular is not installed",
    ),
]

#: Each released checkpoint and the problem types it backs. They are separate files, so one can
#: be reachable while the other is not.
CHECKPOINTS = {
    "classification": ["binary", "multiclass"],
    "regression": ["regression"],
}


def _require_checkpoint(task: str) -> None:
    """Skip unless the released checkpoint for ``task`` can be resolved to a local file.

    Runs the library's own resolution — the one ``from_pretrained`` performs — so a checkpoint
    supplied through ``EXAONETABULAR_{CLASSIFIER,REGRESSOR}_WEIGHTS`` counts as available.
    """
    from exaonetabular.presets import released_checkpoint
    from exaonetabular.weights import resolve_weights

    try:
        resolve_weights(released_checkpoint(task))
    except Exception as err:  # any resolution failure means the fit cannot run here
        pytest.skip(f"EXAONE-Tabular {task} checkpoint is not available: {type(err).__name__}: {err}")


@pytest.mark.parametrize("task", sorted(CHECKPOINTS))
def test_fit_on_toy_datasets(tmp_path, monkeypatch, task: str) -> None:
    """AutoGluon's full model verification on the toy datasets backed by each checkpoint.

    Covers bagging, ``refit_full``, the save/load device round-trip, and seeding. Split per
    checkpoint so an unreachable one does not take the other's coverage with it.
    """
    from autogluon.tabular.testing import FitHelper

    _require_checkpoint(task)
    # FitHelper writes its scratch datasets under the cwd, so keep it out of the repo.
    monkeypatch.chdir(tmp_path)
    FitHelper.verify_model(
        model_cls=EXAONETabularModel,
        model_hyperparameters=SMOKE_HYPERPARAMETERS,
        problem_types=CHECKPOINTS[task],
    )


NUM_ROWS = 120


def _messy_frame(target: np.ndarray) -> pd.DataFrame:
    """``target`` alongside numeric, categorical and string columns carrying NaN and infinities."""
    rng = np.random.default_rng(0)
    numeric = rng.normal(size=NUM_ROWS)
    numeric[:5] = np.nan
    numeric[5:8] = np.inf
    return pd.DataFrame(
        {
            "numeric": numeric,
            "categorical": pd.Categorical(rng.choice(["a", "b", "c", None], size=NUM_ROWS)),
            "string": rng.choice(["x", "y"], size=NUM_ROWS),
            "target": target,
        },
    )


def _fit_predictor(data: pd.DataFrame, path):
    from autogluon.tabular import TabularPredictor

    return TabularPredictor(label="target", path=str(path), verbosity=0).fit(
        data,
        hyperparameters={EXAONETabularModel: [SMOKE_HYPERPARAMETERS]},
    )


@pytest.mark.parametrize("num_classes", [2, 12])
def test_predict_proba_on_messy_frame(tmp_path, num_classes: int) -> None:
    """Categoricals, missing values and infinities reach a usable prediction.

    EXAONE's estimators accept only a real-numeric array and reject infinities outright,
    and the released classifier has a 10-class head. So the wrapper's ordinal encoding,
    its inf-to-NaN fold, and the library's ECOC fallback above 10 classes are all worth
    pinning down here — none of them is exercised by the toy datasets above.
    """
    _require_checkpoint("classification")
    data = _messy_frame(np.random.default_rng(1).integers(0, num_classes, size=NUM_ROWS))
    predictor = _fit_predictor(data, tmp_path / "ag")
    proba = predictor.predict_proba(data.drop(columns=["target"])).to_numpy()

    assert proba.shape == (NUM_ROWS, num_classes)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


def test_predict_on_messy_frame_regression(tmp_path) -> None:
    """The same messy frame with a real-valued target reaches finite point predictions.

    The regressor shares the wrapper's ordinal encoding and inf-to-NaN fold with the classifier
    but takes its own path through the library — an ``f_regression`` column trim, per-member SVD
    augmentation, and the 999-quantile head's trimmed-mean readout — so it is worth its own case.
    """
    _require_checkpoint("regression")
    data = _messy_frame(np.random.default_rng(1).normal(loc=5.0, scale=2.0, size=NUM_ROWS))
    predictor = _fit_predictor(data, tmp_path / "ag")
    predictions = predictor.predict(data.drop(columns=["target"])).to_numpy()

    assert predictions.shape == (NUM_ROWS,)
    assert np.isfinite(predictions).all()
