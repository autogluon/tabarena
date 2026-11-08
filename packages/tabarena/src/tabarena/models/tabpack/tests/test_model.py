"""TabPack wrapper logic the registry fit test does not reach: its preprocessing, a class missing from
a fold's training rows, the regression target scale, the time limit with its epoch reserve, the category
budget and the pack size for wide tables. Cheap CPU fits (two members).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from autogluon.core.metrics import get_metric
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

pytest.importorskip("tabpackage")

from tabarena.models.tabpack.model import TabPackModel, _max_pack_size


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "num_nan": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n) * 1000),
            "flag": rng.choice(["yes", "no"], n),
            "cat": rng.choice(["a", "b", "c", None], n).astype(object),
        }
    )


def _fit(
    tmp_path, X, y, problem_type, metric, X_val=None, y_val=None, hyperparameters=None, **fit_kwargs
) -> TabPackModel:
    """Fit on AutoGluon's default features, as the benchmark does (bool and category special types included)."""
    generator = AutoMLPipelineFeatureGenerator(verbosity=0)
    X = generator.fit_transform(X)
    X_val = None if X_val is None else generator.transform(X_val)
    model = TabPackModel(
        path=str(tmp_path / problem_type) + "/",
        name="TabPack",
        problem_type=problem_type,
        eval_metric=get_metric(metric, problem_type),
        hyperparameters=hyperparameters or {"n_models": 2, "max_epochs": 1},
    )
    model.generator = generator  # the tests transform new rows the same way
    model.fit(
        X=X,
        y=y,
        X_val=X_val,
        y_val=y_val,
        feature_metadata=generator.feature_metadata,
        num_cpus=1,
        num_gpus=0,
        **fit_kwargs,
    )
    return model


def test_preprocessing_meets_the_library_contract_on_new_rows(tmp_path):
    X = _frame(200)
    y = pd.Series((X["num"] > 0).astype(int))
    model = _fit(tmp_path, X.iloc[:150], y.iloc[:150], "binary", "roc_auc", X.iloc[150:], y.iloc[150:])

    X_new = _frame(5, seed=1)
    X_new.loc[0, "num"] = np.inf
    X_new.loc[1, "num_nan"] = np.nan
    X_new["cat"] = [None, "unseen", "b", "a", "b"]
    X_new = model.generator.transform(X_new)
    out = model.preprocess(X_new)

    num_cols = ["num", "num_nan"]
    assert (out[num_cols].dtypes == np.float32).all()
    assert np.isfinite(out[num_cols].to_numpy()).all()
    assert out["cat"].dtype == "string"
    # AutoGluon codes a missing training value as a category of its own and an unseen value as NaN.
    assert out["cat"].tolist()[:2] == ["0", "__missing__"]
    assert out["flag"].dtype == "string"  # bool features are categorical, as in the paper
    assert out["__tabpack_missing__num_nan"].tolist()[1] == "1"
    assert "__tabpack_missing__num" not in out  # no missing values in "num" during training
    assert model.predict_proba(X_new).shape == (5,)
    assert not model.model.tabpack_._checkpoints.state_dicts  # the Developer fix dropped them


def test_a_class_missing_from_the_training_rows_keeps_its_probability_column(tmp_path):
    X = _frame(300)
    y = pd.Series(np.digitize(X["num"], [-0.5, 0.5]))  # classes 0, 1, 2
    train = y != 2
    # In a bagged fit the parent hands every fold model the full class count.
    model = _fit(tmp_path, X[train], y[train], "multiclass", "log_loss", X.iloc[:60], y.iloc[:60], num_classes=3)

    proba = model.predict_proba(model.generator.transform(X.iloc[:10]))
    assert proba.shape == (10, 3)
    np.testing.assert_allclose(proba[:, 2], 0.0, atol=1e-6)  # AutoGluon clips probabilities away from 0
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


def test_regression_predictions_are_on_the_target_scale(tmp_path):
    X = _frame(300)
    y = pd.Series(5000.0 + 100.0 * X["num"])
    model = _fit(tmp_path, X.iloc[:200], y.iloc[:200], "regression", "rmse", X.iloc[200:], y.iloc[200:])

    pred = model.predict(model.generator.transform(X.iloc[200:]))
    assert abs(pred.mean() - 5000.0) < 200.0


def test_time_limit_stops_a_fit_with_no_other_stopping_condition(tmp_path):
    X = _frame(300)
    y = pd.Series((X["num"] > 0).astype(int))
    # No epoch limit, no member early stopping, no stop on ensemble finalization: only the budget ends the loop.
    hps = {"n_models": 2, "early_stopping_patience": None, "stop_when_all_ensembles_are_finalized": False}
    model = _fit(
        tmp_path,
        X.iloc[:200],
        y.iloc[:200],
        "binary",
        "log_loss",
        X.iloc[200:],
        y.iloc[200:],
        hyperparameters=hps,
        time_limit=3.0,
    )

    budget = model.model.training_loop_time_budget
    assert 0 < budget < 0.95 * 3.0
    assert budget <= model.model.tabpack_.get_training_loop_time() < 3.0
    assert model.predict_proba(model.generator.transform(X.iloc[:5])).shape == (5,)


def test_time_limit_reserves_one_probed_epoch_on_large_tables(tmp_path, monkeypatch):
    probed = []
    real_probe = TabPackModel._estimate_epoch_seconds

    def probe_then_reserve_ten_seconds(self, *args):
        probed.append(real_probe(self, *args))
        return 10.0

    monkeypatch.setattr(TabPackModel, "_estimate_epoch_seconds", probe_then_reserve_ten_seconds)
    X = _frame(300)
    y = pd.Series((X["num"] > 0).astype(int))
    # A row threshold of 1 makes this toy table "large"; the explicit batch size is the one the probe reuses.
    hps = {"n_models": 2, "max_epochs": 1, "batch_size": 32, "epoch_probe_min_rows": 1}
    model = _fit(
        tmp_path,
        X.iloc[:200],
        y.iloc[:200],
        "binary",
        "log_loss",
        X.iloc[200:],
        y.iloc[200:],
        hyperparameters=hps,
        time_limit=30.0,
    )

    assert len(probed) == 1 and probed[0] > 0  # the probe fit ran and timed its training loop
    assert 0 < model.model.training_loop_time_budget < 0.95 * 30.0 - 10.0  # the epoch is reserved from the budget
    assert model.predict_proba(model.generator.transform(X.iloc[:5])).shape == (5,)


def test_category_budget_trims_only_the_largest_feature(tmp_path):
    X = _frame(400)
    X["id"] = [f"user{i // 2}" for i in range(400)]  # 200 categories, each seen twice
    y = pd.Series((X["num"] > 0).astype(int))
    hps = {"n_models": 2, "max_epochs": 1, "max_total_categories": 30}
    model = _fit(tmp_path, X, y, "binary", "roc_auc", hyperparameters=hps)

    out = model.preprocess(model.generator.transform(X))
    sizes = {c: out[c].nunique() for c in ["id", "cat", "flag"]}  # the categorical features the budget covers
    assert sizes["id"] == 24  # 200 categories: the 23 most frequent plus the missing category
    assert sum(sizes.values()) == 30
    assert sizes["cat"] == 4 and sizes["flag"] == 2  # below the shared cap, so whole
    assert set(model._kept_categories) == {"id"}
    assert "__missing__" in set(out["id"])  # the trimmed values join the missing category
    assert model.predict_proba(model.generator.transform(X.iloc[:5])).shape == (5,)


def test_pack_size_shrinks_only_when_the_first_layers_exceed_the_budget():
    X = pd.DataFrame(np.zeros((4, 100), dtype=np.float32)).rename(columns=str)
    X["cat"] = pd.array(["a", "b", "c", "a"], dtype="string")
    bytes_per_model = 16 * 384 * (100 * 32 + 3)  # float32 weights, gradients and two AdamW moments

    assert _max_pack_size(X, {}, memory_budget=64 * bytes_per_model) is None  # the default 64 models fit
    assert _max_pack_size(X, {}, memory_budget=10.5 * bytes_per_model) == 10
    assert _max_pack_size(X, {"n_models": 8}, memory_budget=10.5 * bytes_per_model) is None
    assert _max_pack_size(X, {}, memory_budget=0.1 * bytes_per_model) == 1
    # Narrower embeddings than the default range's top make each model smaller.
    assert _max_pack_size(X, {"numerical_embeddings_d": 8}, memory_budget=10.5 * bytes_per_model) == 41


def test_pack_size_keeps_the_stacked_first_layer_below_int32_indexing():
    X = pd.DataFrame(np.zeros((4, 3000), dtype=np.float32)).rename(columns=str)
    weights_per_model = 384 * 3000 * 32
    # Memory alone would allow all 64 models; 64 first layers would hold more than 2**31 weights.
    assert _max_pack_size(X, {}, memory_budget=1e15) == (2**31 - 1) // weights_per_model == 58
