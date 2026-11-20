"""LGBM-Prep-FTE: portfolio integrity, label isolation of TabPrep, row-independent predictions, ensembling.

Everything runs on CPU on small synthetic frames. The only end-to-end fit is a two-config system (the
LightGBM default plus one TabPrep config, 8-fold bags on 200 rows), fitted once per module.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from autogluon.core.metrics import get_metric

from tabarena.benchmark.task.metadata import ValidationMetadata
from tabarena.systems._registry import discover_systems
from tabarena.systems.lgbm_prep_fte import (
    _tabprep,
    system as lgbm_prep_fte_system,
)
from tabarena.systems.lgbm_prep_fte._model import LGBMDefaultModel, LGBMPrepModel
from tabarena.systems.lgbm_prep_fte.system import (
    ARENA_SETUPS,
    NAME,
    NUM_BAG_FOLDS,
    LGBMPrepFTESystemModel,
    ensemble_weights,
    load_configs,
    make_bag_experiment,
    system_time_limit,
)

PORTFOLIOS = [
    # arena, configs (default + filtered portfolio), TabPrep-enabled configs, per-model limit in seconds
    ("tabarena", 201, 54, 3600),
    ("beyondarena", 26, 8, 4 * 3600),
]
METRICS = {"binary": "log_loss", "multiclass": "log_loss", "regression": "rmse"}


def _tabprep_config(arena: str = "beyondarena") -> dict:
    return next(c for c in load_configs(arena) if c.get("hyperparameters", {}).get("tabprep_enabled"))


def _frame(problem_type: str, n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Numeric and categorical features with a target that depends on both, in AutoGluon's label space."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"n{i}": rng.normal(size=n) for i in range(4)})
    X["c0"] = pd.Categorical(rng.choice(list("abcde"), n))
    X["c1"] = pd.Categorical(rng.choice(list("xyz"), n))
    signal = X["n0"].to_numpy() + 1.5 * (X["c0"] == "a").to_numpy() - 0.5 * X["n1"].to_numpy()
    signal = signal + rng.normal(0, 0.5, n)
    if problem_type == "binary":
        y = (signal > 0).astype(int)
    elif problem_type == "multiclass":
        y = np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3]))
    else:
        y = signal
    return X, pd.Series(y, name="target")


def _fit_child(X_train, y_train, X_val, y_val, *, problem_type: str, path, config: dict | None = None):
    """One bag child (``LGBMPrepModel``) fitted the way the bag fits it: training rows plus its inner fold."""
    config = _tabprep_config() if config is None else config
    model = LGBMPrepModel(
        path=str(path),
        name="child",
        problem_type=problem_type,
        eval_metric=get_metric(METRICS[problem_type], problem_type),
        hyperparameters=copy.deepcopy(config["hyperparameters"]),
    )
    model.fit(
        X=X_train,
        y=y_train,
        X_val=X_val,
        y_val=y_val,
        total_resources={"num_cpus": 1, "num_gpus": 0},
    )
    return model


def _tabprep_view(model: LGBMPrepModel, X: pd.DataFrame) -> pd.DataFrame:
    """The features a fitted child hands to LightGBM for ``X``: AutoGluon preprocessing, TabPrep, the cap."""
    return model.tabprep_.transform(model.preprocess(X)).loc[:, model.selected_columns_]


def _chunks(X: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    return [X.iloc[i : i + size] for i in range(0, len(X), size)]


# -- 1. portfolio integrity -------------------------------------------------------------------


@pytest.mark.parametrize(("arena", "n_configs", "n_tabprep", "model_limit"), PORTFOLIOS)
def test_portfolio_size_names_and_tabprep_count(arena, n_configs, n_tabprep, model_limit):
    del model_limit
    configs = load_configs(arena)
    assert len(configs) == n_configs
    names = [c["name"] for c in configs]
    assert len(set(names)) == len(names)
    assert names == [f"c{i}" for i in range(1, n_configs + 1)]
    assert sum(bool(c.get("hyperparameters", {}).get("tabprep_enabled")) for c in configs) == n_tabprep
    # Every non-default config is a complete LGBM-Prep record with its own seed.
    for config in configs[1:]:
        assert config["model"] == "lgbm_prep", config["name"]
        hps = config["hyperparameters"]
        assert set(hps) == {"config_id", "model_seed", "tabprep_enabled", "lightgbm", "tabprep"}, config["name"]
        assert isinstance(hps["model_seed"], int), config["name"]
    config_ids = [c["hyperparameters"]["config_id"] for c in configs[1:]]
    assert len(set(config_ids)) == len(config_ids)


@pytest.mark.parametrize("arena", [a for a, *_ in PORTFOLIOS])
def test_first_config_is_the_plain_lightgbm_default(arena):
    first = load_configs(arena)[0]
    assert first["model"] == "lightgbm_default"
    assert "hyperparameters" not in first
    assert all(c["model"] != "lightgbm_default" for c in load_configs(arena)[1:])

    experiment = make_bag_experiment(first, arena=arena, bag_time_limit=None)
    assert experiment.method_kwargs["model_cls"] is LGBMDefaultModel
    # AutoGluon's defaults: nothing but the bag's own arguments, no TabPrep.
    assert set(experiment.method_kwargs["model_hyperparameters"]) == {"ag_args_ensemble"}


def test_make_bag_experiment_wires_the_config_without_mutating_it():
    config = _tabprep_config("tabarena")
    before = copy.deepcopy(config)
    experiment = make_bag_experiment(config, arena="tabarena", bag_time_limit=10)
    assert config == before
    assert experiment.name == f"{NAME}_{config['name']}"
    assert experiment.method_kwargs["model_cls"] is LGBMPrepModel
    hps = experiment.method_kwargs["model_hyperparameters"]
    assert hps["tabprep"] == config["hyperparameters"]["tabprep"]
    assert hps["ag_args_ensemble"]["model_random_seed"] == config["bag_seed"]
    assert experiment.method_kwargs["fit_kwargs"]["num_gpus"] == 0


@pytest.mark.parametrize(("arena", "n_configs", "n_tabprep", "model_limit"), PORTFOLIOS)
def test_system_time_limit_gives_every_fold_the_per_model_limit(arena, n_configs, n_tabprep, model_limit):
    del n_tabprep
    assert ARENA_SETUPS[arena].model_time_limit == model_limit
    assert ARENA_SETUPS[arena].bag_time_limit == NUM_BAG_FOLDS * model_limit
    assert system_time_limit(arena) == n_configs * NUM_BAG_FOLDS * model_limit
    # `_fit_system` splits `time_limit` evenly over the bags.
    assert system_time_limit(arena) / n_configs == ARENA_SETUPS[arena].bag_time_limit


def test_unknown_arena_is_rejected():
    with pytest.raises(ValueError, match="arena must be one of"):
        LGBMPrepFTESystemModel(arena="openml", problem_type="binary", eval_metric=None)


# -- 2. validation labels are only for scoring ------------------------------------------------


@pytest.fixture
def small_cap(monkeypatch):
    """A feature cap below TabPrep's output, so the y-dependent Spearman selection is part of the fit."""
    monkeypatch.setattr(_tabprep, "FEATURE_CAP", 12)


@pytest.mark.parametrize(
    ("problem_type", "arena"),
    [("binary", "beyondarena"), ("multiclass", "beyondarena"), ("regression", "beyondarena"), ("binary", "tabarena")],
)
def test_tabprep_features_do_not_depend_on_inner_validation_labels(small_cap, tmp_path, problem_type, arena):
    """A bag child's TabPrep (target encoding, feature cap) sees only its training labels.

    Refitting with the inner validation fold's labels permuted (and, for classification, relabelled) must give
    the same kept columns and the same transformed features for the validation rows and for unseen rows.
    """
    X, y = _frame(problem_type)
    train, val, probe = slice(0, 120), slice(120, 160), slice(160, 200)
    config = _tabprep_config(arena)
    rng = np.random.default_rng(1)
    y_val = y.iloc[val]
    other_y_val = pd.Series(rng.permutation(y_val.to_numpy()), index=y_val.index)
    if problem_type == "binary":
        other_y_val = 1 - other_y_val
    elif problem_type == "regression":
        other_y_val = other_y_val * 10.0 + 3.0

    fits = [
        _fit_child(
            X.iloc[train],
            y.iloc[train],
            X.iloc[val],
            labels,
            problem_type=problem_type,
            path=tmp_path / name,
            config=config,
        )
        for name, labels in [("a", y_val), ("b", other_y_val)]
    ]
    a, b = fits
    assert a.prep_report_["feature_cap_applied"]
    assert len(a.selected_columns_) == 12
    assert a.selected_columns_ == b.selected_columns_
    for rows in (val, probe):
        pd.testing.assert_frame_equal(_tabprep_view(a, X.iloc[rows]), _tabprep_view(b, X.iloc[rows]))


def test_tabprep_features_do_depend_on_training_labels(small_cap, tmp_path):
    """Control for the test above: the same comparison does see a change of the *training* labels."""
    X, y = _frame("binary")
    train, val = slice(0, 120), slice(120, 160)
    flipped = 1 - y.iloc[train]
    flipped.iloc[::3] = y.iloc[train].iloc[::3]  # flip two thirds: not a mere relabelling of the classes
    a = _fit_child(X.iloc[train], y.iloc[train], X.iloc[val], y.iloc[val], problem_type="binary", path=tmp_path / "a")
    b = _fit_child(X.iloc[train], flipped, X.iloc[val], y.iloc[val], problem_type="binary", path=tmp_path / "b")
    view_a, view_b = _tabprep_view(a, X.iloc[val]), _tabprep_view(b, X.iloc[val])
    assert a.selected_columns_ != b.selected_columns_ or not view_a.equals(view_b)


def test_tabprep_frames_fit_ignores_the_rows_it_only_transforms(small_cap):
    """``tabprep_frames`` fits on ``train`` alone: what the scored frame contains does not change the fit."""
    X, y = _frame("binary")
    train, probe = X.iloc[:120], X.iloc[160:]
    hps = _tabprep_config()["hyperparameters"]["tabprep"]
    outputs = [
        _tabprep.tabprep_frames({"train": train, "val": val}, y.iloc[:120].to_numpy(), "binary", hps, num_cpus=1)
        for val in (X.iloc[120:160], X.iloc[120:160].sample(frac=1.0, random_state=0).iloc[:7])
    ]
    (frames_a, _, prep_a, selected_a), (frames_b, _, prep_b, selected_b) = outputs
    assert selected_a == selected_b
    pd.testing.assert_frame_equal(frames_a["train"], frames_b["train"])
    pd.testing.assert_frame_equal(
        prep_a.transform(probe).loc[:, selected_a], prep_b.transform(probe).loc[:, selected_b]
    )


# -- 3. row-independent transforms and predictions --------------------------------------------


@pytest.fixture(scope="module")
def fitted_child(tmp_path_factory):
    X, y = _frame("multiclass")
    model = _fit_child(
        X.iloc[:120],
        y.iloc[:120],
        X.iloc[120:160],
        y.iloc[120:160],
        problem_type="multiclass",
        path=tmp_path_factory.mktemp("child"),
    )
    return model, X.iloc[160:]


@pytest.mark.parametrize("chunk", [1, 7])
def test_tabprep_transform_is_row_independent(fitted_child, chunk):
    model, X_test = fitted_child
    whole = _tabprep_view(model, X_test)
    pieces = pd.concat([_tabprep_view(model, part) for part in _chunks(X_test, chunk)])
    pd.testing.assert_frame_equal(whole, pieces)


@pytest.mark.parametrize("chunk", [1, 7])
def test_child_predict_proba_is_row_independent(fitted_child, chunk):
    model, X_test = fitted_child
    whole = np.asarray(model.predict_proba(X_test))
    pieces = np.concatenate([np.asarray(model.predict_proba(part)) for part in _chunks(X_test, chunk)])
    np.testing.assert_allclose(whole, pieces, rtol=0, atol=1e-12)


@pytest.fixture(scope="module")
def fitted_system(tmp_path_factory):
    """The system on a two-config portfolio (the LightGBM default and one TabPrep config)."""
    X, y = _frame("binary")
    y = y.map({0: "no", 1: "yes"})
    configs = load_configs("tabarena")
    portfolio = [configs[0], _tabprep_config("tabarena")]
    metric = get_metric("roc_auc", "binary")
    time_limit = 120.0
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(lgbm_prep_fte_system, "load_configs", lambda arena: copy.deepcopy(portfolio))
        model = LGBMPrepFTESystemModel(
            arena="tabarena",
            path=str(tmp_path_factory.mktemp("system")),
            problem_type="binary",
            eval_metric=metric,
            fit_kwargs={"num_cpus": 1},
        )
        model._fit_system(
            X.iloc[:160],
            y.iloc[:160],
            target_name="target",
            problem_type="binary",
            eval_metric=metric,
            validation_metadata=ValidationMetadata(),
            num_cpus=1,
            num_gpus=0,
            memory_limit=None,
            time_limit=time_limit,
            random_state=None,
        )
    return model, X.iloc[160:], portfolio, time_limit


def test_system_fit_reports_normalised_weights_and_the_bag_budget(fitted_system):
    model, _, portfolio, time_limit = fitted_system
    report = model.get_metadata()
    assert report["arena"] == "tabarena"
    assert report["bag_time_limit"] == time_limit / len(portfolio)
    weights = np.array(list(report["weights"].values()))
    assert list(report["weights"]) == [c["name"] for c in portfolio]
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0)
    assert len(model._bags) == len(portfolio)


@pytest.mark.parametrize("chunk", [1, 7])
def test_system_predictions_are_row_independent(fitted_system, chunk):
    model, X_test, *_ = fitted_system
    # Each call predicts with 16 fold models; one row at a time on a few rows keeps the test fast.
    X_test = X_test if chunk > 1 else X_test.iloc[:10]
    whole = model._predict_proba(X_test)
    pieces = pd.concat([model._predict_proba(part) for part in _chunks(X_test, chunk)])
    pd.testing.assert_frame_equal(whole, pieces, check_exact=False, rtol=0, atol=1e-12)
    np.testing.assert_allclose(whole.sum(axis=1), 1.0, atol=1e-9)
    assert list(whole.columns) == ["no", "yes"]
    labels = model._predict(X_test)
    pd.testing.assert_series_equal(labels, pd.concat([model._predict(part) for part in _chunks(X_test, chunk)]))
    assert set(labels) <= {"no", "yes"}


# -- 4. ensemble weights ----------------------------------------------------------------------


def _toy_oof(problem_type: str, n: int = 120, seed: int = 0):
    """Labels, a perfect out-of-fold prediction and two noisy ones, in AutoGluon's internal label space."""
    rng = np.random.default_rng(seed)
    if problem_type == "regression":
        y = rng.normal(size=n)
        return y, y.copy(), [y + rng.normal(0, s, n) for s in (0.5, 1.0)]
    k = 2 if problem_type == "binary" else 3
    y = rng.integers(0, k, n)
    onehot = np.eye(k)[y]
    noisy = []
    for s in (0.5, 1.0):
        p = onehot + rng.uniform(0, s, (n, k))
        p = p / p.sum(axis=1, keepdims=True)
        noisy.append(p)
    if problem_type == "binary":
        return y, onehot[:, 1], [p[:, 1] for p in noisy]
    return y, onehot, noisy


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_ensemble_weights_are_a_deterministic_distribution(problem_type):
    y, _, noisy = _toy_oof(problem_type)
    rng = np.random.default_rng(3)
    oof = [*noisy, *[np.clip(p + rng.normal(0, 0.05, p.shape), 1e-3, None) for p in noisy]]
    if problem_type == "multiclass":
        oof = [p / p.sum(axis=1, keepdims=True) for p in oof]
    metric = get_metric(METRICS[problem_type], problem_type)
    weights = ensemble_weights(oof, y, problem_type=problem_type, metric=metric)
    assert weights.shape == (len(oof),)
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0)
    again = ensemble_weights([p.copy() for p in oof], y.copy(), problem_type=problem_type, metric=metric)
    np.testing.assert_array_equal(weights, again)


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_ensemble_weights_put_everything_on_a_perfect_bag(problem_type):
    y, perfect, noisy = _toy_oof(problem_type)
    oof = [noisy[0], perfect, noisy[1]]
    weights = ensemble_weights(
        oof, y, problem_type=problem_type, metric=get_metric(METRICS[problem_type], problem_type)
    )
    np.testing.assert_array_equal(weights, [0.0, 1.0, 0.0])


def test_ensemble_weights_tie_breaking_is_deterministic():
    """AutoGluon breaks ties with a fixed seed: the same bag wins every time, whichever it is."""
    y, perfect, _ = _toy_oof("binary")
    metric = get_metric("log_loss", "binary")
    oof = [perfect.copy() for _ in range(5)]
    first = ensemble_weights(oof, y, problem_type="binary", metric=metric)
    np.testing.assert_array_equal(first, ensemble_weights(oof, y, problem_type="binary", metric=metric))
    assert sorted(first.tolist()) == [0.0, 0.0, 0.0, 0.0, 1.0]


# -- 5. registration --------------------------------------------------------------------------


def test_lgbm_prep_fte_is_registered_as_a_system():
    info = discover_systems()[NAME]
    assert info.system_cls is LGBMPrepFTESystemModel
    assert info.method_metadata.method_class == "system"
    assert info.method_metadata.compute == "cpu"
