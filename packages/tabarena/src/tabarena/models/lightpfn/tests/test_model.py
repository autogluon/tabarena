from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightpfn")
import torch
from lightpfn import Config, LightPFN, save_model

from tabarena.models.lightpfn.hpo import gen_lightpfn
from tabarena.models.lightpfn.model import LightPFNModel, prefetch_weights

# The first two tests use the released weights (downloaded at the pinned revision); the rest use tiny random
# weights through LIGHTPFN_WEIGHTS_DIR, with no download.


def test_uncached_predictions_match_the_cache(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=[f"f{i}" for i in range(4)])
    X["cat"] = pd.Categorical(rng.choice(["a", "b", "c"], size=300))
    y = pd.Series((X["f0"] + (X["cat"] == "a") > 0.5).astype(int))
    predictions = {}
    for cache_context in (True, False):
        model = LightPFNModel(
            path=str(tmp_path / str(cache_context)),
            name="m",
            problem_type="binary",
            eval_metric="log_loss",
            hyperparameters={"n_estimators": 2, "cache_context": cache_context},
        )
        model.fit(X=X.iloc[:250], y=y.iloc[:250], num_gpus=0)
        assert bool(model.model.members_) == cache_context
        # uncached fold inputs live on the wrapper as torch tensors, saved beside model.pkl
        assert model.model._uncached_data is None
        assert (model._uncached_context is None) == cache_context
        predictions[cache_context] = LightPFNModel.load(model.save()).predict_proba(X.iloc[250:])

    np.testing.assert_allclose(predictions[True], predictions[False], atol=1e-6)


@pytest.mark.parametrize(("n_classes", "many_class_threshold"), [(15, 10), (5, 3)])
def test_more_classes_than_the_head_are_output_coded(tmp_path, n_classes, many_class_threshold):
    pytest.importorskip("tabpfn_extensions.many_class")
    rng = np.random.default_rng(0)
    y = pd.Series(np.repeat(np.arange(n_classes), 20))
    X = pd.DataFrame(rng.normal(size=(len(y), 4)), columns=[f"f{i}" for i in range(4)])
    X["f0"] += y * 3.0
    X["cat"] = pd.Categorical(np.where(y % 2 == 0, "even", "odd"))
    train = rng.permutation(len(y))[: int(0.8 * len(y))]
    test = np.setdiff1d(np.arange(len(y)), train)
    model = LightPFNModel(
        path=str(tmp_path),
        name="m",
        problem_type="multiclass",
        eval_metric="log_loss",
        hyperparameters={"n_estimators": 1, "ag_args_fit": {"many_class_threshold": many_class_threshold}},
    )
    model.fit(X=X.iloc[train], y=y.iloc[train], num_gpus=0)
    assert model._use_many_class and model.get_device() == "cpu"

    proba = LightPFNModel.load(model.save()).predict_proba(X.iloc[test])
    assert proba.shape == (len(test), n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert (proba.argmax(axis=1) == y.iloc[test].to_numpy()).mean() > 0.8


@pytest.fixture
def weights(tmp_path, monkeypatch):
    """Tiny random inference weights, no download or optimization."""
    root = tmp_path / "weights"
    with torch.random.fork_rng():
        torch.manual_seed(8)
        for task, head in (("classifier", "classification"), ("regressor", "quantiles")):
            cfg = Config(
                col_dim=16,
                col_heads=2,
                col_blocks=1,
                n_inducing=3,
                row_blocks=1,
                row_heads=2,
                n_cls=2,
                icl_blocks=2,
                icl_heads=2,
                decoder_heads=2,
                label_slots=4,
                n_thinking=2,
                n_freq=2,
                n_ecdf_freq=1,
                row_mode="summary",
                cat_adapter=True,
            )
            if head == "quantiles":
                cfg.head = head
            save_model(LightPFN(cfg).eval(), root / task)
    monkeypatch.setenv("LIGHTPFN_WEIGHTS_DIR", str(root))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return root


def data(problem_type):
    rng = np.random.default_rng(16)
    X = pd.DataFrame(rng.normal(size=(48, 3)), columns=list("abc"))
    X["cat"] = pd.Categorical(["x", "y", "z"] * 16)
    X.loc[0, "a"] = np.nan
    n = 2 if problem_type == "binary" else 3
    y = rng.normal(size=len(X)) if problem_type == "regression" else np.arange(len(X)) % n
    return X, pd.Series(y)


def fit(tmp_path, name, problem_type, **hps):
    X, y = data(problem_type)
    model = LightPFNModel(path=str(tmp_path / name), name=name, problem_type=problem_type, hyperparameters=hps)
    before = torch.get_num_threads()
    model.fit(X=X, y=y, X_val=X.iloc[:4], y_val=y.iloc[:4], num_cpus=1, num_gpus=0)
    assert torch.get_num_threads() == before
    return model, X


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_cpu_fit_shared_weights_save_load(tmp_path, weights, problem_type):
    first, X = fit(tmp_path, "first", problem_type)
    second, _ = fit(tmp_path, "second", problem_type)
    if problem_type == "regression":
        assert first.model.model_ is second.model.model_
        assert first.model.regressor_.ctx_ is not second.model.regressor_.ctx_
        assert first.model.n_estimators == 2
    else:
        assert first.model.net_ is second.model.net_
        assert first.model.members_[0][1] is not second.model.members_[0][1]
        assert first.model.n_estimators == 4
    assert first.get_device() == "cpu"
    before = torch.get_num_threads()
    expected = first.predict_proba(X.iloc[:5])
    assert torch.get_num_threads() == before
    assert np.isfinite(expected).all()
    assert expected.shape == ((5, 3) if problem_type == "multiclass" else (5,))
    state = first.__getstate__()
    # AutoGluon removes network modules before pickling, but keeps the fitted context.
    assert len(pickle.dumps(state)) < 200_000
    restored = LightPFNModel.load(first.save())
    np.testing.assert_array_equal(expected, restored.predict_proba(X.iloc[:5]))
    restored.set_device("cpu")
    np.testing.assert_array_equal(expected, restored.predict_proba(X.iloc[:5]))
    unshared, _ = fit(tmp_path, "unshared", problem_type, ag_args_fit={"share_pretrained_weights": False})
    np.testing.assert_array_equal(expected, unshared.predict_proba(X.iloc[:5]))


def test_default_config_refit_and_metadata(tmp_path):
    from tabarena.models import (
        LightPFNModel as ExportedModel,
        get_model_registry,
    )
    from tabarena.website.website_format import get_model_family

    assert ExportedModel is LightPFNModel
    assert get_model_registry()["LightPFN"].model_cls is ExportedModel
    (config,) = gen_lightpfn.manual_configs
    fold = ExportedModel(path=str(tmp_path), name="fold", hyperparameters=config)
    refit = fold.convert_to_refit_full_template()
    assert fold.get_params()["hyperparameters"]["cache_context"] is False
    assert refit.get_params()["hyperparameters"]["cache_context"] is True
    assert get_model_family("TA-LIGHTPFN") == get_model_family("LIGHTPFN")


@pytest.mark.parametrize("cache", [False, True])
def test_classifier_cache_and_unknown_categories(tmp_path, weights, cache):
    model, X = fit(tmp_path, "m", "binary", n_estimators=1, cache_context=cache)
    assert bool(model.model.members_) == cache
    if not cache:
        assert model.model._uncached_data is None
        assert isinstance(model._uncached_context[0], torch.Tensor)
    query = X.iloc[:4].copy()
    query["cat"] = pd.Categorical(["unseen", "x", None, "y"])
    predictions = model.predict_proba(query)
    assert model.model._uncached_data is None
    assert np.isfinite(predictions).all()
    np.testing.assert_array_equal(predictions, LightPFNModel.load(model.save()).predict_proba(query))


def test_explicit_device_and_gpu_failure(tmp_path, weights, monkeypatch):
    # An explicit CPU is honored even when the allocator's default would be CUDA.
    monkeypatch.setattr(LightPFNModel, "_resolve_fit_device", lambda *a, **kw: pytest.fail("overrode explicit device"))
    model, _ = fit(tmp_path, "cpu", "regression", device="cpu")
    assert model.get_device() == "cpu"
    from lightpfn.device import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA/ROCm"):
        resolve_device("cuda")


def test_prefetch_local_both_tasks(weights):
    assert prefetch_weights() == str(weights)
    (weights / "regressor" / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        prefetch_weights()


def test_many_class_path_preserves_categories(tmp_path, weights):
    pytest.importorskip("tabpfn_extensions.many_class")
    X, _ = data("multiclass")
    y = pd.Series(np.arange(len(X)) % 12)
    # The tiny test network has four label slots, so the code alphabet has four symbols.
    model = LightPFNModel(
        path=str(tmp_path),
        name="ecoc",
        problem_type="multiclass",
        hyperparameters={"n_estimators": 1, "ag_args_fit": {"many_class_threshold": 4}},
    )
    model.fit(X=X, y=y, num_cpus=1, num_gpus=0)
    assert model._use_many_class
    assert str(model.model.estimator._dtypes["cat"]) == "category"
    query = X.iloc[:3].copy()
    query["cat"] = pd.Categorical(["unseen", "x", None])
    expected = model.predict_proba(query)
    assert expected.shape == (3, 12) and np.isfinite(expected).all()
    np.testing.assert_allclose(expected.sum(axis=1), 1, atol=1e-5)
    np.testing.assert_array_equal(expected, LightPFNModel.load(model.save()).predict_proba(query))


@pytest.mark.parametrize("problem_type", ["binary", "regression"])
def test_fp16_views_stay_shared_and_out_of_pickle(tmp_path, weights, problem_type):
    from lightpfn.precision import shared_fp16_view

    model, X = fit(tmp_path, "half", problem_type)
    net = model.model.model_ if problem_type == "regression" else model.model.net_
    engine = model.model.regressor_ if problem_type == "regression" else model.model
    half = shared_fp16_view(net)
    size = len(pickle.dumps(model.__getstate__()))
    engine._fp16_net = half  # inspect storage on CPU without executing a GPU fp16 path
    assert len(pickle.dumps(model.__getstate__())) == size
    assert "_fp16_net" not in vars(engine)
    assert shared_fp16_view(net) is half
    expected = model.predict_proba(X.iloc[:4])
    np.testing.assert_array_equal(expected, LightPFNModel.load(model.save()).predict_proba(X.iloc[:4]))


@pytest.mark.parametrize("problem_type", ["binary", "regression"])
def test_device_move_detaches_half_view_before_autogluon(tmp_path, weights, monkeypatch, problem_type):
    from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel
    from lightpfn.precision import shared_fp16_view

    model, _ = fit(tmp_path, "device", problem_type)
    engine = getattr(model.model, "regressor_", model.model)
    net = model.model.model_ if problem_type == "regression" else model.model.net_
    half = shared_fp16_view(net)
    engine._fp16_net = half
    engine.precision_ = "fp16"

    def check_detached(self, device):
        assert "_fp16_net" not in vars(engine)
        assert device == "cpu"

    monkeypatch.setattr(AbstractTorchModel, "set_device", check_detached)
    model.set_device("cpu")
    assert engine.precision_ == "fp32"
    assert shared_fp16_view(net) is half


def test_unshared_fit_restores_global_rng(tmp_path, weights):
    before = torch.random.get_rng_state().clone()
    fit(tmp_path, "rng", "regression", ag_args_fit={"share_pretrained_weights": False})
    assert torch.equal(before, torch.random.get_rng_state())


def test_absent_class_keeps_global_column_order(tmp_path, weights):
    X, _ = data("multiclass")
    y = pd.Series(np.resize([0, 2, 3], len(X)))
    model = LightPFNModel(
        path=str(tmp_path), name="missing", problem_type="multiclass", hyperparameters={"n_estimators": 1}
    )
    model.fit(X=X, y=y, num_classes=4, num_cpus=1, num_gpus=0)
    expected = model.model.predict_proba(X.iloc[:4])
    proba = model.predict_proba(X.iloc[:4])
    assert proba.shape == (4, 4)
    np.testing.assert_allclose(proba[:, [0, 2, 3]], expected)
    assert np.all(proba[:, 1] == 0)
