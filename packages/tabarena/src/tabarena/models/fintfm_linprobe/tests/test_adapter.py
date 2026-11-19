"""The AutoGluon glue of FinTFM-LinProbe: defaults, device choice, class limit, preprocessing, one real fit.

The readout itself is tested in fintfm (``tests/test_readout_*.py``); these tests cover only what the
adapter adds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fintfm")
torch = pytest.importorskip("torch")

from fintfm.modeling.model import FinancialTFM, ModelConfig
from fintfm.readout import entry_defaults

from tabarena.models.fintfm_linprobe.hpo import proposed_search_space_additions, search_space
from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel

METRICS = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}
PER_TYPE = ("checkpoint", "partition", "probe")


def make(problem_type: str, tmp_path, **hyperparameters) -> FinTFMLinearProbeModel:
    """An initialised, unfitted adapter of ``problem_type``."""
    model = FinTFMLinearProbeModel(
        path=str(tmp_path),
        name="fintfm_adapter",
        problem_type=problem_type,
        eval_metric=METRICS[problem_type],
        hyperparameters=hyperparameters,
    )
    model._init_params()  # AutoGluon calls this when a fit starts
    return model


def test_supports_all_three_problem_types() -> None:
    assert FinTFMLinearProbeModel._supported_problem_types == ["binary", "multiclass", "regression"]


def test_the_class_limit_is_declared_to_autogluon() -> None:
    """The multiclass checkpoint represents at most ten classes; ``ag.max_classes`` says so before a fit."""
    assert FinTFMLinearProbeModel._default_auxiliary_params_extra["max_classes"] == 10


@pytest.mark.parametrize("problem_type", sorted(METRICS))
def test_default_hyperparameters_are_the_librarys_entry(problem_type, tmp_path) -> None:
    """One source of truth: the adapter's defaults are ``entry_defaults`` with the problem type in three names."""
    params = make(problem_type, tmp_path)._get_model_params()
    expected = entry_defaults(problem_type)
    for option in PER_TYPE:
        expected[f"{option}_{problem_type}"] = expected.pop(option)
    expected["blas_threads"] = "auto"
    assert {k: v for k, v in params.items() if k != "random_state"} == expected


@pytest.mark.parametrize("problem_type", sorted(METRICS))
def test_estimator_settings_keep_only_this_problem_types_options(problem_type, tmp_path) -> None:
    """Search spaces name every problem type's options; the library receives the one that applies."""
    params = make(
        problem_type, tmp_path, probe_multiclass={"scale": True}, partition_binary={"n_bags": 3}
    )._get_model_params()
    params.pop("blas_threads")
    params.pop("random_state", None)
    settings = FinTFMLinearProbeModel._estimator_settings(params, problem_type)
    entry = entry_defaults(problem_type)
    assert settings["checkpoint"] == entry["checkpoint"]
    assert settings["probe"] == ({"scale": True} if problem_type == "multiclass" else entry["probe"])
    assert settings["partition"] == ({"n_bags": 3} if problem_type == "binary" else entry["partition"])
    assert not any(k.endswith(("_binary", "_multiclass", "_regression")) for k in settings)


def test_search_space_keys_are_read_and_its_first_values_are_the_entry(tmp_path) -> None:
    for problem_type in sorted(METRICS):
        params = make(problem_type, tmp_path)._get_model_params()
        for key, space in search_space.items():
            if not key.endswith(f"_{problem_type}"):
                continue
            assert space.default == params[key], f"the first value of {key} is not the measured entry's"
            assert params[key] in list(space.data)
    assert set(proposed_search_space_additions) == {"probe_crossfit_folds"}
    assert set(proposed_search_space_additions) <= set(make("binary", tmp_path)._get_model_params())


@pytest.mark.parametrize(
    ("raw", "expected"), [(None, None), ("", None), (" cpu ", "cpu"), ("mps", "mps"), ("cuda", "cuda")]
)
def test_forced_device_reads_the_variable(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: str | None
) -> None:
    """Unset or blank leaves the choice automatic; a named device is returned stripped."""
    monkeypatch.delenv("FINTFM_DEVICE", raising=False)
    if raw is not None:
        monkeypatch.setenv("FINTFM_DEVICE", raw)
    assert FinTFMLinearProbeModel._forced_device() == expected


def test_forced_device_refuses_an_unknown_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misspelled device raises rather than silently falling back to automatic choice."""
    monkeypatch.setenv("FINTFM_DEVICE", "gpu")
    with pytest.raises(ValueError, match="FINTFM_DEVICE"):
        FinTFMLinearProbeModel._forced_device()


@pytest.mark.parametrize(
    ("num_gpus", "cuda", "mps", "expected"),
    [
        (1, True, False, "cuda"),
        (0, True, False, "cpu"),
        (1, False, True, "cpu"),
        (0, False, True, "cpu"),
        (0, False, False, "cpu"),
    ],
)
def test_choose_device_never_picks_mps_on_its_own(
    monkeypatch: pytest.MonkeyPatch, num_gpus: int, cuda: bool, mps: bool, expected: str
) -> None:
    """CUDA only when a GPU was granted and CUDA exists; otherwise CPU, even on Apple Silicon."""
    monkeypatch.delenv("FINTFM_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert FinTFMLinearProbeModel._choose_device(num_gpus) == expected


def test_choose_device_follows_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """``FINTFM_DEVICE=mps`` is the only way to fit on MPS."""
    monkeypatch.setenv("FINTFM_DEVICE", "mps")
    assert FinTFMLinearProbeModel._choose_device(0) == "mps"


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    """An untrained network small enough to fit and score in a second."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    cfg = ModelConfig(
        max_features=8, max_classes=10, d_cell=8, d_model=16, n_layers=1, n_col_layers=1, n_heads=2, d_ff=32
    )
    FinancialTFM(cfg).save(str(path))
    return str(path)


@pytest.mark.parametrize("problem_type", sorted(METRICS))
def test_fit_and_predict_through_autogluon_preprocessing(problem_type, tmp_path, tiny_checkpoint, monkeypatch) -> None:
    """A real fit on a table with a categorical column and a missing value, through ``model.fit``.

    The adapter label-encodes the categorical column and keeps NaN; the library target-encodes it.
    Binary predictions are the positive-class column, multiclass rows sum to one, regression is a vector.
    """
    monkeypatch.setenv("FINTFM_DEVICE", "cpu")
    rng = np.random.default_rng(0)
    n = 160
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    X["cat"] = pd.Categorical(rng.choice(["u", "v", "w"], size=n))
    X.loc[::9, "b"] = np.nan
    signal = X["a"] + 0.5 * X["c"].fillna(0) + rng.normal(scale=0.5, size=n)
    y = {
        "binary": pd.Series((signal > 0).astype(int)),
        "multiclass": pd.Series(np.digitize(signal, [-0.6, 0.6])),
        "regression": pd.Series(signal),
    }[problem_type]
    hp = {
        f"checkpoint_{problem_type}": tiny_checkpoint,
        f"partition_{problem_type}": {"n_bags": 2, "max_depth": 1, "min_leaf_rows": 20},
        "max_context": 50,
        "probe_crossfit_folds": 3,
    }
    model = make(problem_type, tmp_path, **hp)
    model.fit(X=X.iloc[:120].reset_index(drop=True), y=y.iloc[:120].reset_index(drop=True))
    out = (
        model.predict_proba(X.iloc[120:].reset_index(drop=True))
        if problem_type != "regression"
        else model.predict(X.iloc[120:].reset_index(drop=True))
    )
    assert out.shape == ((40, 3) if problem_type == "multiclass" else (40,))
    assert np.isfinite(out).all()
    if problem_type == "multiclass":
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)
    assert model.get_device() == "cpu"
