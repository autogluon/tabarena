"""``FINTFM_PARTITION`` and ``partition_<type>`` wrap the FinTFM classifier in tree-partitioned contexts (fintfm fix 4)."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fintfm")
torch = pytest.importorskip("torch")

from fintfm.inference.large_n import PartitionedTFMClassifier
from fintfm.modeling.model import FinancialTFM, ModelConfig

from tabarena.models.fintfm.model import FinTFMModel

ENV_VARS = (
    "FINTFM_PARTITION",
    "FINTFM_PARAMS",
    "FINTFM_CHECKPOINT",
    "FINTFM_CHECKPOINT_MULTICLASS",
    "FINTFM_CHECKPOINT_REGRESSION",
    "FINTFM_REG_THRESHOLDS",
    "FINTFM_REG_BINS",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FINTFM_DEVICE", "cpu")


def _model(problem_type: str = "binary", **hyperparameters) -> FinTFMModel:
    model = FinTFMModel(
        path="",
        name="fintfm_test",
        problem_type=problem_type,
        eval_metric={"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}[problem_type],
        hyperparameters=hyperparameters,
    )
    model._init_params()  # AutoGluon calls this when a fit starts
    return model


def _partition(model: FinTFMModel) -> dict | None:
    """The partition setting the model resolves for its problem type."""
    return model._resolve_entry(model._get_model_params())["partition"]


@pytest.mark.parametrize("raw", [None, "", "  "])
def test_unset_or_blank_leaves_the_classifier_unpartitioned(monkeypatch: pytest.MonkeyPatch, raw: str | None) -> None:
    """The submitted default config never partitions, for any problem type."""
    if raw is not None:
        monkeypatch.setenv("FINTFM_PARTITION", raw)
    assert FinTFMModel._env_entry_overrides() == {}
    for problem_type in ("binary", "multiclass", "regression"):
        assert _partition(_model(problem_type)) is None


def test_empty_object_takes_the_class_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """``{}`` switches partitioning on with §169's depth 2 and 5% leaf floor."""
    monkeypatch.setenv("FINTFM_PARTITION", "{}")
    assert _partition(_model()) == {}


def test_named_arguments_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINTFM_PARTITION", '{"max_depth": 1, "min_leaf_rows": 100}')
    over = FinTFMModel._env_entry_overrides()
    assert {over[f"partition_{pt}"]["max_depth"] for pt in ("binary", "multiclass", "regression")} == {1}
    assert _partition(_model()) == {"max_depth": 1, "min_leaf_rows": 100}


def test_null_switches_the_partition_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINTFM_PARTITION", "null")
    assert FinTFMModel._env_entry_overrides()["partition_binary"] is None
    assert _partition(_model()) is None


def test_explicit_hyperparameter_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINTFM_PARTITION", '{"max_depth": 1}')
    assert _partition(_model(partition_binary={"max_depth": 3})) == {"max_depth": 3}
    assert _partition(_model(partition_binary=None)) is None


def test_threshold_regression_is_not_partitioned_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINTFM_REG_THRESHOLDS", "0.1,0.5")
    assert FinTFMModel._env_entry_overrides() == {"partition_regression": None}
    monkeypatch.setenv("FINTFM_PARTITION", "{}")
    assert FinTFMModel._env_entry_overrides()["partition_regression"] == {}


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "JSON object"),
        ('{"max_detph": 2}', "max_detph"),
        ('{"random_state": 3}', "random_state"),
        ('{"estimator": null}', "estimator"),
    ],
)
def test_bad_values_raise(monkeypatch: pytest.MonkeyPatch, raw: str, match: str) -> None:
    """A misspelled or reserved key raises rather than running the default under the arm's name."""
    monkeypatch.setenv("FINTFM_PARTITION", raw)
    with pytest.raises(ValueError, match=match):
        _partition(_model())


def test_bad_hyperparameter_raises() -> None:
    with pytest.raises(ValueError, match="max_detph"):
        _partition(_model(partition_binary={"max_detph": 2}))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    """An untrained network small enough to fit in a second."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    cfg = ModelConfig(
        max_features=8, max_classes=10, d_cell=8, d_model=16, n_layers=1, n_col_layers=1, n_heads=2, d_ff=32
    )
    FinancialTFM(cfg).save(str(path))
    return str(path)


def _fit(checkpoint: str, monkeypatch: pytest.MonkeyPatch, **hyperparameters) -> tuple[FinTFMModel, list[dict]]:
    """Fit on a toy binary table, returning the model and the arguments of every PartitionedTFMClassifier built."""
    calls: list[dict] = []
    original = PartitionedTFMClassifier.__init__

    @functools.wraps(original)
    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(PartitionedTFMClassifier, "__init__", spy)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + 0.3 * rng.normal(size=200) > 0).astype(int))
    model = _model(checkpoint_binary=checkpoint, max_context=50, n_ensemble=1, **hyperparameters)
    model.fit(X=X, y=y)
    return model, calls


def test_partition_configuration_reaches_the_partitioned_classifier(
    checkpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINTFM_PARTITION", '{"max_depth": 1, "min_leaf_rows": 30}')
    model, calls = _fit(checkpoint, monkeypatch, random_state=7)
    assert len(calls) == 1
    assert calls[0]["max_depth"] == 1
    assert calls[0]["min_leaf_rows"] == 30
    assert calls[0]["random_state"] == 7
    assert isinstance(model.model, PartitionedTFMClassifier)


def test_partition_hyperparameter_reaches_the_partitioned_classifier(
    checkpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, calls = _fit(checkpoint, monkeypatch, partition_binary={"max_depth": 1})
    assert [c["max_depth"] for c in calls] == [1]


def test_default_fit_builds_no_partitioned_classifier(checkpoint: str, monkeypatch: pytest.MonkeyPatch) -> None:
    model, calls = _fit(checkpoint, monkeypatch)
    assert calls == []
    assert not isinstance(model.model, PartitionedTFMClassifier)


class _Net:
    """Stands in for the network: records where it was moved."""

    def __init__(self) -> None:
        self.moved_to: list[str] = []

    def to(self, device: str) -> None:
        self.moved_to.append(device)


class _Clf:
    def __init__(self, net: _Net) -> None:
        self.model = net
        self.device = "mps"


class _Partitioned:
    def __init__(self, net: _Net) -> None:
        self.estimator = _Clf(net)
        self.global_estimator_ = _Clf(net)
        self.leaf_estimators_ = [{1: _Clf(net), 2: None, 3: _Clf(net)}]


def test_set_device_reaches_every_inner_classifier() -> None:
    """Every leaf, the fallback and the template follow a device change; a fallback leaf is skipped."""
    model = FinTFMModel(path="", name="fintfm_test", problem_type="binary", eval_metric="roc_auc")
    net = _Net()
    model.model = _Partitioned(net)
    model._set_device("cpu")
    inner = model._inner_classifiers()
    assert len(inner) == 4
    assert all(clf.device == "cpu" for clf in inner)
    assert np.unique(net.moved_to).tolist() == ["cpu"]
