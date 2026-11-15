from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fintfm")
torch = pytest.importorskip("torch")

from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM, ModelConfig

from tabarena.models.fintfm_linprobe.model import PROBE_CROSSFIT_FOLDS, FinTFMLinearProbeModel

N_TRAIN = 240
N_TEST = 40
MAX_CONTEXT = 60
PARTITION = {"n_bags": 2, "max_depth": 1, "min_samples_leaf": 0.1, "min_leaf_rows": 30, "min_class_rows": 2}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "FINTFM_PARTITION",
        "FINTFM_PROBE",
        "FINTFM_PARAMS",
        "FINTFM_REG_BINS",
        "FINTFM_REG_THRESHOLDS",
        "FINTFM_MAX_CONTEXT",
        "FINTFM_CHECKPOINT",
        "FINTFM_CHECKPOINT_MULTICLASS",
        "FINTFM_CHECKPOINT_REGRESSION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FINTFM_DEVICE", "cpu")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    """An untrained network small enough to fit and score in a second."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    cfg = ModelConfig(
        max_features=8, max_classes=10, d_cell=8, d_model=16, n_layers=1, n_col_layers=1, n_heads=2, d_ff=32
    )
    FinancialTFM(cfg).save(str(path))
    return str(path)


def data(problem_type: str, seed: int = 0, categorical: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_TEST
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    signal = X["f0"] + 0.5 * X["f1"] + rng.normal(scale=0.5, size=n)
    if categorical:
        X["cat"] = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    if problem_type == "binary":
        y = (signal > 0).astype(int)
    elif problem_type == "multiclass":
        y = pd.Series(np.digitize(signal, [-0.6, 0.6]))
    else:
        y = signal
    y = pd.Series(np.asarray(y))
    return X.iloc[:N_TRAIN].reset_index(drop=True), y.iloc[:N_TRAIN].reset_index(drop=True), X.iloc[N_TRAIN:]


def make(problem_type: str, tmp_path, checkpoint: str, **hyperparameters) -> FinTFMLinearProbeModel:
    hp = {
        f"checkpoint_{problem_type}": checkpoint,
        "max_context": MAX_CONTEXT,
        "feature_transform": "squash",
        "max_features_strategy": "select",
    }
    hp.update(hyperparameters)
    model = FinTFMLinearProbeModel(
        path=str(tmp_path),
        name="fintfm_crossfit",
        problem_type=problem_type,
        eval_metric={"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}[problem_type],
        hyperparameters=hp,
    )
    model._init_params()
    return model


class Spy:
    """Records every row that a classifier encodes, with the context it encodes it against."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.restricted: list[np.ndarray] = []
        original = FinancialTFMClassifier.transform_representation
        restrict = FinancialTFMClassifier.restrict_context
        spy = self

        def transform(clf, X, stage="encode_rows"):
            ctx_values = np.asarray(clf._raw_X, dtype=np.float32)[np.asarray(clf.context_indices_)]
            spy.calls.append((np.asarray(X, dtype=np.float32).copy(), ctx_values, np.asarray(clf.context_indices_)))
            return original(clf, X, stage=stage)

        def restrict_context(clf, rows):
            spy.restricted.append(np.asarray(rows))
            return restrict(clf, rows)

        monkeypatch.setattr(FinancialTFMClassifier, "transform_representation", transform)
        monkeypatch.setattr(FinancialTFMClassifier, "restrict_context", restrict_context)

    def rows_seen_in_own_context(self) -> int:
        """Encoded rows that equal a row of the context they were encoded against."""
        n = 0
        for X, ctx_values, _ in self.calls:
            ctx = {row.tobytes() for row in ctx_values}
            n += sum(row.tobytes() in ctx for row in X)
        return n


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("partitioned", [False, True])
def test_no_readout_row_is_encoded_against_a_context_holding_it(
    problem_type, partitioned, tmp_path, checkpoint, monkeypatch
):
    X, y, _ = data(problem_type)
    hp = {"probe_crossfit_folds": 5}
    hp[f"partition_{problem_type}"] = dict(PARTITION) if partitioned else None
    model = make(problem_type, tmp_path, checkpoint, **hp)
    spy = Spy(monkeypatch)
    model.fit(X=X, y=y)
    assert spy.calls, "the readout encoded nothing"
    assert spy.rows_seen_in_own_context() == 0
    # Every restriction leaves some context rows out and keeps some.
    assert spy.restricted
    stored = [np.asarray(c[2]) for c in spy.calls]
    for rows in spy.restricted:
        assert rows.size >= 1
    assert all(len(np.unique(rows)) == len(rows) for rows in spy.restricted)
    assert model._train_matrix is None
    assert model._crossfit_stats["restricted"] >= 5
    if partitioned:
        # Leaves with their own probe were cross-fitted too, not only the global estimator.
        leaf_probes = model._ridge_leaf_probes if problem_type == "regression" else model._leaf_probes
        n_leaf = sum(p is not None for leaves in leaf_probes for p in leaves.values())
        assert n_leaf >= 1
        assert len({tuple(c[2]) for c in spy.calls}) > 5
    assert len(stored) == len(spy.calls)


@pytest.mark.parametrize("problem_type", ["binary", "regression"])
def test_without_crossfit_the_spy_finds_the_leak(problem_type, tmp_path, checkpoint, monkeypatch):
    """The positive control: the spy does detect a row encoded against its own labelled copy."""
    X, y, _ = data(problem_type)
    model = make(problem_type, tmp_path, checkpoint, probe_crossfit_folds=0)
    spy = Spy(monkeypatch)
    model.fit(X=X, y=y)
    assert spy.rows_seen_in_own_context() > 0
    assert not spy.restricted


@pytest.mark.parametrize("folds", [0, 1])
def test_disabled_crossfit_reproduces_the_earlier_readout_inputs(folds, tmp_path, checkpoint, monkeypatch):
    """With 0 or 1 fold the readout input is the plain representation of the replayed matrix."""
    X, y, _ = data("binary", categorical=True)
    model = make("binary", tmp_path, checkpoint, probe_crossfit_folds=folds, probe_binary={})
    seen = []
    original = FinTFMLinearProbeModel._probe_input

    def record(self, est, Xq, fit_idx=None):
        out = original(self, est, Xq, fit_idx)
        seen.append((est, np.asarray(Xq).copy(), fit_idx, out))
        return out

    monkeypatch.setattr(FinTFMLinearProbeModel, "_probe_input", record)
    spy = Spy(monkeypatch)
    model.fit(X=X, y=y)
    assert len(seen) == 1
    est, Xq, fit_idx, out = seen[0]
    assert fit_idx is None
    assert not spy.restricted
    expected = np.asarray(est.transform_representation(Xq, stage="pre_head"), dtype=np.float64)
    assert np.array_equal(out, expected)
    # The rows are the replayed full-statistics encoding.
    replay = np.asarray(model.preprocess(X, is_train=False), dtype=np.float32)
    assert np.array_equal(Xq.astype(np.float32), replay[: len(Xq)])
    assert model._crossfit_on is False
    assert model._train_matrix is None


def test_crossfit_rows_use_out_of_fold_categorical_encoding(tmp_path, checkpoint, monkeypatch):
    X, y, _ = data("binary", categorical=True)
    model = make("binary", tmp_path, checkpoint, probe_crossfit_folds=3, probe_binary={})
    captured = {}
    original = FinTFMLinearProbeModel._probe_input

    def record(self, est, Xq, fit_idx=None):
        captured["Xq"] = np.asarray(Xq).copy()
        captured["fit_idx"] = fit_idx
        return original(self, est, Xq, fit_idx)

    monkeypatch.setattr(FinTFMLinearProbeModel, "_probe_input", record)
    model.fit(X=X, y=y)
    replay = np.asarray(model.preprocess(X, is_train=False), dtype=np.float32)
    cat_col = list(X.columns).index("cat")
    # Out-of-fold encoding differs from the full-statistics encoding for at least some rows.
    assert not np.array_equal(captured["Xq"][:, cat_col], replay[:, cat_col])
    # The numeric columns are the same either way.
    numeric = [i for i in range(replay.shape[1]) if i != cat_col]
    assert np.array_equal(captured["Xq"][:, numeric], replay[:, numeric])
    assert captured["fit_idx"] is not None


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("folds", [PROBE_CROSSFIT_FOLDS, 0])
def test_fit_and_predict_end_to_end(problem_type, partitioned, folds, tmp_path, checkpoint):
    X, y, X_test = data(problem_type, categorical=True)
    hp = {"probe_crossfit_folds": folds}
    hp[f"partition_{problem_type}"] = dict(PARTITION) if partitioned else None
    model = make(problem_type, tmp_path, checkpoint, **hp)
    model.fit(X=X, y=y)
    if problem_type == "regression":
        pred = model.predict(X_test)
        assert pred.shape == (N_TEST,)
        assert np.isfinite(pred).all()
    else:
        proba = model.predict_proba(X_test)
        n_classes = 2 if problem_type == "binary" else 3
        if problem_type == "multiclass":
            assert proba.shape == (N_TEST, n_classes)
            assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        else:
            assert proba.shape == (N_TEST,)
        assert np.isfinite(proba).all()
    if partitioned:
        assert model._leaf_probes is not None or model._ridge_leaf_probes is not None


def test_default_folds_and_validation(tmp_path, checkpoint):
    assert PROBE_CROSSFIT_FOLDS == 5
    model = make("binary", tmp_path, checkpoint)
    params = model._get_model_params()
    assert params["probe_crossfit_folds"] == 5
    model._resolve_entry(params)
    assert "probe_crossfit_folds" not in params
    assert model._crossfit_folds == 5
    for bad in (-1, 2.5, True, "5"):
        m = make("binary", tmp_path, checkpoint, probe_crossfit_folds=bad)
        with pytest.raises(ValueError, match="probe_crossfit_folds"):
            m._resolve_entry(m._get_model_params())


def test_head_option_is_refused_with_crossfit_and_allowed_without(tmp_path, checkpoint):
    m = make("binary", tmp_path, checkpoint, probe_binary={"head": True}, probe_crossfit_folds=5)
    with pytest.raises(ValueError, match="head"):
        m._resolve_entry(m._get_model_params())
    m = make("binary", tmp_path, checkpoint, probe_binary={"head": True}, probe_crossfit_folds=0)
    m._resolve_entry(m._get_model_params())
