"""The contamination audit must pass an honest learner, pass an honest but order-sensitive one, and fail a
model that remembers the published test table in its published column order.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tabarena.tools.audit_contamination import (
    FAIL,
    INFO,
    PASS,
    PUBLISHED,
    REVERSED,
    SKIP,
    VERDICT_CHECK,
    WARN,
    CheckResult,
    DatasetSplits,
    ModelUnderAudit,
    audit_contamination,
    augmentation_off,
    calibrate_order,
    column_orders,
    dataset_verdicts,
    error_of,
    group_probe_data,
    infer_problem_type,
    leakage_tuple,
    log_ratio,
    main,
    make_synthetic_split,
    order_statistics,
    permute_columns,
    probe_ensemble,
    probe_order,
    strip_limix_config,
    write_json,
)

pytest.importorskip("sklearn")
pytest.importorskip("autogluon.core")

from autogluon.core.models import AbstractModel


@pytest.fixture(autouse=True)
def _few_threads():
    """The fake models fit sklearn models; OpenMP over every core of a large machine crawls."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(4):
        yield


def _encode(X: pd.DataFrame) -> np.ndarray:
    """Numeric matrix with the columns in name order, so the encoding ignores the column order."""
    out = X[sorted(X.columns, key=str)].copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(str).map(lambda v: sum(map(ord, v)) % 7)
    return out.to_numpy(dtype=float)


def _row_keys(X: pd.DataFrame, *, as_received: bool) -> list[tuple]:
    """One hashable key per row: the values in the received column order, or sorted (order-free)."""
    values = X.astype(str).to_numpy()
    return [tuple(row) if as_received else tuple(sorted(row)) for row in values]


class _HonestModel(AbstractModel):
    """Gradient boosting on the encoded features; the column order does not reach it."""

    ag_key = "_AUDIT_HONEST"
    ag_name = "_AuditHonest"
    max_iter = 60

    def _fit(self, X, y, **kwargs):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        X = self.preprocess(X, is_train=True)
        cls = HistGradientBoostingRegressor if self.problem_type == "regression" else HistGradientBoostingClassifier
        self.model = cls(random_state=0, max_iter=self.max_iter).fit(self._features_of(X), y)

    def _features_of(self, X: pd.DataFrame) -> np.ndarray:
        return _encode(X)

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X)
        if self.problem_type == "regression":
            return self.model.predict(self._features_of(X))
        return self._convert_proba_to_unified_form(self.model.predict_proba(self._features_of(X)))


class _OrderSensitiveHonest(_HonestModel):
    """Honest, but reads only the first half of the columns as received: order matters on every dataset."""

    ag_key = "_AUDIT_ORDER_SENSITIVE"
    ag_name = "_AuditOrderSensitive"

    def _features_of(self, X: pd.DataFrame) -> np.ndarray:
        half = list(X.columns)[: max(1, X.shape[1] // 2)]
        return _encode(X[half])


class _Memorizer(_HonestModel):
    """Honest, except on the rows it remembers: those get their remembered label at 0.99 (or exactly)."""

    ag_key = "_AUDIT_MEMORIZER"
    ag_name = "_AuditMemorizer"
    LOOKUP: dict = {}
    as_received = True

    def _predict_proba(self, X, **kwargs):
        X_pre = self.preprocess(X)
        pred = np.asarray(super()._predict_proba(X), dtype=float)
        keys = _row_keys(X_pre, as_received=self.as_received)
        for i, key in enumerate(keys):
            if key not in self.LOOKUP:
                continue
            label = self.LOOKUP[key]
            if self.problem_type == "regression":
                pred[i] = label
            elif self.problem_type == "binary":
                pred[i] = 0.99 if label == 1 else 0.01
            else:
                pred[i] = 0.01 / (pred.shape[1] - 1)
                pred[i, label] = 0.99
        return pred


class _OrderMemorizer(_Memorizer):
    """Remembers rows in their published column order: a permutation breaks the memory."""

    ag_key = "_AUDIT_ORDER_MEMORIZER"
    ag_name = "_AuditOrderMemorizer"
    LOOKUP: dict = {}
    as_received = True


class _ContextIgnorer(_Memorizer):
    """Remembers rows whatever the column order, and ignores the context labels on them."""

    ag_key = "_AUDIT_CONTEXT_IGNORER"
    ag_name = "_AuditContextIgnorer"
    LOOKUP: dict = {}
    as_received = False


def _synthetic(name: str, problem_type: str, seed: int, n_samples: int = 600):
    data = make_synthetic_split(problem_type, seed=seed, n_samples=n_samples)
    data.name = name
    return data


def _memorize(model_cls, data, *, as_received: bool) -> None:
    """Fill ``model_cls.LOOKUP`` with the test rows of ``data`` and their cleaned labels."""
    from autogluon.core.data.label_cleaner import LabelCleaner

    if data.problem_type == "regression":
        labels = data.y_test.astype(float).to_numpy()
    else:
        cleaner = LabelCleaner.construct(problem_type=data.problem_type, y=data.y_train, y_uncleaned=data.y_test)
        labels = cleaner.transform(data.y_test).to_numpy()
    model_cls.LOOKUP = dict(zip(_row_keys(data.X_test, as_received=as_received), labels, strict=True))


def _five_datasets():
    return [
        _synthetic("synth-a", "binary", 1),
        _synthetic("synth-b", "binary", 2),
        _synthetic("synth-c", "multiclass", 3),
        _synthetic("synth-d", "binary", 4),
        _synthetic("synth-e", "regression", 5),
    ]


def _by_check(results, check: str) -> dict[str, CheckResult]:
    return {r.dataset: r for r in results if r.check == check}


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------
def test_column_orders_are_permutations_and_deterministic():
    columns = [f"f{i}" for i in range(6)]
    orders = column_orders(columns, n_random=3, seed=7, dataset="x")
    assert list(orders) == [PUBLISHED, REVERSED, "random0", "random1", "random2"]
    assert orders[PUBLISHED] == columns
    assert orders[REVERSED] == columns[::-1]
    for order in orders.values():
        assert sorted(order) == columns
    assert orders == column_orders(columns, n_random=3, seed=7, dataset="x")
    assert orders["random0"] != column_orders(columns, n_random=3, seed=8, dataset="x")["random0"]


def test_permute_columns_round_trips_and_rejects_non_permutations():
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    out = permute_columns(X, ["c", "a", "b"])
    assert list(out.columns) == ["c", "a", "b"]
    pd.testing.assert_frame_equal(permute_columns(out, ["a", "b", "c"]), X)
    with pytest.raises(ValueError, match="permutation"):
        permute_columns(X, ["a", "b"])


def test_log_ratio_is_symmetric_and_floored():
    assert log_ratio(0.2, 0.2) == 0.0
    assert log_ratio(0.2, 0.1) == pytest.approx(-log_ratio(0.1, 0.2))
    assert log_ratio(0.1536, 0.0066) == pytest.approx(3.147, abs=1e-3)
    assert log_ratio(0.1, 0.0) == pytest.approx(log_ratio(0.1, 1e-3))


def test_error_of_each_metric():
    binary = _synthetic("b", "binary", 1, n_samples=200)
    classes = sorted({str(v) for v in binary.y_test})
    perfect = pd.DataFrame(
        {c: (binary.y_test.astype(str) == c).astype(float) for c in classes}, index=binary.X_test.index
    )
    assert error_of(binary, perfect) == pytest.approx(0.0)
    regression = _synthetic("r", "regression", 1, n_samples=200)
    assert error_of(regression, regression.y_test.to_numpy()) == pytest.approx(0.0)
    multiclass = _synthetic("m", "multiclass", 1, n_samples=200)
    uniform = pd.DataFrame(1 / 3, index=multiclass.X_test.index, columns=sorted({str(v) for v in multiclass.y_test}))
    assert error_of(multiclass, uniform) == pytest.approx(np.log(3))


def test_order_statistics_rank_and_minimum():
    err = {PUBLISHED: {"0": 0.01, "1": 0.02}, REVERSED: {"0": 0.2, "1": 0.3}, "random0": {"0": 0.1, "1": 0.2}}
    stats = order_statistics(err, eps=1e-3)
    assert stats["rank_pub"] == 1.0
    assert stats["rank_p"] == pytest.approx(1 / 3)
    assert stats["d_min"] == pytest.approx(np.mean([np.log(10), np.log(10)]))
    assert stats["d_mean"] > stats["d_min"]
    assert stats["n_splits"] == 2
    flat = {PUBLISHED: {"0": 0.1}, REVERSED: {"0": 0.1}, "random0": {"0": 0.1}}
    assert order_statistics(flat, eps=1e-3)["rank_pub"] == 2.0


def test_infer_problem_type():
    assert infer_problem_type(pd.Series(["a", "b", "a"])) == "binary"
    assert infer_problem_type(pd.Series([0, 1, 2, 1])) == "multiclass"
    assert infer_problem_type(pd.Series(["x", "y", "z"])) == "multiclass"
    assert infer_problem_type(pd.Series(np.linspace(0, 1, 50))) == "regression"


def test_strip_limix_config_keeps_order_preserving_members():
    member = {
        "FeatureShuffler": {"mode": "shuffle"},
        "PolynomialInteractionGenerator": {"max_interaction_features": 50},
        "RebalanceFeatureDistribution": {"worker_tags": ["power"], "svd_tag": "svd", "original_flag": True},
    }
    config = {
        "feature_view_strategy": "disabled",
        "pipelines": [
            {**member, "FeatureShuffler": {"mode": "rotate"}},
            {**member, "FeatureShuffler": {"mode": None}},
            member,
            {**member, "FeatureShuffler": {"mode": None}},
        ],
    }
    stripped, kept = strip_limix_config(config, classification=True)
    assert kept == [1, 3]
    assert stripped["feature_view_strategy"] == "disabled"
    assert len(stripped["pipelines"]) == 2
    for p in stripped["pipelines"]:
        assert p["FeatureShuffler"]["mode"] is None
        assert "PolynomialInteractionGenerator" not in p
        assert p["RebalanceFeatureDistribution"] == {"worker_tags": [None], "svd_tag": None, "original_flag": False}
    assert config["pipelines"][1]["PolynomialInteractionGenerator"]  # the input is untouched
    regression = {"pipelines": [member, member]}
    stripped, kept = strip_limix_config(regression, classification=False)
    assert kept == [0]
    assert stripped["pipelines"][0]["FeatureShuffler"]["mode"] is None
    with pytest.raises(ValueError, match="column order"):
        strip_limix_config(regression, classification=True)


def test_augmentation_off_registry_and_fallback():
    off = augmentation_off("TabPFN-3", _HonestModel, "binary")
    assert off.source == "registry"
    assert off.hyperparameters == {"n_estimators": 1, "inference_config": {"FEATURE_SHIFT_METHOD": None}}
    fallback = augmentation_off("not-registered", _HonestModel, "binary")
    assert fallback.source == "fallback"
    assert fallback.hyperparameters == {}


# --------------------------------------------------------------------------------------------------
# Probes and verdicts on fake models
# --------------------------------------------------------------------------------------------------
def test_model_under_audit_returns_class_columns():
    data = _synthetic("b", "binary", 1, n_samples=300)
    mua = ModelUnderAudit(_HonestModel, num_gpus=0)
    fitted, pred = mua.fit_predict(data, data.X_train, data.y_train, data.X_test, probe="base")
    assert sorted(map(str, pred.columns)) == sorted({str(v) for v in data.y_train})
    assert np.allclose(pred.sum(axis=1), 1.0)
    assert error_of(data, pred) < 0.3
    assert mua.calls[0]["probe"] == "base"
    fitted.cleanup()
    multiclass = _synthetic("m", "multiclass", 1, n_samples=300)
    _, pred = mua.fit_predict(multiclass, multiclass.X_train, multiclass.y_train, multiclass.X_test, probe="base")
    assert set(map(str, pred.columns)) == {str(v) for v in multiclass.y_train}


def test_honest_model_passes_every_dataset():
    results = audit_contamination(_HonestModel, datasets=_five_datasets(), checks=("order",), num_gpus=0, seed=3)
    order = _by_check(results, "order")
    assert set(order) == {"synth-a", "synth-b", "synth-c", "synth-d", "synth-e"}
    for r in order.values():
        assert r.verdict == PASS, r.summary
        assert abs(r.details["d_mean"]) < 0.05
    assert all(r.verdict == PASS for r in _by_check(results, VERDICT_CHECK).values())
    assert leakage_tuple(results) == ()


def test_order_memorizer_fails_only_the_memorized_dataset():
    datasets = _five_datasets()
    _memorize(_OrderMemorizer, datasets[1], as_received=True)
    results = audit_contamination(
        _OrderMemorizer, datasets=datasets, checks=("order",), confirm=True, num_gpus=0, seed=3
    )
    order = _by_check(results, "order")
    assert order["synth-b"].verdict == FAIL, order["synth-b"].summary
    assert order["synth-b"].details["d_min"] > 1.0
    assert order["synth-b"].details["rank_p"] <= 0.1
    assert order["synth-b"].details["n_orders"] == 11  # the confirmation rerun
    for name in ("synth-a", "synth-c", "synth-d", "synth-e"):
        assert order[name].verdict == PASS, order[name].summary
    assert leakage_tuple(results) == ("synth-b",)


def test_order_memorizer_without_confirmation_is_a_warning():
    datasets = _five_datasets()
    _memorize(_OrderMemorizer, datasets[1], as_received=True)
    results = audit_contamination(_OrderMemorizer, datasets=datasets, checks=("order",), num_gpus=0, seed=3)
    assert _by_check(results, "order")["synth-b"].verdict == WARN
    assert "--confirm" in _by_check(results, "order")["synth-b"].summary
    assert leakage_tuple(results) == ()
    assert leakage_tuple(results, WARN) == ("synth-b",)


def test_order_sensitive_honest_model_is_not_an_outlier():
    results = audit_contamination(
        _OrderSensitiveHonest, datasets=_five_datasets(), checks=("order",), num_gpus=0, seed=3
    )
    for r in _by_check(results, "order").values():
        assert r.verdict != FAIL, r.summary
    assert leakage_tuple(results) == ()


def test_context_ignorer_is_caught_by_the_secondary_probes_as_a_warning():
    datasets = _five_datasets()[:1]
    _memorize(_ContextIgnorer, datasets[0], as_received=False)
    results = audit_contamination(
        _ContextIgnorer,
        datasets=datasets,
        checks=("order", "labels", "shuffled"),
        num_gpus=0,
        seed=3,
        no_calibration=True,
    )
    order = _by_check(results, "order")["synth-a"]
    assert order.verdict == PASS, order.summary
    secondary = {r.check: r.verdict for r in results if r.check in ("labels", "shuffled")}
    assert FAIL in secondary.values() or WARN in secondary.values(), secondary
    assert _by_check(results, VERDICT_CHECK)["synth-a"].verdict == WARN


def test_ensemble_probe_warns_when_stripping_helps():
    data = _synthetic("synth-a", "binary", 1)
    group = DatasetSplits("synth-a", "binary", [data], [0])
    _memorize(_OrderMemorizer, data, as_received=True)
    off = probe_order(ModelUnderAudit(_OrderMemorizer, num_gpus=0), group, n_random=1, seed=1)
    released = ModelUnderAudit(_HonestModel, num_gpus=0)  # the "released" model forgets: its error is honest
    result = probe_ensemble(released, group, off, seed=1)
    assert result.verdict == WARN
    assert result.details["d"] > 1.0
    same = probe_ensemble(ModelUnderAudit(_OrderMemorizer, num_gpus=0), group, off, seed=1)
    assert same.verdict == INFO


def test_calibration_skips_below_five_datasets_unless_disabled():
    datasets = _five_datasets()[:2]
    results = audit_contamination(_HonestModel, datasets=datasets, checks=("order",), num_gpus=0, seed=3)
    assert all(r.verdict == SKIP for r in _by_check(results, "order").values())
    results = audit_contamination(
        _HonestModel, datasets=datasets, checks=("order",), num_gpus=0, seed=3, no_calibration=True
    )
    assert all(r.verdict == PASS for r in _by_check(results, "order").values())


def test_calibrate_order_uses_the_problem_type_center_when_it_has_enough_datasets():
    def order_row(name: str, d: float) -> CheckResult:
        details = {
            "d_perm": {REVERSED: d},
            "d_mean": d,
            "d_min": d,
            "d_min_per_split": {"0": d},
            "rank_pub": 1.0,
            "rank_p": 0.5,
            "n_orders": 2,
            "n_splits": 1,
            "metric": "roc_auc",
        }
        return CheckResult("order", name, INFO, "x over 1 split(s)", details)

    # Five binary datasets with a wide honest spread, one regression dataset at 0.3 (pooled: not an outlier).
    rows = [order_row(f"b{i}", d) for i, d in enumerate([0.0, 0.4, -0.3, 0.5, -0.4])] + [order_row("r", 0.3)]
    types = {f"b{i}": "binary" for i in range(5)} | {"r": "regression"}
    calibration = calibrate_order(rows, types)
    assert "binary" in calibration["by_problem_type"]
    assert "regression" not in calibration["by_problem_type"]
    assert rows[-1].verdict == PASS
    assert rows[-1].details["z"] is not None
    assert calibrate_order(rows, types) == calibration  # recalibration is stable and does not stack notes
    assert rows[-1].summary.count("; z") == 1


def test_dataset_verdicts_and_leakage_tuple():
    rows = [
        CheckResult("order", "a", FAIL, "", {}),
        CheckResult("order", "b", PASS, "", {}),
        CheckResult("ensemble", "b", WARN, "", {}),
        CheckResult("order", "c", PASS, "", {}),
        CheckResult("jitter", "c", FAIL, "", {}),
        CheckResult("order", "d", PASS, "", {}),
        CheckResult("probe", "e", FAIL, "", {}),
    ]
    verdicts = {r.dataset: r.verdict for r in dataset_verdicts(rows)}
    assert verdicts == {"a": FAIL, "b": WARN, "c": WARN, "d": PASS, "e": SKIP}
    assert leakage_tuple(rows + dataset_verdicts(rows)) == ("a",)


def test_group_probe_data_groups_splits_by_name():
    a0, a1, b0 = _synthetic("a", "binary", 1, 100), _synthetic("a", "binary", 2, 100), _synthetic("b", "binary", 3, 100)
    groups = group_probe_data([a0, a1, b0])
    assert [(g.name, len(g.splits), g.split_indices) for g in groups] == [("a", 2, [0, 1]), ("b", 1, [0])]


def test_write_json_round_trips_the_verdicts(tmp_path):
    results = audit_contamination(
        _HonestModel, datasets=_five_datasets()[:1], checks=("order",), num_gpus=0, seed=3, no_calibration=True
    )
    meta = {"seed": 3}
    path = tmp_path / "audit.json"
    write_json(results, path, meta)
    payload = json.loads(path.read_text())
    assert payload["meta"] == meta
    assert [r["verdict"] for r in payload["results"]] == [r.verdict for r in results]
    order = next(r for r in payload["results"] if r["check"] == "order")
    assert set(order["details"]["err"]) == {PUBLISHED, REVERSED, "random0", "random1", "random2"}


def test_main_usage_error_exits_2(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["--model", "x", "--splits", "zero"])
    assert exc.value.code == 2


@pytest.mark.models
def test_real_model_smoke(tmp_path):
    """One registered model on a synthetic table, through the CLI, on whatever device is there."""
    import torch

    assert (
        main(
            [
                "--model",
                "TabPFN-3",
                "--datasets",
                "synthetic-binary",
                "--checks",
                "order",
                "--random-orders",
                "1",
                "--max-train-rows",
                "300",
                "--no-calibration",
                "--num-gpus",
                "1" if torch.cuda.is_available() else "0",
                "--json",
                str(tmp_path / "smoke.json"),
            ]
        )
        == 0
    )
    assert (tmp_path / "smoke.json").exists()
