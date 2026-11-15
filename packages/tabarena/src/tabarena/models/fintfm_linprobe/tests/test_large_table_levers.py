# ruff: noqa: F811
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fintfm")
pytest.importorskip("torch")

from tabarena.models.fintfm_linprobe import model as lp_model
from tabarena.models.fintfm_linprobe.model import (
    LEAF_MIN_SAMPLE,
    PARTITION_LIBRARY_MAX_DEPTH,
    TRAIN_SAMPLE_SIZE,
    FinTFMLinearProbeModel,
)
from tabarena.models.fintfm_linprobe.tests.test_crossfit import (  # noqa: F401 (fixtures)
    N_TRAIN,
    PARTITION,
    Spy,
    checkpoint,
    clean_env,
    data,
    make,
)

PROBLEMS = ["binary", "multiclass", "regression"]


@pytest.fixture(autouse=True)
def clean_lever_env(monkeypatch):
    monkeypatch.delenv("FINTFM_PARTITION_SCALE_N", raising=False)
    monkeypatch.delenv("FINTFM_PROBE_BUDGET", raising=False)


def depth_for(tmp_path, checkpoint, n, problem_type="binary", partition=None, **hp):
    """The ``max_depth`` the scaling gives a table of ``n`` rows, without fitting."""
    part = dict(PARTITION) if partition is None else partition
    hyper = {"partition_scale_with_n": True, f"partition_{problem_type}": part, **hp}
    m = make(problem_type, tmp_path, checkpoint, **hyper)
    m._n_fit_rows = n
    return m._resolve_entry(m._get_model_params())["partition"]["max_depth"], m


@pytest.mark.parametrize("problem_type", PROBLEMS)
def test_off_by_default_and_byte_identical(problem_type, tmp_path, checkpoint):
    X, y, X_test = data(problem_type)
    part = {f"partition_{problem_type}": dict(PARTITION)}
    base = make(problem_type, tmp_path / "a", checkpoint, **part)
    explicit = make(
        problem_type,
        tmp_path / "b",
        checkpoint,
        partition_scale_with_n=False,
        probe_budget=TRAIN_SAMPLE_SIZE,
        **part,
    )
    params = base._get_model_params()
    assert params["partition_scale_with_n"] is False
    assert params["probe_budget"] == TRAIN_SAMPLE_SIZE
    # Even with k and cap given, nothing changes while the switch is off.
    third = make(problem_type, tmp_path / "c", checkpoint, partition_scale_k=0.01, partition_depth_cap=3, **part)
    for m in (base, explicit, third):
        m.fit(X=X, y=y)
        assert m.model.max_depth == PARTITION["max_depth"]
        assert m.partition_depth_chosen_ is None
    predict = "predict" if problem_type == "regression" else "predict_proba"
    ref = getattr(base, predict)(X_test)
    assert np.array_equal(ref, getattr(explicit, predict)(X_test))
    assert np.array_equal(ref, getattr(third, predict)(X_test))


def test_depth_grows_with_n_as_specified(tmp_path, checkpoint):
    # max_context 60, k 4: one target leaf is 240 rows, so depth = ceil(log2(n / 240)).
    got = {}
    for n in (100, 240, 480, 960, 1920, 100_000):
        got[n], _ = depth_for(tmp_path, checkpoint, n, partition={"n_bags": 1, "max_depth": 1})
    assert got[100] == got[240] == 1  # below the target: the base depth, never lower
    assert got[480] == 1
    assert got[960] == 2
    assert got[1920] == 3
    assert got[100_000] == PARTITION_LIBRARY_MAX_DEPTH  # default cap 5 is clipped to the library's 3
    values = list(got.values())
    assert values == sorted(values)


def test_cap_k_and_base_are_respected(tmp_path, checkpoint):
    part = {"n_bags": 1, "max_depth": 1}
    d, _ = depth_for(tmp_path, checkpoint, 100_000, partition=part, partition_depth_cap=2)
    assert d == 2
    d, _ = depth_for(tmp_path, checkpoint, 100_000, partition=part, partition_depth_cap=1)
    assert d == 1
    # A larger k wants fewer leaves; k=1 wants more.
    d_k8, _ = depth_for(tmp_path, checkpoint, 1920, partition=part, partition_scale_k=8)
    d_k1, _ = depth_for(tmp_path, checkpoint, 1920, partition=part, partition_scale_k=1)
    assert d_k8 == 2
    assert d_k1 == 3
    # The partition's own max_depth is the floor.
    d, _ = depth_for(tmp_path, checkpoint, 100, partition={"n_bags": 1, "max_depth": 3, "min_samples_leaf": 0.2})
    assert d == 3
    # With bags, one tree sees bag_frac * n rows.
    d_bag, m = depth_for(tmp_path, checkpoint, 960, partition={"n_bags": 2, "bag_frac": 0.5, "max_depth": 1})
    assert m.partition_depth_chosen_["n_effective"] == 480
    assert d_bag == 1


def test_scaling_keeps_other_partition_arguments_and_records_the_choice(tmp_path, checkpoint):
    part = {"n_bags": 2, "max_depth": 1, "min_samples_leaf": 0.1, "min_leaf_rows": 30}
    _, m = depth_for(tmp_path, checkpoint, 100_000, partition=part)
    entry_part = m._resolve_entry(m._get_model_params())["partition"]
    assert {k: v for k, v in entry_part.items() if k != "max_depth"} == {
        k: v for k, v in part.items() if k != "max_depth"
    }
    assert part["max_depth"] == 1  # the caller's dict is not mutated
    rec = m.partition_depth_chosen_
    assert rec["chosen"] == 3
    assert rec["cap_requested"] == 5
    assert rec["cap_effective"] == 3


@pytest.mark.parametrize("problem_type", PROBLEMS)
def test_scaled_depth_reaches_the_partition_classifier(problem_type, tmp_path, checkpoint, caplog):
    X, y, X_test = data(problem_type)
    part = {"n_bags": 1, "max_depth": 1, "min_samples_leaf": 0.1, "min_leaf_rows": 30, "min_class_rows": 2}
    # max_context 60, k 1 on 240 rows: ceil(log2(4)) = 2.
    m = make(
        problem_type,
        tmp_path,
        checkpoint,
        partition_scale_with_n=True,
        partition_scale_k=1,
        **{f"partition_{problem_type}": part},
    )
    with caplog.at_level("WARNING"):
        m.fit(X=X, y=y)
    assert m.model.max_depth == 2
    assert m.partition_depth_chosen_["chosen"] == 2
    assert any("partition depth 2 for n=240" in r.getMessage() for r in caplog.records)
    out = m.predict(X_test) if problem_type == "regression" else m.predict_proba(X_test)
    assert np.isfinite(out).all()


def test_scaling_without_a_partition_is_a_no_op(tmp_path, checkpoint):
    X, y, _ = data("binary")
    m = make("binary", tmp_path, checkpoint, partition_scale_with_n=True, partition_binary=None)
    m.fit(X=X, y=y)
    assert m.partition_depth_chosen_ is None
    assert not hasattr(m.model, "trees_")


def test_probe_budget_is_honoured_in_the_plain_probe(tmp_path, checkpoint, monkeypatch):
    X, y, _ = data("binary")
    seen = []
    original = lp_model._probe_rows
    monkeypatch.setattr(
        lp_model, "_probe_rows", lambda yy, budget, seed: seen.append(budget) or original(yy, budget, seed)
    )
    make("binary", tmp_path / "a", checkpoint, partition_binary=None, probe_budget=100).fit(X=X, y=y)
    assert seen == [100]
    seen.clear()
    make("binary", tmp_path / "all", checkpoint, partition_binary=None, probe_budget="all").fit(X=X, y=y)
    assert seen == [N_TRAIN]
    seen.clear()
    make("binary", tmp_path / "d", checkpoint, partition_binary=None).fit(X=X, y=y)
    assert seen == [TRAIN_SAMPLE_SIZE]


@pytest.mark.parametrize("budget", [50_000, "all", 150])
def test_probe_budget_is_honoured_per_leaf(budget, tmp_path, checkpoint, monkeypatch):
    X, y, _ = data("multiclass")
    seen = []
    original = FinTFMLinearProbeModel._fit_probe

    def spy(self, est, Xp, y_arr, members, b, seed, fit_rows=None):
        seen.append((len(members), b))
        return original(self, est, Xp, y_arr, members, b, seed, fit_rows=fit_rows)

    monkeypatch.setattr(FinTFMLinearProbeModel, "_fit_probe", spy)
    make("multiclass", tmp_path, checkpoint, partition_multiclass=dict(PARTITION), probe_budget=budget).fit(X=X, y=y)
    total = budget if budget != "all" else N_TRAIN
    assert seen[0] == (N_TRAIN, total)  # the global probe gets the whole budget
    for n_members, b in seen[1:]:
        assert b == max(LEAF_MIN_SAMPLE, round(total * n_members / N_TRAIN))


@pytest.mark.parametrize("budget", [100, "all"])
def test_probe_budget_is_honoured_by_the_ridge_readout(budget, tmp_path, checkpoint, monkeypatch):
    X, y, _ = data("regression")
    seen = []
    original = FinTFMLinearProbeModel._fit_ridge

    def spy(self, clf, Xp, y_arr, members, b, seed, **kw):
        seen.append(b)
        return original(self, clf, Xp, y_arr, members, b, seed, **kw)

    monkeypatch.setattr(FinTFMLinearProbeModel, "_fit_ridge", spy)
    make("regression", tmp_path, checkpoint, partition_regression=None, probe_budget=budget).fit(X=X, y=y)
    assert seen == [100 if budget == 100 else N_TRAIN]


@pytest.mark.parametrize("bad", [0, -5, 2.5, True, "most", None])
def test_bad_probe_budget_raises(bad, tmp_path, checkpoint):
    m = make("binary", tmp_path, checkpoint, probe_budget=bad)
    with pytest.raises(ValueError, match="probe_budget"):
        m._resolve_entry(m._get_model_params())


@pytest.mark.parametrize(
    ("hp", "match"),
    [
        ({"partition_scale_with_n": "yes"}, "partition_scale_with_n"),
        ({"partition_scale_k": 0}, "partition_scale_k"),
        ({"partition_depth_cap": 0}, "partition_depth_cap"),
        ({"partition_depth_cap": 2.5}, "partition_depth_cap"),
    ],
)
def test_bad_scale_levers_raise(hp, match, tmp_path, checkpoint):
    m = make("binary", tmp_path, checkpoint, **hp)
    with pytest.raises(ValueError, match=match):
        m._resolve_entry(m._get_model_params())


@pytest.mark.parametrize("problem_type", PROBLEMS)
def test_crossfit_still_covers_every_context_row_with_both_levers_on(problem_type, tmp_path, checkpoint, monkeypatch):
    X, y, _ = data(problem_type)
    part = {"n_bags": 2, "max_depth": 1, "min_samples_leaf": 0.1, "min_leaf_rows": 30, "min_class_rows": 2}
    m = make(
        problem_type,
        tmp_path,
        checkpoint,
        partition_scale_with_n=True,
        partition_scale_k=1,
        probe_budget="all",
        probe_crossfit_folds=5,
        **{f"partition_{problem_type}": part},
    )
    spy = Spy(monkeypatch)
    m.fit(X=X, y=y)
    assert m.model.max_depth == 2
    assert spy.calls
    assert spy.rows_seen_in_own_context() == 0
    assert spy.restricted
    assert m._crossfit_stats["restricted"] >= 5
    assert m._train_matrix is None
    assert m._crossfit_stats["context_rows"] > 0


def test_environment_variables_set_the_defaults(monkeypatch, tmp_path, checkpoint):
    monkeypatch.setenv("FINTFM_PARTITION_SCALE_N", "on")
    monkeypatch.setenv("FINTFM_PROBE_BUDGET", "50000")
    params = make("binary", tmp_path, checkpoint)._get_model_params()
    assert params["partition_scale_with_n"] is True
    assert params["probe_budget"] == 50_000
    assert params["partition_scale_k"] == 4.0
    monkeypatch.setenv("FINTFM_PARTITION_SCALE_N", "2.5")
    monkeypatch.setenv("FINTFM_PROBE_BUDGET", "ALL")
    params = make("binary", tmp_path, checkpoint)._get_model_params()
    assert params["partition_scale_with_n"] is True
    assert params["partition_scale_k"] == 2.5
    assert params["probe_budget"] == "all"
    monkeypatch.setenv("FINTFM_PARTITION_SCALE_N", "off")
    assert make("binary", tmp_path, checkpoint)._get_model_params()["partition_scale_with_n"] is False


def test_hyperparameters_beat_the_environment(monkeypatch, tmp_path, checkpoint):
    monkeypatch.setenv("FINTFM_PARTITION_SCALE_N", "on")
    monkeypatch.setenv("FINTFM_PROBE_BUDGET", "50000")
    params = make("binary", tmp_path, checkpoint, partition_scale_with_n=False, probe_budget=777)._get_model_params()
    assert params["partition_scale_with_n"] is False
    assert params["probe_budget"] == 777


def test_environment_scaling_changes_the_fitted_depth(monkeypatch, tmp_path, checkpoint):
    monkeypatch.setenv("FINTFM_PARTITION_SCALE_N", "1")  # k = 1: 240 rows / (1 * 60) = 4 -> depth 2
    X, y, _ = data("binary")
    part = {"max_depth": 1, "min_samples_leaf": 0.1, "min_leaf_rows": 30}
    m = make("binary", tmp_path, checkpoint, partition_binary=part)
    m.fit(X=X, y=y)
    assert m.model.max_depth == 2


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("FINTFM_PARTITION_SCALE_N", "maybe"),
        ("FINTFM_PARTITION_SCALE_N", "-1"),
        ("FINTFM_PROBE_BUDGET", "lots"),
        ("FINTFM_PROBE_BUDGET", "0"),
    ],
)
def test_malformed_environment_values_raise(monkeypatch, tmp_path, checkpoint, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        make("binary", tmp_path, checkpoint)
