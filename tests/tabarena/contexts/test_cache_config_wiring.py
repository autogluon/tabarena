from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.caching import CacheConfig
from tabarena.contexts import TabArenaContext


@pytest.fixture(autouse=True)
def _reset_cache_state():
    """Reset the TabArena cache-root holder and the OpenML root cache around each test."""
    import openml

    from tabarena.loaders import set_tabarena_cache_root

    saved_openml_root = openml.config._root_cache_directory
    try:
        yield
    finally:
        set_tabarena_cache_root(None)
        openml.config.set_root_cache_directory(str(saved_openml_root))


def _make_ctx(**kwargs) -> TabArenaContext:
    """A light context (no method-metadata load) over a 2-dataset / 2-fold native collection."""
    df = pd.DataFrame(
        {
            "tid": [0, 1],
            "dataset": ["small_ds", "big_ds"],
            "name": ["small_ds", "big_ds"],
            "problem_type": ["binary", "regression"],
            "n_folds": [2, 2],
            "n_repeats": [1, 1],
            "n_features": [10, 10],
            "n_classes": [2, 0],
            "NumberOfInstances": [150, 75_000],
            "n_samples_train_per_fold": [100, 50_000],
            "n_samples_test_per_fold": [50, 25_000],
            "target_feature": ["t", "t"],
        },
    )
    return TabArenaContext(methods=[], task_metadata=TaskMetadataCollection.from_legacy_df(df), **kwargs)


class _StubExperiment:
    """Stand-in for an Experiment: build_jobs only reads ``.name`` / ``.model_constraints``."""

    model_constraints = None

    def __init__(self, name: str = "exp"):
        self.name = name


def _stub_runner(record: dict):
    """A fake ExperimentBatchRunner that records its ``expname`` and runs nothing."""

    class _FakeRunner:
        def __init__(self, *, expname, task_metadata, **kwargs):
            record["expname"] = expname

        def run_jobs(self, jobs):
            return []

    return _FakeRunner


def test_apply_on_init_configures_constructing_process(tmp_path):
    from tabarena.loaders import get_tabarena_cache_root

    _make_ctx(cache_config=CacheConfig(tabarena=tmp_path / "tab"))
    assert get_tabarena_cache_root() == tmp_path / "tab"


def test_run_jobs_reapplies_cache_config(monkeypatch, tmp_path):
    import tabarena.benchmark.experiment as exp_mod
    from tabarena.loaders import get_tabarena_cache_root, set_tabarena_cache_root

    monkeypatch.setattr(exp_mod, "ExperimentBatchRunner", _stub_runner({}))
    ctx = _make_ctx(cache_config=CacheConfig(tabarena=tmp_path / "tab"))
    jobs = ctx.build_jobs([_StubExperiment()], subset="lite")

    set_tabarena_cache_root(None)  # clear what __init__ applied; run_jobs must re-apply it
    ctx.run_jobs(jobs, expname=str(tmp_path / "exp"), register=False)
    assert get_tabarena_cache_root() == tmp_path / "tab"


def test_run_jobs_does_not_reapply_when_disabled(monkeypatch, tmp_path):
    import tabarena.benchmark.experiment as exp_mod
    from tabarena.loaders import get_tabarena_cache_root, set_tabarena_cache_root

    monkeypatch.setattr(exp_mod, "ExperimentBatchRunner", _stub_runner({}))
    ctx = _make_ctx(cache_config=CacheConfig(tabarena=tmp_path / "tab", apply_on_run=False))
    jobs = ctx.build_jobs([_StubExperiment()], subset="lite")

    set_tabarena_cache_root(None)
    ctx.run_jobs(jobs, expname=str(tmp_path / "exp"), register=False)
    assert get_tabarena_cache_root() != tmp_path / "tab"  # holder was not re-set


def test_run_jobs_omitted_expname_defaults_from_results(monkeypatch, tmp_path):
    import tabarena.benchmark.experiment as exp_mod

    record: dict = {}
    monkeypatch.setattr(exp_mod, "ExperimentBatchRunner", _stub_runner(record))
    ctx = _make_ctx(cache_config=CacheConfig(results=tmp_path / "res"))
    jobs = ctx.build_jobs([_StubExperiment()], subset="lite")

    ctx.run_jobs(jobs, register=False)  # expname omitted entirely -> falls back to results
    assert record["expname"] == str(tmp_path / "res")


def test_run_jobs_explicit_none_is_throwaway_even_with_results(monkeypatch, tmp_path):
    import tabarena.benchmark.experiment as exp_mod

    record: dict = {}
    monkeypatch.setattr(exp_mod, "ExperimentBatchRunner", _stub_runner(record))
    ctx = _make_ctx(cache_config=CacheConfig(results=tmp_path / "res"))
    jobs = ctx.build_jobs([_StubExperiment()], subset="lite")

    ctx.run_jobs(jobs, expname=None, register=False)  # explicit None -> throwaway temp dir, NOT results
    assert record["expname"] != str(tmp_path / "res")


def test_run_jobs_requires_expname_without_results(tmp_path):
    ctx = _make_ctx()  # no cache_config -> no results default
    jobs = ctx.build_jobs([_StubExperiment()], subset="lite")
    with pytest.raises(TypeError, match="expname"):
        ctx.run_jobs(jobs, register=False)  # omitted and no default -> required


def test_explicit_expname_overrides_results_default(monkeypatch, tmp_path):
    import tabarena.benchmark.experiment as exp_mod

    record: dict = {}
    monkeypatch.setattr(exp_mod, "ExperimentBatchRunner", _stub_runner(record))
    ctx = _make_ctx(cache_config=CacheConfig(results=tmp_path / "res"))
    jobs = ctx.build_jobs([_StubExperiment()], subset="lite")

    ctx.run_jobs(jobs, expname=str(tmp_path / "explicit"), register=False)
    assert record["expname"] == str(tmp_path / "explicit")


def test_scope_openml_init_leaves_ambient_openml_untouched(tmp_path):
    import openml

    from tabarena.loaders import get_tabarena_cache_root

    openml.config.set_root_cache_directory(str(tmp_path / "ambient"))
    _make_ctx(cache_config=CacheConfig(openml=tmp_path / "x", tabarena=tmp_path / "tab", scope_openml=True))
    # OpenML stays at the ambient location; the TabArena cache (needed by compare) is applied.
    assert Path(openml.config._root_cache_directory) == tmp_path / "ambient"
    assert get_tabarena_cache_root() == tmp_path / "tab"


def test_scope_openml_run_jobs_uses_x_then_restores_openml(monkeypatch, tmp_path):
    import openml

    import tabarena.benchmark.experiment as exp_mod

    record: dict = {}

    class _OpenmlCapturingRunner:
        def __init__(self, *, expname, task_metadata, **kwargs):
            # Constructed inside the cache scope -> records the OpenML root active during the run.
            record["openml_in_scope"] = str(openml.config._root_cache_directory)

        def run_jobs(self, jobs):
            return []

    monkeypatch.setattr(exp_mod, "ExperimentBatchRunner", _OpenmlCapturingRunner)
    openml.config.set_root_cache_directory(str(tmp_path / "ambient"))
    ctx = _make_ctx(cache_config=CacheConfig(openml=tmp_path / "x", tabarena=tmp_path / "tab", scope_openml=True))
    jobs = ctx.build_jobs([_StubExperiment()], subset="lite")

    ctx.run_jobs(jobs, expname=str(tmp_path / "exp"), register=False)
    assert record["openml_in_scope"] == str(tmp_path / "x")  # X active during the run
    assert Path(openml.config._root_cache_directory) == tmp_path / "ambient"  # restored after
