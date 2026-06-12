"""Tests for BeyondArena eval config dataclasses (no heavy imports / no real runs)."""

from __future__ import annotations

from tabarena.evaluation.beyond_arena_eval import BenchmarkRun, BeyondArenaEvalConfig


def _run(only_load_cache):
    return BenchmarkRun(
        benchmark_name="b",
        output_dir="out/b",
        models=["TabPFN-3", "CatBoost", "LightGBM"],
        only_load_cache=only_load_cache,
    )


def test_loads_from_cache_bool_false_regenerates_all():
    run = _run(only_load_cache=False)
    assert [run.loads_from_cache(m) for m in run.models] == [False, False, False]


def test_loads_from_cache_bool_true_loads_all():
    run = _run(only_load_cache=True)
    assert [run.loads_from_cache(m) for m in run.models] == [True, True, True]


def test_loads_from_cache_list_is_per_model():
    # Only the named models load from cache; the rest are re-generated.
    run = _run(["CatBoost", "LightGBM"])
    assert run.loads_from_cache("TabPFN-3") is False
    assert run.loads_from_cache("CatBoost") is True
    assert run.loads_from_cache("LightGBM") is True


def test_loads_from_cache_empty_list_regenerates_all():
    run = _run([])
    assert [run.loads_from_cache(m) for m in run.models] == [False, False, False]


def _config(**kwargs) -> BeyondArenaEvalConfig:
    return BeyondArenaEvalConfig(runs=[_run(False)], figure_output_dir="out/figs", **kwargs)


def test_aux_metric_disabled_by_default():
    # The aux metric slows down raw post-processing, so it is opt-in.
    config = _config()
    assert config.compute_aux_metric is False
    assert config.effective_aux_metric_map() is None


def test_aux_metric_opt_in_publishes_map():
    config = _config(compute_aux_metric=True)
    assert config.effective_aux_metric_map() == {
        "binary": "balanced_accuracy",
        "multiclass": "balanced_accuracy",
        "regression": "r2",
    }


def test_aux_metric_custom_map_only_used_when_enabled():
    custom = {"regression": "rmse"}
    assert _config(aux_metric_map=custom).effective_aux_metric_map() is None
    assert _config(compute_aux_metric=True, aux_metric_map=custom).effective_aux_metric_map() == custom
