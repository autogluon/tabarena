"""Tests for BeyondArena eval config dataclasses (no heavy imports / no real runs)."""

from __future__ import annotations

from tabarena.evaluation.beyond_arena_eval import BenchmarkRun


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
