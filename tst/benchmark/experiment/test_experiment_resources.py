"""Tests for Experiment lazy compute-resource auto-detection."""

from __future__ import annotations

from tabarena.benchmark.experiment.experiment_constructor import Experiment


def test_apply_resources_noop_when_concrete():
    mk = {"fit_kwargs": {"num_cpus": 4, "num_gpus": 0, "memory_limit": 16}}
    out = Experiment._apply_resources(mk)
    assert out is mk  # unchanged, no copy made


def test_apply_resources_no_fit_kwargs_is_noop():
    mk = {}
    assert Experiment._apply_resources(mk) is mk


def test_apply_resources_fills_none_without_mutating_original():
    mk = {"fit_kwargs": {"num_cpus": None, "num_gpus": 0, "memory_limit": None}}
    out = Experiment._apply_resources(mk)

    assert out is not mk  # a copy is returned
    assert isinstance(out["fit_kwargs"]["num_cpus"], int)
    assert out["fit_kwargs"]["num_cpus"] > 0
    assert isinstance(out["fit_kwargs"]["memory_limit"], int)
    assert out["fit_kwargs"]["memory_limit"] > 0
    assert out["fit_kwargs"]["num_gpus"] == 0
    # original left untouched (still requests auto-detection)
    assert mk["fit_kwargs"]["num_cpus"] is None
    assert mk["fit_kwargs"]["memory_limit"] is None
