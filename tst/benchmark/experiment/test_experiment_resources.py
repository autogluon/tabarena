"""Tests for Experiment compute-resource baking + lazy auto-detection."""

from __future__ import annotations

from tabarena.benchmark.experiment.experiment_constructor import Experiment


class _DummyMethod:
    """Placeholder method_cls; never instantiated in these tests."""


def _make_experiment(fit_kwargs: dict | None = None) -> Experiment:
    method_kwargs: dict = {}
    if fit_kwargs is not None:
        method_kwargs["fit_kwargs"] = fit_kwargs
    return Experiment(name="x", method_cls=_DummyMethod, method_kwargs=method_kwargs)


def test_set_resources_writes_fit_kwargs():
    exp = _make_experiment()
    exp.set_resources(num_cpus=8, num_gpus=2, memory_limit=64)
    assert exp.method_kwargs["fit_kwargs"] == {"num_cpus": 8, "num_gpus": 2, "memory_limit": 64}


def test_autodetect_resources_noop_when_concrete():
    mk = {"fit_kwargs": {"num_cpus": 4, "num_gpus": 0, "memory_limit": 16}}
    out = Experiment._autodetect_resources(mk)
    assert out is mk  # unchanged, no copy made


def test_autodetect_resources_no_fit_kwargs_is_noop():
    mk = {}
    assert Experiment._autodetect_resources(mk) is mk


def test_autodetect_resources_fills_none_without_mutating_original():
    mk = {"fit_kwargs": {"num_cpus": None, "num_gpus": 0, "memory_limit": None}}
    out = Experiment._autodetect_resources(mk)

    assert out is not mk  # a copy is returned
    assert isinstance(out["fit_kwargs"]["num_cpus"], int)
    assert out["fit_kwargs"]["num_cpus"] > 0
    assert isinstance(out["fit_kwargs"]["memory_limit"], int)
    assert out["fit_kwargs"]["memory_limit"] > 0
    assert out["fit_kwargs"]["num_gpus"] == 0
    # original left untouched (still requests auto-detection)
    assert mk["fit_kwargs"]["num_cpus"] is None
    assert mk["fit_kwargs"]["memory_limit"] is None


def test_set_resources_none_then_autodetect():
    exp = _make_experiment()
    exp.set_resources(num_cpus=None, num_gpus=0, memory_limit=None)
    out = Experiment._autodetect_resources(exp.method_kwargs)
    assert isinstance(out["fit_kwargs"]["num_cpus"], int)
    assert isinstance(out["fit_kwargs"]["memory_limit"], int)
