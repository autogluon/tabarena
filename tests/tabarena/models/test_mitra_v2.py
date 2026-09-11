"""Focused tests for Mitra-v2's fine-tuning OOM fallback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("autogluon.tabular.models.mitra.sklearn_interface")

from tabarena.models.mitra_v2._internal import estimators


class _FakeModel:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()


class _FakeEstimator(estimators.MitraV2Mixin):
    def __init__(self, *, support: int = 16_384, query: int = 1_024):
        self.cfg = SimpleNamespace(
            seed=0,
            hyperparams={
                "max_samples_support": support,
                "max_samples_query": query,
            },
        )
        self.n_estimators = 1
        self.hf_model = "unused"
        self.device = "cpu"
        self.verbose = False
        self.trainers = []
        self.train_time = 0

    def _create_config(self, *_args, **_kwargs):
        return self.cfg, _FakeModel


def _train(estimator: _FakeEstimator):
    return estimator._train_ensemble(None, None, None, None, task=None, dim_output=1)


@pytest.mark.parametrize(
    "oom",
    [
        RuntimeError("CUDA out of memory"),
        pytest.param(
            pytest.importorskip("torch").cuda.OutOfMemoryError("CUDA out of memory"),
            id="typed-oom",
        ),
    ],
    ids=["runtime-oom", None],
)
def test_train_ensemble_retries_cuda_oom(monkeypatch, oom):
    attempts = 0

    class _FakeTrainer:
        def __init__(self, *_args, **_kwargs):
            pass

        def train(self, *_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise oom

    monkeypatch.setattr(estimators, "MitraV2Trainer", _FakeTrainer)
    monkeypatch.setattr(estimators.torch.cuda, "empty_cache", lambda: None)

    estimator = _FakeEstimator()
    assert _train(estimator) is estimator
    assert attempts == 2
    assert estimator.cfg.hyperparams["max_samples_support"] == 8_192
    assert len(estimator.trainers) == 1


def test_train_ensemble_reraises_non_oom_runtime_error(monkeypatch):
    class _FakeTrainer:
        def __init__(self, *_args, **_kwargs):
            pass

        def train(self, *_args, **_kwargs):
            raise RuntimeError("shape mismatch")

    monkeypatch.setattr(estimators, "MitraV2Trainer", _FakeTrainer)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        _train(_FakeEstimator())


def test_train_ensemble_stops_after_caps_are_exhausted(monkeypatch):
    attempts = 0

    class _FakeTrainer:
        def __init__(self, *_args, **_kwargs):
            pass

        def train(self, *_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(estimators, "MitraV2Trainer", _FakeTrainer)
    monkeypatch.setattr(estimators.torch.cuda, "empty_cache", lambda: None)

    with pytest.raises(RuntimeError, match="Failed to train Mitra-v2"):
        _train(_FakeEstimator(support=2, query=2))

    assert attempts == 2
