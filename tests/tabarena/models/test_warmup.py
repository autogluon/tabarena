"""Tests for the untimed environment warm-up helpers and per-model-class dispatch.

Nothing heavy is ever imported: torch / library warm-ups are monkeypatched, so these
exercise the dispatch precedence (declared ``warmup`` classmethod > ``AbstractTorchModel``
torch fallback > ``WARMUP_IMPORTS_BY_AG_KEY`` import map > no-op) and the CUDA gating only.
"""

from __future__ import annotations

import sys

from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

import tabarena.models.warmup as wu


class _DeclaredWarmupModel:
    """Model class opting in via the ``warmup`` classmethod convention."""

    calls: list[dict] = []

    @classmethod
    def warmup(cls, **kwargs) -> None:
        cls.calls.append(kwargs)


class _TorchModel(AbstractTorchModel):
    pass


class _MappedModel:
    ag_key = "GBM"


class _PlainModel:
    ag_key = "SOME_UNKNOWN_KEY"


def test_warmup_imports_imports_modules():
    sys.modules.pop("wave", None)
    wu.warmup_imports("wave")
    assert "wave" in sys.modules


def test_declared_warmup_classmethod_wins_and_gets_full_context():
    _DeclaredWarmupModel.calls.clear()
    wu.warmup_model_cls(
        _DeclaredWarmupModel, problem_type="binary", num_cpus=4, num_gpus=1, hyperparameters={"lr": 0.1}
    )
    assert _DeclaredWarmupModel.calls == [
        {"problem_type": "binary", "num_cpus": 4, "num_gpus": 1, "hyperparameters": {"lr": 0.1}}
    ]


def test_torch_model_falls_back_to_torch_warmup(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(wu, "warmup_torch", lambda **kw: calls.append(kw))
    wu.warmup_model_cls(_TorchModel)
    assert calls == [{"cuda": None}]  # num_gpus unknown -> auto-detect


def test_num_gpus_gates_cuda_warmup(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(wu, "warmup_torch", lambda **kw: calls.append(kw))
    wu.warmup_model_cls(_TorchModel, num_gpus=0)
    wu.warmup_model_cls(_TorchModel, num_gpus=1)
    assert calls == [{"cuda": False}, {"cuda": True}]


def test_ag_key_map_import_only_warmup(monkeypatch):
    imported: list[str] = []
    monkeypatch.setattr(wu, "warmup_imports", lambda *names: imported.extend(names))
    wu.warmup_model_cls(_MappedModel)
    assert imported == ["lightgbm"]


def test_ag_key_map_torch_entry_routes_to_torch_warmup(monkeypatch):
    torch_calls: list[dict] = []
    imported: list[str] = []
    monkeypatch.setattr(wu, "warmup_torch", lambda **kw: torch_calls.append(kw))
    monkeypatch.setattr(wu, "warmup_imports", lambda *names: imported.extend(names))

    class _NNTorchModel:
        ag_key = "NN_TORCH"

    wu.warmup_model_cls(_NNTorchModel, num_gpus=0)
    assert torch_calls == [{"cuda": False}]
    assert imported == []


def test_unknown_model_is_a_noop(monkeypatch):
    monkeypatch.setattr(wu, "warmup_torch", lambda **kw: (_ for _ in ()).throw(AssertionError("not expected")))
    monkeypatch.setattr(wu, "warmup_imports", lambda *n: (_ for _ in ()).throw(AssertionError("not expected")))
    wu.warmup_model_cls(_PlainModel)  # no warmup / not torch / unmapped ag_key -> nothing to do
