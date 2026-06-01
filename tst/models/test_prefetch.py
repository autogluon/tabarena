"""Tests for the standardized foundation-model weight prefetch dispatcher.

The actual downloads are never invoked: model resolution is monkeypatched so each fake model
carries a ``prefetch_weights`` callable (as real models declare via ``ModelInfo.prefetch_weights``).
These exercise the dispatch / dedup / skip / error-handling logic only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import tabarena.models.prefetch as pf


def _fake_info(method: str, prefetch=None) -> SimpleNamespace:
    return SimpleNamespace(method_metadata=SimpleNamespace(method=method), prefetch_weights=prefetch)


def _patch_registry(monkeypatch, infos: dict[str, SimpleNamespace]) -> None:
    """Make ``get_model_info_from_name`` resolve from ``infos`` (KeyError -> ValueError)."""

    def fake_get(name: str):
        if name not in infos:
            raise ValueError(f"unknown {name}")
        return infos[name]

    monkeypatch.setattr("tabarena.models.utils.get_model_info_from_name", fake_get)


def test_declared_prefetcher_called_once(monkeypatch):
    calls: list[str] = []
    _patch_registry(monkeypatch, {"M": _fake_info("M_GPU", lambda: calls.append("M"))})
    pf.prefetch_weights(["M"])
    assert calls == ["M"]


def test_shared_prefetcher_deduplicated(monkeypatch):
    calls: list[str] = []
    shared = lambda: calls.append("tabpfn")  # noqa: E731 — two variants share one prefetcher
    _patch_registry(
        monkeypatch,
        {"TabPFN-3": _fake_info("TabPFN-3", shared), "TabPFN-2.6": _fake_info("TabPFN-v2.6", shared)},
    )
    pf.prefetch_weights(["TabPFN-3", "TabPFN-2.6"])
    assert calls == ["tabpfn"]  # warmed once despite two variants


def test_non_foundation_model_skipped(monkeypatch):
    _patch_registry(monkeypatch, {"RandomForest": _fake_info("RandomForest", None)})
    pf.prefetch_weights(["RandomForest"])  # prefetch_weights is None -> skipped, no error


def test_unknown_model_skipped_unless_raising(monkeypatch):
    _patch_registry(monkeypatch, {})
    pf.prefetch_weights(["Nonexistent"])  # ValueError swallowed
    with pytest.raises(ValueError, match="unknown"):
        pf.prefetch_weights(["Nonexistent"], raise_on_error=True)


def test_prefetch_error_logged_then_raised_when_requested(monkeypatch):
    def boom():
        raise RuntimeError("download failed")

    _patch_registry(monkeypatch, {"M": _fake_info("M_GPU", boom)})
    pf.prefetch_weights(["M"])  # swallowed
    with pytest.raises(RuntimeError, match="download failed"):
        pf.prefetch_weights(["M"], raise_on_error=True)


def test_missing_optional_dependency_always_skipped(monkeypatch):
    def missing():
        raise ImportError("no module named 'tabpfn'")

    _patch_registry(monkeypatch, {"M": _fake_info("M_GPU", missing)})
    pf.prefetch_weights(["M"])  # ImportError swallowed
    pf.prefetch_weights(["M"], raise_on_error=True)  # even when raising
