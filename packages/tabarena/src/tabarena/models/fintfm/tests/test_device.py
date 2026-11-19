"""``FINTFM_DEVICE`` forces the FinTFM adapter's device; unset, the adapter chooses."""

from __future__ import annotations

import pytest

from tabarena.models.fintfm.model import FinTFMModel


@pytest.mark.parametrize(
    ("raw", "expected"), [(None, None), ("", None), (" cpu ", "cpu"), ("mps", "mps"), ("cuda", "cuda")]
)
def test_forced_device_reads_the_variable(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: str | None
) -> None:
    """Unset or blank leaves the choice automatic; a named device is returned stripped."""
    monkeypatch.delenv("FINTFM_DEVICE", raising=False)
    if raw is not None:
        monkeypatch.setenv("FINTFM_DEVICE", raw)
    assert FinTFMModel._forced_device() == expected


def test_forced_device_refuses_an_unknown_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misspelled device raises rather than silently falling back to automatic choice."""
    monkeypatch.setenv("FINTFM_DEVICE", "gpu")
    with pytest.raises(ValueError, match="FINTFM_DEVICE"):
        FinTFMModel._forced_device()


@pytest.mark.parametrize(
    ("num_gpus", "cuda", "mps", "expected"),
    [
        (1, True, False, "cuda"),
        (0, True, False, "cpu"),
        (1, False, True, "cpu"),
        (0, False, True, "cpu"),
        (0, False, False, "cpu"),
    ],
)
def test_choose_device_never_picks_mps_on_its_own(
    monkeypatch: pytest.MonkeyPatch, num_gpus: int, cuda: bool, mps: bool, expected: str
) -> None:
    """CUDA only when a GPU was granted and CUDA exists; otherwise CPU, even on Apple Silicon."""
    import torch

    monkeypatch.delenv("FINTFM_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert FinTFMModel._choose_device(num_gpus) == expected


def test_choose_device_follows_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """``FINTFM_DEVICE=mps`` is the only way to fit on MPS."""
    monkeypatch.setenv("FINTFM_DEVICE", "mps")
    assert FinTFMModel._choose_device(0) == "mps"
