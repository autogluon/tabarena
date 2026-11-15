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
