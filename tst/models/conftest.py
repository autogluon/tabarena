"""Shared fixtures for the model wrapper tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    """Run every model test with the CWD in its own tmp dir.

    AutoGluon's ``FitHelper`` (used by these tests via ``verify_model``) hardcodes
    CWD-relative paths (``./datasets/`` for toy datasets and the ``AutogluonOutput_*``
    predictor dirs) with no parameter to redirect them — without this, running the
    model tests litters the repo root.
    """
    monkeypatch.chdir(tmp_path)
