"""The linear probe's row sampler keeps every class of a multiclass task (fintfm 2026-11-10)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fintfm")

from tabarena.models.fintfm_linprobe.model import _probe_rows


def test_multiclass_keeps_every_class_and_the_rare_ones_whole():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(30_000), np.ones(8_000), np.full(150, 2), np.full(40, 3)]).astype(int)
    rng.shuffle(y)
    rows = _probe_rows(y, 20_000, seed=1)
    assert len(rows) <= 20_000
    assert len(np.unique(rows)) == len(rows)
    kept = np.bincount(y[rows], minlength=4)
    assert kept[2] == 150 and kept[3] == 40
    assert kept[0] > 0 and kept[1] > 0
    assert len(rows) == 20_000


def test_multiclass_within_budget_keeps_every_row():
    y = np.array([0, 1, 2, 2, 1, 0, 2])
    assert np.array_equal(_probe_rows(y, 100, seed=0), np.arange(7))


def test_binary_draw_is_unchanged():
    y = (np.random.default_rng(3).random(50_000) < 0.05).astype(int)
    rows = _probe_rows(y, 20_000, seed=7)
    # The binary rule: every positive kept, the rest of the budget from negatives.
    assert (y[rows] == 1).sum() == (y == 1).sum()
    assert len(rows) == 20_000
