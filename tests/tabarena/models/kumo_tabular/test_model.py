from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tabarena.models.kumo_tabular.model import context_subsample_index


def test_no_index_when_the_context_fits():
    assert context_subsample_index(n_rows=100, num_estimators=4, max_context_size=100, seed=0) is None
    assert context_subsample_index(n_rows=100, num_estimators=4, max_context_size=None, seed=0) is None


def test_members_use_every_row_about_equally_often():
    index = context_subsample_index(n_rows=250, num_estimators=4, max_context_size=100, seed=0)

    assert index.shape == (4, 100)
    counts = torch.bincount(index.flatten(), minlength=250)
    assert counts.min() >= 1
    assert counts.max() <= 2


def test_seed_makes_the_draw_reproducible():
    first = context_subsample_index(n_rows=250, num_estimators=4, max_context_size=100, seed=3)
    assert torch.equal(first, context_subsample_index(n_rows=250, num_estimators=4, max_context_size=100, seed=3))
    assert not torch.equal(first, context_subsample_index(n_rows=250, num_estimators=4, max_context_size=100, seed=4))
