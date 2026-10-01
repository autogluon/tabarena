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


def test_out_of_memory_splits_the_query_rows_in_order(monkeypatch):
    import numpy as np

    from tabarena.models.kumo_tabular import model

    monkeypatch.setattr(model, "_MIN_QUERY_PASS_ROWS", 2)
    passes = []

    def forward(x_query, device):
        passes.append(len(x_query))
        if len(x_query) > 3:
            raise torch.OutOfMemoryError("fake")
        return ["a"], x_query[:, None] * 10

    wrapper = model.KumoTabularModel.__new__(model.KumoTabularModel)
    wrapper._forward = forward
    labels, values = wrapper._predict_values(np.arange(10), device=None)

    assert labels == ["a"]
    assert values[:, 0].tolist() == [i * 10 for i in range(10)]
    assert passes == [10, 5, 2, 3, 5, 2, 3]


def test_out_of_memory_below_the_smallest_pass_is_raised(monkeypatch):
    import numpy as np

    from tabarena.models.kumo_tabular import model

    def forward(x_query, device):
        raise torch.OutOfMemoryError("fake")

    wrapper = model.KumoTabularModel.__new__(model.KumoTabularModel)
    wrapper._forward = forward
    with pytest.raises(torch.OutOfMemoryError):
        wrapper._predict_values(np.arange(model._MIN_QUERY_PASS_ROWS), device=None)
