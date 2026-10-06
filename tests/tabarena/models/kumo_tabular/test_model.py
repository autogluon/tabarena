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


def test_default_config_caches_the_context_only_for_the_refit(tmp_path):
    from tabarena.models.kumo_tabular.hpo import gen_kumo_tabular
    from tabarena.models.kumo_tabular.model import KumoTabularModel

    (config,) = gen_kumo_tabular.manual_configs
    fold_model = KumoTabularModel(path=str(tmp_path), name="fold", hyperparameters=config)
    refit_model = fold_model.convert_to_refit_full_template()

    assert fold_model.get_params()["hyperparameters"]["cache_context"] is False
    assert refit_model.get_params()["hyperparameters"]["cache_context"] is True


@pytest.mark.models
@pytest.mark.parametrize("max_context_size", [None, 150])
def test_uncached_predictions_match_the_cache(tmp_path, max_context_size):
    pytest.importorskip("sdm")
    import numpy as np
    import pandas as pd

    from tabarena.models.kumo_tabular.model import KumoTabularSmallModel

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X["f0"] > 0).astype(int))
    predictions = {}
    for cache_context in (True, False):
        model = KumoTabularSmallModel(
            path=str(tmp_path / str(cache_context)),
            name="m",
            problem_type="binary",
            eval_metric="log_loss",
            hyperparameters={"num_estimators": 2, "max_context_size": max_context_size, "cache_context": cache_context},
        )
        model.fit(X=X.iloc[:250], y=y.iloc[:250], num_gpus=0)
        assert (model.model.cache is not None) == cache_context
        assert (model.model.context is None) == cache_context
        model.save()
        predictions[cache_context] = KumoTabularSmallModel.load(model.path).predict_proba(X.iloc[250:])

    np.testing.assert_allclose(predictions[True], predictions[False], atol=1e-5)


@pytest.mark.models
def test_resave_over_a_memory_mapped_cache(tmp_path):
    pytest.importorskip("sdm")
    import numpy as np
    import pandas as pd

    from tabarena.models.kumo_tabular.model import KumoTabularSmallModel

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X["f0"] > 0).astype(int))
    model = KumoTabularSmallModel(
        path=str(tmp_path),
        name="m",
        problem_type="binary",
        eval_metric="log_loss",
        hyperparameters={"num_estimators": 2},
    )
    model.fit(X=X, y=y, num_gpus=0)
    expected = model.predict_proba(X.iloc[:20])
    loaded = KumoTabularSmallModel.load(model.save())
    # On the CPU the loaded cache is memory-mapped from the file this save replaces.
    loaded.save()

    np.testing.assert_allclose(loaded.predict_proba(X.iloc[:20]), expected, atol=1e-6)


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


def test_unsigned_columns_become_signed_and_pickle():
    import pickle

    import numpy as np
    import pandas as pd

    from tabarena.models.kumo_tabular.model import to_signed_integers

    X = pd.DataFrame(
        {
            "u8": np.array([1, 2], dtype=np.uint8),
            "u32": np.array([1, 2**32 - 1], dtype=np.uint32),
            "u64": np.array([1, 2**64 - 1], dtype=np.uint64),
            "f": [0.5, 1.5],
        }
    )
    out = to_signed_integers(X)

    assert out.dtypes.astype(str).tolist() == ["uint8", "int64", "float64", "float64"]
    assert out["u32"].tolist() == [1, 2**32 - 1]
    for column in ("u8", "u32"):
        tensor = torch.from_numpy(out[column].to_numpy())
        assert torch.equal(pickle.loads(pickle.dumps(tensor)), tensor)
