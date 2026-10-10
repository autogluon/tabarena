from __future__ import annotations

import pytest


def test_default_config_caches_the_context_only_for_the_refit(tmp_path):
    from tabarena.models.lightpfn.hpo import gen_lightpfn
    from tabarena.models.lightpfn.model import LightPFNModel

    (config,) = gen_lightpfn.manual_configs
    fold_model = LightPFNModel(path=str(tmp_path), name="fold", hyperparameters=config)
    refit_model = fold_model.convert_to_refit_full_template()

    assert fold_model.get_params()["hyperparameters"]["cache_context"] is False
    assert refit_model.get_params()["hyperparameters"]["cache_context"] is True


def test_uncached_predictions_match_the_cache(tmp_path):
    pytest.importorskip("lightpfn")
    import numpy as np
    import pandas as pd

    from tabarena.models.lightpfn.model import LightPFNModel

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=[f"f{i}" for i in range(4)])
    X["cat"] = pd.Categorical(rng.choice(["a", "b", "c"], size=300))
    y = pd.Series((X["f0"] + (X["cat"] == "a") > 0.5).astype(int))
    predictions = {}
    for cache_context in (True, False):
        model = LightPFNModel(
            path=str(tmp_path / str(cache_context)),
            name="m",
            problem_type="binary",
            eval_metric="log_loss",
            hyperparameters={"n_estimators": 2, "cache_context": cache_context},
        )
        model.fit(X=X.iloc[:250], y=y.iloc[:250], num_gpus=0)
        assert bool(model.model.members_) == cache_context
        assert (model.model._uncached_data is None) == cache_context
        predictions[cache_context] = LightPFNModel.load(model.save()).predict_proba(X.iloc[250:])

    np.testing.assert_allclose(predictions[True], predictions[False], atol=1e-6)


@pytest.mark.parametrize(("n_classes", "many_class_threshold"), [(15, 10), (5, 3)])
def test_more_classes_than_the_head_are_output_coded(tmp_path, n_classes, many_class_threshold):
    pytest.importorskip("lightpfn")
    pytest.importorskip("tabpfn_extensions.many_class")
    import numpy as np
    import pandas as pd

    from tabarena.models.lightpfn.model import LightPFNModel

    rng = np.random.default_rng(0)
    y = pd.Series(np.repeat(np.arange(n_classes), 20))
    X = pd.DataFrame(rng.normal(size=(len(y), 4)), columns=[f"f{i}" for i in range(4)])
    X["f0"] += y * 3.0
    X["cat"] = pd.Categorical(np.where(y % 2 == 0, "even", "odd"))
    train = rng.permutation(len(y))[: int(0.8 * len(y))]
    test = np.setdiff1d(np.arange(len(y)), train)
    model = LightPFNModel(
        path=str(tmp_path),
        name="m",
        problem_type="multiclass",
        eval_metric="log_loss",
        hyperparameters={"n_estimators": 1, "ag_args_fit": {"many_class_threshold": many_class_threshold}},
    )
    model.fit(X=X.iloc[train], y=y.iloc[train], num_gpus=0)
    assert model._use_many_class and model.get_device() == "cpu"

    proba = LightPFNModel.load(model.save()).predict_proba(X.iloc[test])
    assert proba.shape == (len(test), n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert (proba.argmax(axis=1) == y.iloc[test].to_numpy()).mean() > 0.8
