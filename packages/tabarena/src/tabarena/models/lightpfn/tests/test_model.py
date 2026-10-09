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
