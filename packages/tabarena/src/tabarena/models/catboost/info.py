from __future__ import annotations

from autogluon.tabular.models import CatBoostModel

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.catboost.hpo import gen_catboost

catboost_descriptor = ModelDescriptor(
    display_name="CatBoost",
    compute="cpu",
    is_bag=True,
    reference_url="https://arxiv.org/abs/1706.09516",
    date_introduced="2017-06",
)

# Superseded by the rerun below; kept so the hosted artifacts stay loadable.
catboost_method_metadata = catboost_descriptor.method_metadata(
    method="CatBoost",
    suite="tabarena-2025-06-12",
    ag_key="CAT",
    config_default="CatBoost_c1_BAG_L1",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    date="2025-06-12",
    # FIXME: technically GBDTs are not verified
)

# Pareto-front rerun with the improved train/infer time measurement.
catboost_new_method_metadata = catboost_descriptor.method_metadata(
    method="CatBoost",
    suite="tabarena-2026-07-13",
    ag_key="CAT",
    config_default="CatBoost_c1_default_BAG_L1",
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    date="2026-07-15",
)


catboost_info = ModelInfo(
    model_cls=CatBoostModel,
    search_space=gen_catboost,
    method_metadata=catboost_new_method_metadata,
)
