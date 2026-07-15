from __future__ import annotations

from autogluon.tabular.models import LGBModel

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.lightgbm.hpo import gen_lightgbm

lightgbm_descriptor = ModelDescriptor(
    display_name="LightGBM",
    compute="cpu",
    is_bag=True,
    reference_url="https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
    date_introduced="2016-09",
)

lightgbm_method_metadata = lightgbm_descriptor.method_metadata(
    method="LightGBM",
    suite="tabarena-2025-06-12",
    ag_key="GBM",
    config_default="LightGBM_c1_BAG_L1",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    date="2025-06-12",
    # FIXME: technically GBDTs are not verified
)


lightgbm_info = ModelInfo(
    model_cls=LGBModel,
    search_space=gen_lightgbm,
    method_metadata=lightgbm_method_metadata,
)
