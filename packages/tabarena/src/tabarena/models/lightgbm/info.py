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
)

lightgbm_method_metadata = lightgbm_descriptor.method_metadata(
    method="LightGBM",
    date="2025-06-12",
    ag_key="GBM",
    config_default="LightGBM_c1_BAG_L1",
    can_hpo=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    artifact_name="tabarena-2025-06-12",
    cache_type="s3",
    s3_bucket="tabarena",
    s3_prefix="cache",
    cache_kwargs={"upload_as_public": True},
    name_suffix=None,
    # FIXME: technically GBDTs are not verified
    verified=True,
)


lightgbm_info = ModelInfo(
    model_cls=LGBModel,
    search_space=gen_lightgbm,
    method_metadata=lightgbm_method_metadata,
)
