from __future__ import annotations

from autogluon.tabular.models import XTModel

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.extra_trees.hpo import gen_extratrees

extra_trees_method_metadata = MethodMetadata(
    method="ExtraTrees",
    method_type="config",
    display_name="ExtraTrees",
    compute="cpu",
    date="2025-06-12",
    ag_key="XT",
    config_default="ExtraTrees_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    artifact_name="tabarena-2025-06-12",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    name_suffix=None,
    # FIXME: technically baselines are not verified
    verified=True,
    reference_url="https://link.springer.com/article/10.1007/s10994-006-6226-1",
)


extra_trees_info = ModelInfo(
    model_cls=XTModel,
    search_space=gen_extratrees,
    method_metadata=extra_trees_method_metadata,
)
