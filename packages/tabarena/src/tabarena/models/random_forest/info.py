from __future__ import annotations

from autogluon.tabular.models import RFModel

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.random_forest.hpo import gen_randomforest

random_forest_descriptor = ModelDescriptor(
    display_name="RandomForest",
    compute="cpu",
    is_bag=True,
    reference_url="https://link.springer.com/article/10.1023/A:1010933404324",
)

random_forest_method_metadata = random_forest_descriptor.method_metadata(
    method="RandomForest",
    date="2025-06-12",
    ag_key="RF",
    config_default="RandomForest_c1_BAG_L1",
    can_hpo=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    artifact_name="tabarena-2025-06-12",
    cache_type="s3",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    name_suffix=None,
    # FIXME: technically baselines are not verified
    verified=True,
)


random_forest_info = ModelInfo(
    model_cls=RFModel,
    search_space=gen_randomforest,
    method_metadata=random_forest_method_metadata,
)
