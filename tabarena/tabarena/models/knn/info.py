from __future__ import annotations

from tabarena.models.knn.model import KNNNewModel
from tabarena.models._model_info import ModelInfo
from tabarena.models.knn.hpo import gen_knn
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


knn_method_metadata = MethodMetadata(
    method="KNeighbors",
    method_type="config",
    display_name="KNN",
    compute="cpu",
    date="2025-10-20",
    ag_key="KNN",
    config_default="KNeighbors_c1_BAG_L1",
    can_hpo=True,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    reference_url="https://scikit-learn.org/stable/modules/neighbors.html",
    artifact_name="tabarena-2025-10-20",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_results=True,
    name_suffix=None,
    # FIXME: technically kNN is not verified
    verified=True,
)


knn_info = ModelInfo(
    model_cls=KNNNewModel,
    search_space=gen_knn,
    method_metadata=knn_method_metadata,
)
