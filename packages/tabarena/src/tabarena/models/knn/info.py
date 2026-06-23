from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.knn.hpo import gen_knn
from tabarena.models.knn.model import KNNNewModel

knn_method_metadata = MethodMetadata.config(
    method="KNeighbors",
    artifact_name="tabarena-2025-10-20",
    ag_key="KNN",
    config_default="KNeighbors_c1_BAG_L1",
    compute="cpu",
    is_bag=False,
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    date="2025-10-20",
    reference_url="https://scikit-learn.org/stable/modules/neighbors.html",
    display_name="KNN",
    # FIXME: technically kNN is not verified
)


knn_info = ModelInfo(
    model_cls=KNNNewModel,
    search_space=gen_knn,
    method_metadata=knn_method_metadata,
)
