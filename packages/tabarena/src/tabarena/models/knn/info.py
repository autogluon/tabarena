from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.knn.hpo import gen_knn
from tabarena.models.knn.model import KNNNewModel

knn_method_metadata = MethodMetadata.config(
    method="KNeighbors",
    display_name="KNN",
    compute="cpu",
    date="2025-10-20",
    ag_key="KNN",
    config_default="KNeighbors_c1_BAG_L1",
    can_hpo=True,
    is_bag=False,
    reference_url="https://scikit-learn.org/stable/modules/neighbors.html",
    artifact_name="tabarena-2025-10-20",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    # FIXME: technically kNN is not verified
)


knn_info = ModelInfo(
    model_cls=KNNNewModel,
    search_space=gen_knn,
    method_metadata=knn_method_metadata,
)
