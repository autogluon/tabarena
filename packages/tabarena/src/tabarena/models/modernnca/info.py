from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.modernnca.hpo import gen_modernnca
from tabarena.models.modernnca.model import ModernNCAModel

modernnca_method_metadata = MethodMetadata.config(
    method="ModernNCA",
    artifact_name="tabarena-2025-06-12",
    ag_key="MNCA",
    config_default="ModernNCA_c1_BAG_L1",
    compute="cpu",
    is_bag=True,
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    date="2025-06-12",
    reference_url="https://arxiv.org/abs/2407.03257",
    display_name="ModernNCA (CPU)",
)


modernnca_gpu_method_metadata = MethodMetadata.config(
    method="ModernNCA_GPU",
    artifact_name="tabarena-2025-06-12",
    ag_key="MNCA",
    config_default="ModernNCA_GPU_c1_BAG_L1",
    compute="gpu",
    is_bag=True,
    name_suffix="_GPU",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    date="2025-06-12",
    reference_url="https://arxiv.org/abs/2407.03257",
    display_name="ModernNCA",
)


modernnca_info = ModelInfo(
    model_cls=ModernNCAModel,
    search_space=gen_modernnca,
    method_metadata=modernnca_method_metadata,
    pip_extra=("category_encoders",),
)


modernnca_gpu_info = ModelInfo(
    model_cls=ModernNCAModel,
    search_space=gen_modernnca,
    method_metadata=modernnca_gpu_method_metadata,
    pip_extra=("category_encoders",),
)
