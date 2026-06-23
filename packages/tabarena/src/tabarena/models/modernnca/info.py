from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.modernnca.hpo import gen_modernnca
from tabarena.models.modernnca.model import ModernNCAModel

modernnca_method_metadata = MethodMetadata.config(
    method="ModernNCA",
    display_name="ModernNCA (CPU)",
    compute="cpu",
    date="2025-06-12",
    ag_key="MNCA",
    config_default="ModernNCA_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    artifact_name="tabarena-2025-06-12",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    reference_url="https://arxiv.org/abs/2407.03257",
)


modernnca_gpu_method_metadata = MethodMetadata.config(
    method="ModernNCA_GPU",
    display_name="ModernNCA",
    compute="gpu",
    date="2025-06-12",
    ag_key="MNCA",
    config_default="ModernNCA_GPU_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    artifact_name="tabarena-2025-06-12",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    name_suffix="_GPU",
    reference_url="https://arxiv.org/abs/2407.03257",
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
