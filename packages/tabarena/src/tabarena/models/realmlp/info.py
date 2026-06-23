from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.realmlp.hpo import gen_realmlp
from tabarena.models.realmlp.model import RealMLPModel

realmlp_descriptor = ModelDescriptor(
    display_name="RealMLP",
    compute="gpu",
    is_bag=True,
    reference_url="https://arxiv.org/abs/2407.04491",
)

realmlp_method_metadata = realmlp_descriptor.method_metadata(
    method="RealMLP_GPU",
    date="2025-09-03",
    ag_key="TA-REALMLP",
    model_key="REALMLP_GPU",
    config_default="RealMLP_GPU_c1_BAG_L1",
    can_hpo=True,
    verified=True,
    artifact_name="tabarena-2025-09-03",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    name_suffix=None,
)


# CPU variant — same model class, same search space, different compute target.
realmlp_cpu_method_metadata = realmlp_descriptor.method_metadata(
    method="RealMLP",
    display_name="RealMLP (CPU)",
    compute="cpu",
    date="2025-06-12",
    ag_key="REALMLP",
    config_default="RealMLP_c1_BAG_L1",
    can_hpo=True,
    verified=True,
    artifact_name="tabarena-2025-06-12",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    name_suffix=None,
)


realmlp_info = ModelInfo(
    model_cls=RealMLPModel,
    search_space=gen_realmlp,
    method_metadata=realmlp_method_metadata,
    pip_extra=("pytabkit>=1.5.0,<2.0",),
)


realmlp_cpu_info = ModelInfo(
    model_cls=RealMLPModel,
    search_space=gen_realmlp,
    method_metadata=realmlp_cpu_method_metadata,
    pip_extra=("pytabkit>=1.5.0,<2.0",),
)
