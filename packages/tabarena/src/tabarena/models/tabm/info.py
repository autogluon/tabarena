from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabm.hpo import gen_tabm
from tabarena.models.tabm.model import TabMModel

tabm_descriptor = ModelDescriptor(
    display_name="TabM",
    compute="gpu",
    is_bag=True,
    reference_url="https://arxiv.org/abs/2410.24210",
)

# CPU variant — same model class, same search space, different compute target.
tabm_method_metadata = tabm_descriptor.method_metadata(
    method="TabM",
    suite="tabarena-2025-06-12",
    ag_key="TABM",
    config_default="TabM_c1_BAG_L1",
    compute="cpu",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    date="2025-06-12",
    display_name="TabM (CPU)",
)


tabm_gpu_method_metadata = tabm_descriptor.method_metadata(
    method="TabM_GPU",
    suite="tabarena-2025-06-12",
    ag_key="TABM",
    config_default="TabM_GPU_c1_BAG_L1",
    name_suffix="_GPU",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    date="2025-06-12",
)


# DRAFT — re-implementation of TabM on top of the official ``tabm`` PyPI package
# (the entries above describe TabArena's earlier custom implementation, suite
# ``tabarena-2025-06-12``). This is a single GPU-capable run (the model also runs on
# CPU) that has not been benchmarked yet, hence ``verified=False``. Before an official
# run, update ``suite``/``date`` to the run date and configure remote storage
# (``cache_type``/``cache_kwargs``), then wrap it in a ``ModelInfo`` to register it.
tabm_v2_method_metadata = tabm_descriptor.method_metadata(
    method="TabM",
    suite="tabarena-2026-06-26",  # placeholder: set to the actual benchmark run date
    ag_key="TABM",
    config_default="TabM_c1_BAG_L1",
    compute="gpu",
    date="2026-06-26",
    verified=False,
)


tabm_info = ModelInfo(
    model_cls=TabMModel,
    search_space=gen_tabm,
    method_metadata=tabm_method_metadata,
    pip_extra=("torch",),
)


tabm_gpu_info = ModelInfo(
    model_cls=TabMModel,
    search_space=gen_tabm,
    method_metadata=tabm_gpu_method_metadata,
    pip_extra=("torch", "tabm>=0.0.3", "rtdl_num_embeddings>=0.0.12"),
)
