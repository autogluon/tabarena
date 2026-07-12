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

# Superseded runs of TabArena's earlier custom TabM implementation (suite ``tabarena-2025-06-12``),
# CPU and GPU variants. Kept (unregistered) so their hosted artifacts stay loadable for comparison;
# the current implementation is described by ``tabm_new_method_metadata`` below.
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


# Re-implementation of TabM on top of the official ``tabm`` PyPI package — the TabM used by
# TabArena going forward (registered below and referenced by the tabarena arena collection).
# ``ag_key``/``config_default`` match the processed artifacts: the raw ``TA-TabM_*`` configs are
# renamed to the ``TabM`` prefix during processing.
tabm_new_method_metadata = tabm_descriptor.method_metadata(
    method="TabM",
    suite="tabarena-2026-06-26",
    ag_key="TA-TABM",
    config_default="TabM_c1_default_BAG_L1",
    date="2026-06-26",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)


tabm_info = ModelInfo(
    model_cls=TabMModel,
    search_space=gen_tabm,
    method_metadata=tabm_new_method_metadata,
    pip_extra=("torch", "tabm>=0.0.3", "rtdl_num_embeddings>=0.0.12"),
)
