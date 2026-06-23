from __future__ import annotations

from autogluon.tabular.models import MitraModel

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.mitra.hpo import gen_mitra


# NOTE: Prefetchers normally live in the model's ``model.py``. Mitra is the exception: it is
# supported out of the box via AutoGluon's external ``MitraModel`` (re-exported below), so there is
# no tabarena-side ``model.py`` to host this — it lives in ``info.py`` alongside the registration.
def prefetch_weights() -> None:
    """Pre-download the Mitra classifier + regressor checkpoints from Hugging Face."""
    from huggingface_hub import hf_hub_download

    for repo_id in ("autogluon/mitra-classifier", "autogluon/mitra-regressor"):
        hf_hub_download(repo_id=repo_id, filename="config.json")
        hf_hub_download(repo_id=repo_id, filename="model.safetensors")


mitra_method_metadata = MethodMetadata.config(
    method="Mitra_GPU",
    display_name="Mitra",
    compute="gpu",
    date="2025-09-03",
    ag_key="MITRA",
    model_key="MITRA_GPU",
    config_default="Mitra_GPU_c1_BAG_L1",
    can_hpo=False,
    is_bag=True,
    verified=True,
    reference_url="https://arxiv.org/abs/2510.21204",
    artifact_name="tabarena-2025-09-03",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    has_raw=True,
    has_processed=True,
    has_results=True,
    name_suffix=None,
)


mitra_info = ModelInfo(
    model_cls=MitraModel,
    search_space=gen_mitra,
    method_metadata=mitra_method_metadata,
    prefetch_weights=prefetch_weights,
)
