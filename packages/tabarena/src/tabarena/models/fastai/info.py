from __future__ import annotations

from autogluon.tabular.models import NNFastAiTabularModel

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.fastai.hpo import gen_fastai

fastai_method_metadata = MethodMetadata.config(
    method="NeuralNetFastAI",
    display_name="FastaiMLP",
    compute="cpu",
    date="2025-06-12",
    ag_key="FASTAI",
    config_default="NeuralNetFastAI_c1_BAG_L1",
    is_bag=True,
    artifact_name="tabarena-2025-06-12",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    reference_url="https://arxiv.org/abs/2003.06505",
)


fastai_info = ModelInfo(
    model_cls=NNFastAiTabularModel,
    search_space=gen_fastai,
    method_metadata=fastai_method_metadata,
)
