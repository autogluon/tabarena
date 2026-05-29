from __future__ import annotations

from autogluon.tabular.models import NNFastAiTabularModel

from tabarena.models._model_info import ModelInfo
from tabarena.models.fastai.hpo import gen_fastai
from tabarena.models._method_metadata import MethodMetadata


fastai_method_metadata = MethodMetadata(
    method="NeuralNetFastAI",
    method_type="config",
    display_name="FastaiMLP",
    compute="cpu",
    date="2025-06-12",
    ag_key="FASTAI",
    config_default="NeuralNetFastAI_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    artifact_name="tabarena-2025-06-12",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    name_suffix=None,
    verified=True,
    reference_url="https://arxiv.org/abs/2003.06505",
)


fastai_info = ModelInfo(
    model_cls=NNFastAiTabularModel,
    search_space=gen_fastai,
    method_metadata=fastai_method_metadata,
)
