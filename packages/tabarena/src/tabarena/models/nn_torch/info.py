from __future__ import annotations

from autogluon.tabular.models import TabularNeuralNetTorchModel

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.nn_torch.hpo import gen_nn_torch

nn_torch_method_metadata = MethodMetadata(
    method="NeuralNetTorch",
    method_type="config",
    display_name="TorchMLP",
    compute="cpu",
    date="2025-06-12",
    ag_key="NN_TORCH",
    config_default="NeuralNetTorch_c1_BAG_L1",
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


nn_torch_info = ModelInfo(
    model_cls=TabularNeuralNetTorchModel,
    search_space=gen_nn_torch,
    method_metadata=nn_torch_method_metadata,
)
