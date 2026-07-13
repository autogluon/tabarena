from __future__ import annotations

from autogluon.tabular.models import TabularNeuralNetTorchModel

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.nn_torch.hpo import gen_nn_torch

nn_torch_method_metadata = MethodMetadata.config(
    method="NeuralNetTorch",
    suite="tabarena-2025-06-12",
    ag_key="NN_TORCH",
    config_default="NeuralNetTorch_c1_BAG_L1",
    compute="cpu",
    is_bag=True,
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    date="2025-06-12",
    date_introduced="2019-12",
    reference_url="https://arxiv.org/abs/2003.06505",
    display_name="TorchMLP",
)


nn_torch_info = ModelInfo(
    model_cls=TabularNeuralNetTorchModel,
    search_space=gen_nn_torch,
    method_metadata=nn_torch_method_metadata,
)
