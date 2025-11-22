from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import REGRESSION
from autogluon.core.models import AbstractModel
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


logger = logging.getLogger(__name__)


# FIXME: model is for some reason super slow for 200 features and 50k samples (363616)
class SAPRPTOSSModel(AbstractModel):
    """ConTextTab Model: https://github.com/SAP-samples/sap-rpt-1-oss."""

    ag_key = "SAP-RPT-OSS"
    ag_name = "SAP-RPT-OSS"
    ag_priority = 65
    seed_name = "random_state"

    # TODO: Figure out if num_cpus could be used somewhere
    # TODO: Pre-download the used LM checkpoint used for the embeddings
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        available_num_gpus = ResourceManager.get_gpu_count_torch(cuda_only=True)
        if num_gpus > available_num_gpus:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but only {available_num_gpus} "
                "CUDA GPUs are available. Please activate CUDA or switch to CPU usage.",
            )

        # FIXME: remove workaround and integrate into codebase 
        from sap_rpt_oss.model import attention
        attention.TorchSelfAttention = TorchSelfAttention

        from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor

        if self.problem_type in ["binary", "multiclass"]:
            model_cls = SAP_RPT_OSS_Classifier
        elif self.problem_type in ["regression"]:
            model_cls = SAP_RPT_OSS_Regressor
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")

        hps = self._get_model_params()
        random_state = hps.pop(self.seed_name, 42)

        self.model = model_cls(
            **hps,
        )
        # TODO: make code support this like a normal sklearn model
        self.model.seed = random_state

        X = self.preprocess(X)
        self.model = self.model.fit(
            X=X,
            y=y,
        )

    def _set_default_params(self):
        # Default values from the current version of the code base
        default_params = {
            "checkpoint": "2025-11-04_sap-rpt-one-oss.pt",
            "max_context_size": 8192,
            "bagging": 8,
            "test_chunk_size": 1000,  # TODO, optimize based on dataset/VRAM?
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)

        if self.problem_type == REGRESSION:
            # FIXME: make model code flatten by default as normal for sklearn models
            return self.model.predict(X).flatten()

        y_pred_proba = self.model.predict_proba(X)
        return self._convert_proba_to_unified_form(y_pred_proba)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)

        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(
        self, is_gpu_available: bool = False
    ) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 1 if is_gpu_available else 0,
        }

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Set fold_fitting_strategy to sequential_local,
        as parallel folding crashes if model weights aren't pre-downloaded.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            "fold_fitting_strategy": "sequential_local",
            "refit_folds": True,
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    @classmethod
    def _class_tags(cls) -> dict:
        # TODO: support memory estimate!
        return {"can_estimate_memory_usage_static": False}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    # TODO: Configure the AutoGluon preprocessing to pass the raw data format to the model
    #  (without preprocessing dates or texts) and not remove it from the features.
    # def _get_default_auxiliary_params(self) -> dict:
    #     default_auxiliary_params = super()._get_default_auxiliary_params()
    #     extra_auxiliary_params = dict(
    #         get_features_kwargs=dict(
    #             valid_special_types=[S_TEXT],
    #         )
    #     )
    #     default_auxiliary_params.update(extra_auxiliary_params)
    #     return default_auxiliary_params


def pre_download_model():
    from huggingface_hub import hf_hub_download

    # Hardcoded to the checkpoint we use in TabArena.
    hf_hub_download(
        repo_id="SAP/sap-rpt-1-oss", filename="2025-11-04_sap-rpt-one-oss.pt"
    )


# FIXME: model is not pickable -> _SDPBackend -> make pickable with code below
#   Comes from sap_rpt_oss.model.attention.TorchSelfAttention
#         if config.attention_implementation == 'efficient' and torch.cuda.is_available():
#             self.backend = SDPBackend.EFFICIENT_ATTENTION
#         elif config.attention_implementation == 'math' or not torch.cuda.is_available():
#             self.backend = SDPBackend.MATH
class TorchSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0, (
            f"{config.hidden_size=} must be divisible by {config.num_attention_heads=}"
        )
        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = config.attention_probs_dropout_prob

        if config.attention_implementation == "efficient" and torch.cuda.is_available():
            self.backend = "EF"
        elif config.attention_implementation == "math" or not torch.cuda.is_available():
            self.backend = "SDP"
        else:
            raise ValueError(
                f"Unknown attention implementation {config.attention_implementation}"
            )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = (*x.size()[:-1], self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        """hidden_states: shape (batch_size, seq_len, hidden_size)
        attention_mask: shape (batch_size, seq_len, seq_len) or (batch_size, 1, seq_len, seq_len).
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        backend = (
            SDPBackend.EFFICIENT_ATTENTION if self.backend == "EF" else SDPBackend.MATH
        )

        with sdpa_kernel(backend):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        context_layer = attn_output.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = (*context_layer.size()[:-2], self.config.hidden_size)
        return context_layer.view(new_context_layer_shape)
