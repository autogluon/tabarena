# Copyright (C) 2026 Xiaomi Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sklearn loaders for the isolated TabLDM classifier MoE variants."""

from __future__ import annotations

from pathlib import Path

import re
import torch

from .classifier import TabLDMBaseClassifier
from tabarena.models.tabldm._vendor._model.attnres_light_rmsnorm_moe import TabLDMMoE
from tabarena.models.tabldm._vendor._model.attnres_light_rmsnorm_moe import TabLDMSparseMoE
from tabarena.models.tabldm._vendor._model.embedding_dual_stream import ColEmbeddingDualStream



def _moe_load_mismatch(model_keys, missing, unexpected):
    """Return real load mismatches after :meth:`drop_dense_ffn`.

    The frozen dense FFN copy (``linear1``/``linear2``) is dropped from
    MoE layers, so checkpoints saved before that removal still carry those
    tensors. They are dead weights, so they are tolerated as the only
    unexpected keys; any other missing/unexpected key is a real mismatch.
    """
    bad_unexpected = set()
    for key in unexpected:
        match = re.match(r"^(.*)\.(linear1|linear2)\.(weight|bias)$", key)
        if match and f"{match.group(1)}.moe_ffn.router.weight" in model_keys:
            continue
        bad_unexpected.add(key)
    return set(missing), bad_unexpected


class _MoEClassifierLoader(TabLDMBaseClassifier):
    _model_cls = TabLDMMoE

    def _load_model(self) -> None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import LocalEntryNotFoundError

        repo_id = "occams/Xiaomi-TabLDM"
        filename = self.checkpoint_version
        if self.model_path is None:
            try:
                model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True))
            except LocalEntryNotFoundError:
                if not self.allow_auto_download:
                    raise ValueError(
                        f"Checkpoint '{filename}' not cached and automatic download is disabled."
                    )
                model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename))
            checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
        else:
            model_path_ = Path(self.model_path) if isinstance(self.model_path, str) else self.model_path
            if not model_path_.exists():
                if not self.allow_auto_download:
                    raise ValueError(f"Checkpoint not found at '{model_path_}'.")
                model_path_.parent.mkdir(parents=True, exist_ok=True)
                cache_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path_.parent)
                Path(cache_path).rename(model_path_)
            checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)

        if "config" not in checkpoint or "state_dict" not in checkpoint:
            raise ValueError("The checkpoint must contain 'config' and 'state_dict'.")

        self.model_path_ = model_path_
        config = dict(checkpoint["config"])
        self.model_config_ = config
        dual_stream_cfg = checkpoint.get("dual_stream_config", {})
        global_dilation = config.get("global_dilation", dual_stream_cfg.get("global_dilation", "adaptive"))
        global_max_span = config.get("global_max_span", dual_stream_cfg.get("global_max_span", 32))

        parent_config = {
            key: value
            for key, value in config.items()
            if key not in ("global_dilation", "global_max_span")
            and not key.startswith("icl_moe_")
        }
        for legacy_name, native_name in {
            "icl_moe_num_experts": "moe_num_experts",
            "icl_moe_top_k": "moe_top_k",
            "icl_moe_num_shared_experts": "moe_num_shared_experts",
            "icl_moe_layers": "moe_layers",
            "icl_moe_router_z_loss_coef": "moe_router_z_loss_coef",
            "icl_moe_load_balance_loss_coef": "moe_load_balance_loss_coef",
            "icl_moe_router_jitter": "moe_router_jitter",
            "icl_moe_router_weight_mode": "moe_router_weight_mode",
            "icl_moe_expert_init_noise": "moe_expert_init_noise",
            "icl_moe_init_from_dense": "moe_init_from_dense",
        }.items():
            if legacy_name in config:
                parent_config[native_name] = config[legacy_name]
        self.model_ = self._model_cls(**parent_config)
        self.model_.col_embedder = ColEmbeddingDualStream(
            embed_dim=config.get("embed_dim", 128),
            num_blocks=config.get("col_num_blocks", 3),
            nhead=config.get("col_nhead", 8),
            dim_feedforward=config.get("embed_dim", 128) * config.get("ff_factor", 2),
            num_inds=config.get("col_num_inds", 128),
            dropout=config.get("dropout", 0.0),
            activation=config.get("activation", "gelu"),
            norm_first=config.get("norm_first", True),
            bias_free_ln=True,
            affine=config.get("col_affine", False),
            feature_group=config.get("col_feature_group", "same"),
            feature_group_size=config.get("col_feature_group_size", 3),
            global_dilation=global_dilation,
            global_max_span=global_max_span,
            target_aware=config.get("col_target_aware", True),
            max_classes=config.get("max_classes", 0),
            reserve_cls_tokens=config.get("row_num_cls", 4),
            ssmax=config.get("col_ssmax", False),
            zero_init=config.get("zero_init", False),
            mixed_radix_ensemble=True,
            recompute=False,
        )

        self.model_.drop_dense_ffn()
        state_dict = checkpoint["state_dict"]
        try:
            missing, unexpected = self.model_.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Inference requires an exact current AttnRes/RMSNorm/MoE checkpoint; "
                "legacy checkpoints must be converted during training first."
            ) from exc
        bad_missing, bad_unexpected = _moe_load_mismatch(
            set(self.model_.state_dict().keys()), missing, unexpected
        )
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "Inference requires an exact current AttnRes/RMSNorm/MoE checkpoint; "
                "legacy checkpoints must be converted during training first."
            )
        self.model_.eval()


class TabLDMClassifier(_MoEClassifierLoader):
    _model_cls = TabLDMSparseMoE


__all__ = [
    "TabLDMClassifier",
]
