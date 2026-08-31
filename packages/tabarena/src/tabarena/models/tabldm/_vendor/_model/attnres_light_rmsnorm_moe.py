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
"""AttnRes/RMSNorm classifier backbone with an isolated ICL FFN MoE."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .attnres_light_rmsnorm import (
    AttnResEncoderLightRMSNorm,
    AttnResTransformerLayerLightRMSNorm,
    ICLearningAttnResLightRMSNorm,
    TabLDMAttnResLightRMSNorm,
)
from .embedding_dual_stream import ColEmbeddingDualStream
from .moe import SparseMoEFeedForward, collect_moe_aux_loss, collect_moe_aux_stats


class AttnResTransformerLayerLightRMSNormMoE(AttnResTransformerLayerLightRMSNorm):
    def __init__(
        self,
        *args,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        moe_num_shared_experts: int = 1,
        moe_router_z_loss_coef: float = 1e-3,
        moe_load_balance_loss_coef: float = 1e-2,
        moe_router_jitter: float = 0.0,
        moe_router_weight_mode: str = "normalized",
        moe_expert_init_noise: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.moe_ffn = None
        if moe_num_experts > 0:
            self.moe_ffn = SparseMoEFeedForward(
                d_model=self.linear1.in_features,
                hidden_dim=self.linear1.out_features,
                dropout=self.dropout.p,
                activation=self.activation,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                num_shared_experts=moe_num_shared_experts,
                router_z_loss_coef=moe_router_z_loss_coef,
                load_balance_loss_coef=moe_load_balance_loss_coef,
                router_jitter=moe_router_jitter,
                router_weight_mode=moe_router_weight_mode,
                expert_init_noise=moe_expert_init_noise,
            )
            self.linear1.requires_grad_(False)
            self.linear2.requires_grad_(False)

    def _mlp(self, x: Tensor) -> Tensor:
        if self.moe_ffn is not None:
            return self.moe_ffn(x)
        return super()._mlp(x)

    @torch.no_grad()
    def initialize_moe_from_dense(self) -> None:
        if self.moe_ffn is not None:
            self.moe_ffn.copy_from_dense(self.linear1, self.linear2)

    @torch.no_grad()
    def drop_dense_ffn(self) -> None:
        """Remove the frozen dense FFN copy from a MoE layer.

        ``linear1``/``linear2`` only stage dense weights for
        ``initialize_moe_from_dense`` and are never used in the forward
        pass once ``moe_ffn`` is attached. Dropping them shrinks
        checkpoints and frees dead weights. Idempotent; no-op for
        non-MoE layers.
        """
        if self.moe_ffn is not None:
            for name in ("linear1", "linear2"):
                if hasattr(self, name):
                    delattr(self, name)


class AttnResEncoderLightRMSNormMoE(AttnResEncoderLightRMSNorm):
    def __init__(
        self,
        *args,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        moe_num_shared_experts: int = 1,
        moe_layers: str = "none",
        moe_router_z_loss_coef: float = 1e-3,
        moe_load_balance_loss_coef: float = 1e-2,
        moe_router_jitter: float = 0.0,
        moe_router_weight_mode: str = "normalized",
        moe_expert_init_noise: float = 0.0,
        **kwargs,
    ):
        ssmax = kwargs.get("ssmax", False)
        super().__init__(*args, **kwargs)
        moe_layer_indices = self._resolve_moe_layers(len(self.layers), moe_layers)
        for block_idx in moe_layer_indices:
            dense_layer = self.layers[block_idx]
            moe_layer = AttnResTransformerLayerLightRMSNormMoE(
                d_model=dense_layer.attn.embed_dim,
                nhead=dense_layer.attn.num_heads,
                dim_feedforward=dense_layer.linear1.out_features,
                dropout=dense_layer.dropout.p,
                activation=dense_layer.activation,
                norm_first=dense_layer.norm_first,
                ssmax=ssmax,
                moe_num_experts=moe_num_experts,
                moe_top_k=moe_top_k,
                moe_num_shared_experts=moe_num_shared_experts,
                moe_router_z_loss_coef=moe_router_z_loss_coef,
                moe_load_balance_loss_coef=moe_load_balance_loss_coef,
                moe_router_jitter=moe_router_jitter,
                moe_router_weight_mode=moe_router_weight_mode,
                moe_expert_init_noise=moe_expert_init_noise,
            )
            moe_layer.load_state_dict(dense_layer.state_dict(), strict=False)
            self.layers[block_idx] = moe_layer

    @staticmethod
    def _resolve_moe_layers(num_blocks: int, moe_layers: str) -> set[int]:
        if not moe_layers or moe_layers == "none":
            return set()
        if moe_layers == "all":
            return set(range(num_blocks))
        if moe_layers == "last_half":
            return set(range(num_blocks // 2, num_blocks))
        if moe_layers in {"last_8", "last8"}:
            return set(range(max(num_blocks - 8, 0), num_blocks))
        if moe_layers.startswith("every_"):
            stride = int(moe_layers.split("_", 1)[1])
            if stride <= 0:
                raise ValueError("moe_layers stride must be positive")
            return set(range(stride - 1, num_blocks, stride))
        indices = set()
        for item in moe_layers.split(","):
            item = item.strip()
            if not item:
                continue
            index = int(item)
            if index < 0:
                index = num_blocks + index
            if index < 0 or index >= num_blocks:
                raise ValueError(f"MoE layer index {item} out of range for {num_blocks} blocks")
            indices.add(index)
        return indices

    @torch.no_grad()
    def initialize_moe_from_dense(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "initialize_moe_from_dense"):
                layer.initialize_moe_from_dense()

    @torch.no_grad()
    def drop_dense_ffn(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "drop_dense_ffn"):
                layer.drop_dense_ffn()

    def moe_aux_loss(self) -> Tensor:
        return collect_moe_aux_loss(self)

    def moe_aux_stats(self) -> dict[str, float]:
        return collect_moe_aux_stats(self)


class ICLearningAttnResLightRMSNormMoE(ICLearningAttnResLightRMSNorm):
    def __init__(
        self,
        *args,
        block_size: int = 4,
        attnres_stride: int = 2,
        ssmax=False,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        moe_num_shared_experts: int = 1,
        moe_layers: str = "none",
        moe_router_z_loss_coef: float = 1e-3,
        moe_load_balance_loss_coef: float = 1e-2,
        moe_router_jitter: float = 0.0,
        moe_router_weight_mode: str = "normalized",
        moe_expert_init_noise: float = 0.0,
        moe_init_from_dense: bool = True,
        recompute: bool = False,
        **kwargs,
    ):
        super().__init__(*args, block_size=block_size, attnres_stride=attnres_stride, ssmax=ssmax, recompute=recompute, **kwargs)
        dense_encoder = self.tf_icl
        self.tf_icl = AttnResEncoderLightRMSNormMoE(
            num_blocks=len(dense_encoder.layers),
            d_model=dense_encoder.layers[0].attn.embed_dim,
            nhead=dense_encoder.layers[0].attn.num_heads,
            dim_feedforward=dense_encoder.layers[0].linear1.out_features,
            dropout=dense_encoder.layers[0].dropout.p,
            activation=dense_encoder.layers[0].activation,
            norm_first=dense_encoder.layers[0].norm_first,
            block_size=block_size,
            use_rope=dense_encoder.rope is not None,
            rope_base=100000,
            rope_interleaved=True,
            ssmax=ssmax,
            seed_initial_block=dense_encoder.seed_initial_block,
            attnres_stride=attnres_stride,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_num_shared_experts=moe_num_shared_experts,
            moe_layers=moe_layers,
            moe_router_z_loss_coef=moe_router_z_loss_coef,
            moe_load_balance_loss_coef=moe_load_balance_loss_coef,
            moe_router_jitter=moe_router_jitter,
            moe_router_weight_mode=moe_router_weight_mode,
            moe_expert_init_noise=moe_expert_init_noise,
            recompute=recompute,
        )
        self.tf_icl.load_state_dict(dense_encoder.state_dict(), strict=False)
        del dense_encoder
        if moe_num_experts > 0 and moe_init_from_dense:
            self.tf_icl.initialize_moe_from_dense()

    def moe_aux_loss(self) -> Tensor:
        return self.tf_icl.moe_aux_loss()

    def moe_aux_stats(self) -> dict[str, float]:
        return self.tf_icl.moe_aux_stats()

    @torch.no_grad()
    def initialize_moe_from_dense(self) -> None:
        self.tf_icl.initialize_moe_from_dense()

    @torch.no_grad()
    def drop_dense_ffn(self) -> None:
        self.tf_icl.drop_dense_ffn()


class TabLDMMoE(TabLDMAttnResLightRMSNorm):
    moe_preset: Optional[int] = None

    @classmethod
    def moe_variant(cls, variant: int):
        if variant != 1:
            raise ValueError(f"Unknown TabLDM MoE variant: {variant!r}")
        return TabLDMSparseMoE

    def __init__(
        self,
        *args,
        block_size: int = 4,
        attnres_stride: int = 2,
        moe_num_experts: Optional[int] = None,
        moe_top_k: Optional[int] = None,
        moe_num_shared_experts: Optional[int] = None,
        moe_layers: Optional[str] = None,
        moe_router_z_loss_coef: float = 1e-3,
        moe_load_balance_loss_coef: float = 1e-2,
        moe_router_jitter: float = 0.0,
        moe_router_weight_mode: str = "normalized",
        moe_expert_init_noise: float = 0.0,
        moe_init_from_dense: bool = True,
        dual_stream: bool = True,
        global_dilation="adaptive",
        global_max_span: int = 32,
        **kwargs,
    ):
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
            if legacy_name in kwargs:
                value = kwargs[legacy_name]
                if native_name == "moe_num_experts" and moe_num_experts is None:
                    moe_num_experts = value
                elif native_name == "moe_top_k" and moe_top_k is None:
                    moe_top_k = value
                elif native_name == "moe_num_shared_experts" and moe_num_shared_experts is None:
                    moe_num_shared_experts = value
                elif native_name == "moe_layers" and moe_layers is None:
                    moe_layers = value
                elif native_name == "moe_router_z_loss_coef":
                    moe_router_z_loss_coef = value
                elif native_name == "moe_load_balance_loss_coef":
                    moe_load_balance_loss_coef = value
                elif native_name == "moe_router_jitter":
                    moe_router_jitter = value
                elif native_name == "moe_router_weight_mode":
                    moe_router_weight_mode = value
                elif native_name == "moe_expert_init_noise":
                    moe_expert_init_noise = value
                elif native_name == "moe_init_from_dense":
                    moe_init_from_dense = value
                kwargs.pop(legacy_name)

        presets = {
            1: dict(moe_num_experts=2, moe_top_k=1, moe_num_shared_experts=1, moe_layers="last_8"),
            2: dict(moe_num_experts=4, moe_top_k=2, moe_num_shared_experts=1, moe_layers="last_8"),
            3: dict(moe_num_experts=8, moe_top_k=2, moe_num_shared_experts=1, moe_layers="all"),
        }
        preset = presets.get(self.moe_preset, {})
        moe_num_experts = preset.get("moe_num_experts", 0) if moe_num_experts is None else moe_num_experts
        moe_top_k = preset.get("moe_top_k", 2) if moe_top_k is None else moe_top_k
        moe_num_shared_experts = (
            preset.get("moe_num_shared_experts", 1)
            if moe_num_shared_experts is None
            else moe_num_shared_experts
        )
        moe_layers = preset.get("moe_layers", "none") if moe_layers is None else moe_layers

        parent_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            not in {
                "moe_num_experts",
                "moe_top_k",
                "moe_num_shared_experts",
                "moe_layers",
                "moe_router_z_loss_coef",
                "moe_load_balance_loss_coef",
                "moe_router_jitter",
                "moe_router_weight_mode",
                "moe_expert_init_noise",
                "moe_init_from_dense",
                "global_dilation",
                "global_max_span",
            }
        }
        super().__init__(*args, block_size=block_size, attnres_stride=attnres_stride, **parent_kwargs)
        old_predictor = self.icl_predictor
        self.icl_predictor = ICLearningAttnResLightRMSNormMoE(
            out_dim=self.max_classes if self.max_classes > 0 else self.num_quantiles,
            max_classes=self.max_classes,
            d_model=self.embed_dim * self.row_num_cls,
            num_blocks=self.icl_num_blocks,
            nhead=self.icl_nhead,
            dim_feedforward=self.embed_dim * self.row_num_cls * self.ff_factor,
            dropout=self.dropout,
            activation=self.activation,
            norm_first=self.norm_first,
            ssmax=self.icl_ssmax,
            block_size=block_size,
            attnres_stride=attnres_stride,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_num_shared_experts=moe_num_shared_experts,
            moe_layers=moe_layers,
            moe_router_z_loss_coef=moe_router_z_loss_coef,
            moe_load_balance_loss_coef=moe_load_balance_loss_coef,
            moe_router_jitter=moe_router_jitter,
            moe_router_weight_mode=moe_router_weight_mode,
            moe_expert_init_noise=moe_expert_init_noise,
            moe_init_from_dense=False,
            recompute=kwargs.get("recompute", False),
        )
        self.icl_predictor.load_state_dict(old_predictor.state_dict(), strict=False)
        if moe_num_experts > 0 and moe_init_from_dense:
            self.icl_predictor.initialize_moe_from_dense()
        del old_predictor

        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_num_shared_experts = moe_num_shared_experts
        self.moe_layers = moe_layers
        self.moe_router_z_loss_coef = moe_router_z_loss_coef
        self.moe_load_balance_loss_coef = moe_load_balance_loss_coef
        self.moe_router_jitter = moe_router_jitter
        self.moe_router_weight_mode = moe_router_weight_mode
        self.moe_expert_init_noise = moe_expert_init_noise
        self.moe_init_from_dense = moe_init_from_dense

        if dual_stream:
            self.col_embedder = ColEmbeddingDualStream(
                embed_dim=self.embed_dim,
                num_blocks=self.col_num_blocks,
                nhead=self.col_nhead,
                dim_feedforward=self.embed_dim * self.ff_factor,
                num_inds=self.col_num_inds,
                dropout=self.dropout,
                activation=self.activation,
                norm_first=self.norm_first,
                bias_free_ln=self.bias_free_ln,
                affine=self.col_affine,
                feature_group=self.col_feature_group,
                feature_group_size=self.col_feature_group_size,
                global_dilation=global_dilation,
                global_max_span=global_max_span,
                target_aware=self.col_target_aware,
                max_classes=self.max_classes,
                reserve_cls_tokens=self.row_num_cls,
                ssmax=self.col_ssmax,
                zero_init=self.zero_init,
                mixed_radix_ensemble=True,
                recompute=kwargs.get("recompute", False),
            )

    def moe_aux_loss(self) -> Tensor:
        return self.icl_predictor.moe_aux_loss()

    def moe_aux_stats(self) -> dict[str, float]:
        return self.icl_predictor.moe_aux_stats()

    @torch.no_grad()
    def initialize_moe_from_dense(self) -> None:
        self.icl_predictor.tf_icl.initialize_moe_from_dense()

    @torch.no_grad()
    def drop_dense_ffn(self) -> None:
        self.icl_predictor.drop_dense_ffn()


class TabLDMSparseMoE(TabLDMMoE):
    moe_preset = 1
