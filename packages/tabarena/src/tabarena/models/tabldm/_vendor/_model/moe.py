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
"""Sparse feed-forward MoE layers used by the isolated classifier variants."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FeedForwardExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0, activation: str | callable = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        if isinstance(activation, str):
            if activation == "gelu":
                self.activation = F.gelu
            elif activation == "relu":
                self.activation = F.relu
            else:
                raise ValueError(f"Unsupported activation for MoE expert: {activation}")
        else:
            self.activation = activation
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

    @torch.no_grad()
    def copy_from_dense(self, linear1: nn.Linear, linear2: nn.Linear) -> None:
        self.linear1.weight.copy_(linear1.weight)
        if linear1.bias is not None and self.linear1.bias is not None:
            self.linear1.bias.copy_(linear1.bias)
        self.linear2.weight.copy_(linear2.weight)
        if linear2.bias is not None and self.linear2.bias is not None:
            self.linear2.bias.copy_(linear2.bias)


class SparseMoEFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        num_experts: int = 4,
        top_k: int = 2,
        num_shared_experts: int = 1,
        router_z_loss_coef: float = 1e-3,
        load_balance_loss_coef: float = 1e-2,
        router_jitter: float = 0.0,
        router_weight_mode: str = "normalized",
        expert_init_noise: float = 0.0,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if num_shared_experts < 0:
            raise ValueError("num_shared_experts must be >= 0")

        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.router_z_loss_coef = router_z_loss_coef
        self.load_balance_loss_coef = load_balance_loss_coef
        self.router_jitter = router_jitter
        if router_weight_mode not in {"normalized", "raw"}:
            raise ValueError(f"router_weight_mode must be 'normalized' or 'raw', got {router_weight_mode!r}")
        self.router_weight_mode = router_weight_mode
        self.expert_init_noise = expert_init_noise

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            FeedForwardExpert(d_model, hidden_dim, dropout, activation) for _ in range(num_experts)
        )
        self.shared_experts = nn.ModuleList(
            FeedForwardExpert(d_model, hidden_dim, dropout, activation) for _ in range(num_shared_experts)
        )
        self.register_buffer("residual_format", torch.tensor(1, dtype=torch.uint8), persistent=True)
        self._last_aux: Dict[str, Tensor] = {}
        nn.init.normal_(self.router.weight, mean=0.0, std=1e-2)

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        flat_x = x.reshape(-1, orig_shape[-1])
        router_input = flat_x.float()
        if self.training and self.router_jitter > 0:
            noise = torch.empty_like(router_input).uniform_(1.0 - self.router_jitter, 1.0 + self.router_jitter)
            router_input = router_input * noise

        logits = self.router(router_input)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=self.top_k, dim=-1)
        if self.router_weight_mode == "raw":
            top_weights = top_probs
        else:
            top_weights = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        routed_out = torch.zeros_like(flat_x)
        unused_expert_dep = flat_x.new_zeros(())
        for expert_idx, expert in enumerate(self.experts):
            selected = top_indices == expert_idx
            if not selected.any():
                unused_expert_dep = unused_expert_dep + self._zero_module_dependency(expert, flat_x)
                continue
            token_idx, slot_idx = selected.nonzero(as_tuple=True)
            expert_out = expert(flat_x.index_select(0, token_idx)).to(dtype=routed_out.dtype)
            weights = top_weights[token_idx, slot_idx].to(dtype=routed_out.dtype).unsqueeze(-1)
            routed_out.index_add_(0, token_idx, expert_out * weights)

        if self.shared_experts:
            base_out = torch.zeros_like(flat_x)
            for expert in self.shared_experts:
                base_out = base_out + expert(flat_x).to(dtype=base_out.dtype)
            base_out = base_out / len(self.shared_experts)
        else:
            base_out = torch.zeros_like(flat_x)

        flat_out = routed_out + base_out + unused_expert_dep.to(dtype=routed_out.dtype)
        self._last_aux = self._compute_aux(logits, probs, top_indices, routed_out, base_out)
        return flat_out.reshape(orig_shape)

    @staticmethod
    def _zero_module_dependency(module: nn.Module, reference: Tensor) -> Tensor:
        zero = reference.new_zeros(())
        for param in module.parameters():
            zero = zero + param.sum().to(dtype=reference.dtype) * 0.0
        return zero

    def _compute_aux(
        self,
        logits: Tensor,
        probs: Tensor,
        top_indices: Tensor,
        routed_out: Tensor,
        base_out: Tensor,
    ) -> Dict[str, Tensor]:
        aux: Dict[str, Tensor] = {}
        if self.router_z_loss_coef > 0:
            z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
            aux["z_loss"] = z_loss * self.router_z_loss_coef
        if self.load_balance_loss_coef > 0:
            expert_mask = F.one_hot(top_indices, num_classes=self.num_experts).sum(dim=1).float()
            tokens_per_expert = expert_mask.mean(dim=0) / self.top_k
            router_prob_per_expert = probs.mean(dim=0)
            balance = self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)
            aux["load_balance_loss"] = balance * self.load_balance_loss_coef
            aux["load_entropy"] = -(tokens_per_expert * tokens_per_expert.clamp_min(1e-9).log()).sum()
            aux["load_entropy_norm"] = aux["load_entropy"] / torch.log(
                routed_out.new_tensor(float(self.num_experts))
            ).clamp_min(1e-9)
            aux["load_max"] = tokens_per_expert.max()
        aux["router_entropy"] = -(
            probs * probs.clamp_min(1e-9).log()
        ).sum(dim=-1).mean()
        base_rms = base_out.float().square().mean().sqrt()
        routed_rms = routed_out.float().square().mean().sqrt()
        aux["base_rms"] = base_rms
        aux["residual_rms"] = routed_rms
        aux["residual_base_ratio"] = routed_rms / base_rms.clamp_min(1e-9)
        return aux

    def aux_loss(self) -> Tensor:
        losses = [value for key, value in self._last_aux.items() if key.endswith("loss")]
        if not losses:
            return self.router.weight.new_zeros(())
        return torch.stack([loss.to(self.router.weight.dtype) for loss in losses]).sum()

    def aux_stats(self) -> Dict[str, float]:
        return {key: float(value.detach().float().cpu()) for key, value in self._last_aux.items()}

    @torch.no_grad()
    def copy_from_dense(self, linear1: nn.Linear, linear2: nn.Linear) -> None:
        for expert in self.shared_experts:
            expert.copy_from_dense(linear1, linear2)
        for expert in self.experts:
            expert.copy_from_dense(linear1, linear2)
            if self.router_weight_mode == "normalized":
                nn.init.zeros_(expert.linear2.weight)
                nn.init.zeros_(expert.linear2.bias)
            elif self.expert_init_noise > 0:
                expert.linear1.weight.add_(torch.randn_like(expert.linear1.weight) * self.expert_init_noise)
                expert.linear2.weight.add_(torch.randn_like(expert.linear2.weight) * self.expert_init_noise)
                if expert.linear1.bias is not None:
                    expert.linear1.bias.add_(torch.randn_like(expert.linear1.bias) * self.expert_init_noise)
                if expert.linear2.bias is not None:
                    expert.linear2.bias.add_(torch.randn_like(expert.linear2.bias) * self.expert_init_noise)


def collect_moe_aux_loss(module: nn.Module) -> Tensor:
    losses = [submodule.aux_loss() for submodule in module.modules() if isinstance(submodule, SparseMoEFeedForward)]
    if not losses:
        first_param = next(module.parameters(), None)
        if first_param is None:
            return torch.zeros(())
        return first_param.new_zeros(())
    return torch.stack(losses).mean()


def collect_moe_aux_stats(module: nn.Module) -> Dict[str, float]:
    moe_modules = [submodule for submodule in module.modules() if isinstance(submodule, SparseMoEFeedForward)]
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for layer_idx, moe in enumerate(moe_modules):
        for key, value in moe.aux_stats().items():
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
            totals[f"layer_{layer_idx}/{key}"] = value
            counts[f"layer_{layer_idx}/{key}"] = 1
    return {f"moe/{key}": totals[key] / max(counts[key], 1) for key in totals}
