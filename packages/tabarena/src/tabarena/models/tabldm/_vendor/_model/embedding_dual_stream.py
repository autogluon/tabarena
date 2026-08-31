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
from __future__ import annotations

from typing import List, Optional, Union, Literal
from collections import OrderedDict
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .layers import SkippableLinear, OneHotAndLinear
from .encoders import SetTransformer
from .kv_cache import KVCache
from .inference import InferenceManager
from .inference_config import MgrConfig, InferenceConfig


class ColEmbeddingDualStream(nn.Module):
    """Dual-stream column embedding with shared in_linear and additive fusion.

    Two parallel feature grouping streams:
    - Local stream: fixed shifts [2^0, 2^1, ..., 2^{k-1}] (original TabICLv2)
    - Global stream: configurable large-stride dilation shifts

    Both streams share the same in_linear projection. Their outputs are summed
    before entering the SetTransformer, so tf_col runs only once.

    Parameters
    ----------
    All parameters from ColEmbedding are supported, plus:

    global_dilation : None, str, or List[int], default="adaptive"
        Controls the dilation strategy for the global stream:
        - None or "default": same as local ([2^0, ..., 2^{k-1}]), effectively
          doubling the local signal (2x local, no global benefit)
        - "adaptive": log-spaced shifts adapted to H, targeting global_max_span
        - List[int]: explicit dilation rates, e.g. [1, 8, 32]

    global_max_span : int, default=32
        Maximum target span for adaptive global dilation mode.
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        affine: bool = True,
        feature_group: Union[bool, Literal["same", "valid"]] = False,
        feature_group_size: int = 3,
        global_dilation: Union[None, str, List[int]] = "adaptive",
        global_max_span: int = 32,
        target_aware: bool = False,
        max_classes: int = 10,
        reserve_cls_tokens: int = 4,
        ssmax: Union[bool, str] = False,
        zero_init: bool = True,
        mixed_radix_ensemble: bool = True,
        recompute: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.reserve_cls_tokens = reserve_cls_tokens
        self.feature_group = feature_group
        self.feature_group_size = feature_group_size
        self.global_dilation = global_dilation
        self.global_max_span = global_max_span
        self.target_aware = target_aware
        self.max_classes = max_classes
        self.affine = affine
        self.mixed_radix_ensemble = mixed_radix_ensemble

        # Shared in_linear for both streams
        self.in_linear = SkippableLinear(feature_group_size if feature_group else 1, embed_dim)

        self.tf_col = SetTransformer(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_inds=num_inds,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            ssmax=ssmax,
            zero_init=zero_init,
            recompute=recompute,
        )

        if target_aware:
            if max_classes > 0:
                self.y_encoder = OneHotAndLinear(max_classes, embed_dim)
            else:
                self.y_encoder = nn.Linear(1, embed_dim)

        if affine:
            self.out_w = SkippableLinear(embed_dim, embed_dim)
            self.ln_w = nn.LayerNorm(embed_dim, bias=not bias_free_ln) if norm_first else nn.Identity()

            self.out_b = SkippableLinear(embed_dim, embed_dim)
            self.ln_b = nn.LayerNorm(embed_dim, bias=not bias_free_ln) if norm_first else nn.Identity()

        self.inference_mgr = InferenceManager(enc_name="tf_col", out_dim=embed_dim)

    # ------------------------------------------------------------------
    # Global dilation shift computation
    # ------------------------------------------------------------------

    def _compute_global_shifts(self, H: int) -> List[int]:
        """Compute global stream dilation shifts based on strategy and current H."""
        size = self.feature_group_size
        dilation = self.global_dilation

        if dilation is None or dilation == "default":
            return [2**i for i in range(size)]

        if dilation == "adaptive":
            max_span = min(self.global_max_span, H - 1) if H > 1 else 1
            if size == 1:
                return [1]
            rates = []
            for i in range(size):
                r = int(round(max_span ** (i / (size - 1))))
                r = max(1, r)
                rates.append(r)
            rates = self._ensure_distinct_mod_h(rates, H)
            return rates

        if isinstance(dilation, (list, tuple)):
            rates = list(dilation)
            if len(rates) != size:
                raise ValueError(
                    f"global_dilation list length ({len(rates)}) must match "
                    f"feature_group_size ({size})"
                )
            rates = self._ensure_distinct_mod_h(rates, H)
            return rates

        raise ValueError(f"Unknown global_dilation: {dilation}")

    @staticmethod
    def _ensure_distinct_mod_h(rates: List[int], H: int) -> List[int]:
        """Adjust shift rates so they are all distinct modulo H."""
        if H <= 1:
            return [0] * len(rates)

        result = []
        seen_mod = set()
        for r in rates:
            mod_val = r % H
            if mod_val == 0:
                mod_val = 1
            attempts = 0
            while mod_val in seen_mod and attempts < H:
                mod_val = (mod_val + 1) % H
                if mod_val == 0:
                    mod_val = 1
                attempts += 1
            seen_mod.add(mod_val)
            result.append(mod_val)
        return result

    # ------------------------------------------------------------------
    # Feature grouping: local and global streams
    # ------------------------------------------------------------------

    def feature_grouping_local(self, X: Tensor) -> Tensor:
        """Local stream: fixed shifts [2^0, 2^1, ..., 2^{k-1}]."""
        if not self.feature_group:
            return X.unsqueeze(-1)

        B, T, H = X.shape
        size = self.feature_group_size
        mode = "same" if self.feature_group is True else self.feature_group

        if mode == "same":
            if H <= size:
                x_pad_cols = (size - H % size) % size
                if x_pad_cols > 0:
                    X = F.pad(X, (0, x_pad_cols), value=0)
                return X.reshape(B, T, -1, size)

            idxs = torch.arange(H, dtype=torch.long, device=X.device)
            X = torch.stack([X[:, :, (idxs + 2**i) % H] for i in range(size)], dim=-1)
        else:
            x_pad_cols = (size - H % size) % size
            if x_pad_cols > 0:
                X = F.pad(X, (0, x_pad_cols), value=0)
            X = X.reshape(B, T, -1, size)

        return X

    def feature_grouping_global(self, X: Tensor) -> Tensor:
        """Global stream: configurable dilation shifts."""
        if not self.feature_group:
            return X.unsqueeze(-1)

        B, T, H = X.shape
        size = self.feature_group_size
        mode = "same" if self.feature_group is True else self.feature_group

        if mode == "same":
            if H <= size:
                x_pad_cols = (size - H % size) % size
                if x_pad_cols > 0:
                    X = F.pad(X, (0, x_pad_cols), value=0)
                return X.reshape(B, T, -1, size)

            shifts = self._compute_global_shifts(H)
            idxs = torch.arange(H, dtype=torch.long, device=X.device)
            X = torch.stack([X[:, :, (idxs + s) % H] for s in shifts], dim=-1)
        else:
            x_pad_cols = (size - H % size) % size
            if x_pad_cols > 0:
                X = F.pad(X, (0, x_pad_cols), value=0)
            X = X.reshape(B, T, -1, size)

        return X

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def map_feature_shuffle(reference_pattern: List[int], other_pattern: List[int]) -> List[int]:
        orig_to_other = {feature: idx for idx, feature in enumerate(other_pattern)}
        mapping = [orig_to_other[feature] for feature in reference_pattern]
        return mapping

    def _compute_mixed_radix_bases(self, num_classes: int) -> List[int]:
        if num_classes <= self.max_classes:
            return [num_classes]

        D = math.ceil(math.log(num_classes) / math.log(self.max_classes))
        k = math.ceil(num_classes ** (1.0 / D))
        k = min(k, self.max_classes)

        bases = [k] * D
        product = k**D
        idx = 0
        while product < num_classes and idx < D:
            if bases[idx] < self.max_classes:
                product = product // bases[idx] * (bases[idx] + 1)
                bases[idx] += 1
            idx += 1

        return bases

    def _extract_mixed_radix_digit(self, y: Tensor, digit_idx: int, bases: List[int]) -> Tensor:
        divisor = 1
        for i in range(digit_idx + 1, len(bases)):
            divisor *= bases[i]
        return (y.long() // divisor) % bases[digit_idx]

    # ------------------------------------------------------------------
    # Core computation: dual-stream fusion
    # ------------------------------------------------------------------

    def _compute_embeddings(
        self,
        features_local: Tensor,
        features_global: Tensor,
        train_size: int,
        y_train: Optional[Tensor] = None,
        embed_with_test: bool = False,
    ) -> Tensor:
        """Shared in_linear + additive fusion, then tf_col once."""
        # Dual-stream fusion: shared in_linear, additive merge
        src = self.in_linear(features_local) + self.in_linear(features_global)

        # Keep pre-tf_col projection for affine path
        src_projected = src

        if not self.target_aware:
            src = self.tf_col(src, train_size=None if embed_with_test else train_size)
        else:
            assert y_train is not None, "y_train must be provided when target_aware=True."

            if self.max_classes > 0:
                num_classes = int(y_train.max().item()) + 1
                needs_mixed_radix = self.max_classes > 0 and num_classes > self.max_classes
            else:
                needs_mixed_radix = False

            if not needs_mixed_radix:
                if self.max_classes > 0:
                    y_emb = self.y_encoder(y_train.float())
                else:
                    y_emb = self.y_encoder(y_train.unsqueeze(-1))
                src[..., :train_size, :] = src[..., :train_size, :] + y_emb
                src = self.tf_col(src, train_size=None if embed_with_test else train_size)
            else:
                if not self.mixed_radix_ensemble:
                    raise ValueError(
                        f"Number of classes ({num_classes}) exceeds max_classes ({self.max_classes}). "
                        f"Set mixed_radix_ensemble=True to enable mixed-radix ensembling."
                    )

                bases = self._compute_mixed_radix_bases(num_classes)
                num_digits = len(bases)
                src_accum = torch.zeros_like(src)
                src_with_y = src.clone()

                for digit_idx in range(num_digits):
                    y_digit = self._extract_mixed_radix_digit(y_train, digit_idx, bases)
                    y_emb = self.y_encoder(y_digit.float())
                    src_with_y[..., :train_size, :] = src[..., :train_size, :] + y_emb
                    src_accum = src_accum + self.tf_col(src_with_y, train_size=None if embed_with_test else train_size)

                src = src_accum / num_digits

        if self.affine:
            weights = self.ln_w(self.out_w(src))
            biases = self.ln_b(self.out_b(src))
            embeddings = src_projected * weights + biases
        else:
            embeddings = src

        return embeddings

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def _train_forward(
        self, X: Tensor, y_train: Tensor, d: Optional[Tensor] = None, embed_with_test: bool = False
    ) -> Tensor:
        if self.feature_group:
            return self._train_forward_with_feature_group(X, y_train, embed_with_test)
        else:
            return self._train_forward_without_feature_group(X, y_train, d, embed_with_test)

    def _train_forward_with_feature_group(self, X: Tensor, y_train: Tensor, embed_with_test: bool) -> Tensor:
        train_size = y_train.shape[1]

        X_local = self.feature_grouping_local(X)
        X_global = self.feature_grouping_global(X)

        if self.reserve_cls_tokens > 0:
            X_local = F.pad(X_local, (0, 0, self.reserve_cls_tokens, 0), value=-100.0)
            X_global = F.pad(X_global, (0, 0, self.reserve_cls_tokens, 0), value=-100.0)

        features_local = X_local.transpose(1, 2)
        features_global = X_global.transpose(1, 2)

        if self.target_aware:
            assert y_train is not None, "y_train must be provided when target_aware=True."
            y_train = y_train.unsqueeze(1).expand(-1, features_local.shape[1], -1)

        embeddings = self._compute_embeddings(features_local, features_global, train_size, y_train, embed_with_test)
        return embeddings.transpose(1, 2)

    def _train_forward_without_feature_group(
        self, X: Tensor, y_train: Tensor, d: Optional[Tensor], embed_with_test: bool
    ) -> Tensor:
        """Without feature group, both streams are identical (single scalar per cell)."""
        train_size = y_train.shape[1]

        if self.reserve_cls_tokens > 0:
            X = F.pad(X, (self.reserve_cls_tokens, 0), value=-100.0)

        if d is None:
            features = X.transpose(1, 2).unsqueeze(-1)
            if self.target_aware:
                assert y_train is not None, "y_train must be provided when target_aware=True."
                y_train = y_train.unsqueeze(1).expand(-1, features.shape[1], -1)
            # No dual-stream when feature_group is disabled (single scalar input)
            embeddings = self._compute_embeddings(features, features, train_size, y_train, embed_with_test)
        else:
            if self.reserve_cls_tokens > 0:
                d = d + self.reserve_cls_tokens

            B, T, HC = X.shape
            X = X.transpose(1, 2)

            indices = torch.arange(HC, device=X.device).unsqueeze(0).expand(B, HC)
            mask = indices < d.unsqueeze(1)
            features = X[mask].unsqueeze(-1)

            if self.target_aware:
                assert y_train is not None, "y_train must be provided when target_aware=True."
                y_train = y_train.unsqueeze(1).expand(-1, HC, -1)
                y_train = y_train[mask]

            effective_embeddings = self._compute_embeddings(
                features, features, train_size, y_train, embed_with_test
            )

            embeddings = torch.zeros(B, HC, T, self.embed_dim, device=X.device, dtype=effective_embeddings.dtype)
            embeddings[mask] = effective_embeddings

        return embeddings.transpose(1, 2)

    # ------------------------------------------------------------------
    # Inference forward
    # ------------------------------------------------------------------

    def _inference_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        embed_with_test: bool = False,
        feature_shuffles: Optional[List[List[int]]] = None,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        if mgr_config is None:
            mgr_config = InferenceConfig().COL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        train_size = y_train.shape[1]
        if self.feature_group:
            embeddings = self._inference_with_feature_group(X, y_train, train_size, embed_with_test)
        else:
            embeddings = self._inference_without_feature_group(
                X, y_train, train_size, embed_with_test, feature_shuffles
            )

        return embeddings.transpose(1, 2)

    def _inference_with_feature_group(
        self, X: Tensor, y_train: Tensor, train_size: int, embed_with_test: bool
    ) -> Tensor:
        X_local = self.feature_grouping_local(X)
        X_global = self.feature_grouping_global(X)

        if self.reserve_cls_tokens > 0:
            X_local = F.pad(X_local, (0, 0, self.reserve_cls_tokens, 0), value=-100.0)
            X_global = F.pad(X_global, (0, 0, self.reserve_cls_tokens, 0), value=-100.0)

        features_local = X_local.transpose(1, 2)
        features_global = X_global.transpose(1, 2)

        if self.target_aware:
            assert y_train is not None, "y_train must be provided when target_aware=True."
            y_train = y_train.unsqueeze(1).expand(-1, features_local.shape[1], -1)
        else:
            y_train = None

        return self.inference_mgr(
            self._compute_embeddings,
            inputs=OrderedDict(
                [
                    ("features_local", features_local),
                    ("features_global", features_global),
                    ("train_size", train_size),
                    ("y_train", y_train),
                    ("embed_with_test", embed_with_test),
                ]
            ),
        )

    def _inference_without_feature_group(
        self,
        X: Tensor,
        y_train: Tensor,
        train_size: int,
        embed_with_test: bool,
        feature_shuffles: Optional[List[List[int]]],
    ) -> Tensor:
        if feature_shuffles is None:
            if self.reserve_cls_tokens > 0:
                X = F.pad(X, (self.reserve_cls_tokens, 0), value=-100.0)

            features = X.transpose(1, 2).unsqueeze(-1)
            if self.target_aware:
                assert y_train is not None, "y_train must be provided when target_aware=True."
                y_train = y_train.unsqueeze(1).expand(-1, features.shape[1], -1)
            else:
                y_train = None

            embeddings = self.inference_mgr(
                self._compute_embeddings,
                inputs=OrderedDict(
                    [
                        ("features_local", features),
                        ("features_global", features),
                        ("train_size", train_size),
                        ("y_train", y_train),
                        ("embed_with_test", embed_with_test),
                    ]
                ),
            )
        else:
            B = X.shape[0]
            first_table = X[0]
            if self.reserve_cls_tokens > 0:
                first_table = F.pad(first_table, (self.reserve_cls_tokens, 0), value=-100.0)

            features = first_table.transpose(0, 1).unsqueeze(-1)
            if self.target_aware:
                assert y_train is not None, "y_train must be provided when target_aware=True."
                y_first = y_train[0].unsqueeze(0).expand(features.shape[0], -1)
            else:
                y_first = None

            first_embeddings = self.inference_mgr(
                self._compute_embeddings,
                inputs=OrderedDict(
                    [
                        ("features_local", features),
                        ("features_global", features),
                        ("train_size", train_size),
                        ("y_train", y_first),
                        ("embed_with_test", embed_with_test),
                    ]
                ),
                output_repeat=B,
            )

            embeddings = first_embeddings.unsqueeze(0).repeat(B, 1, 1, 1)
            first_pattern = feature_shuffles[0]
            for i in range(1, B):
                mapping = self.map_feature_shuffle(first_pattern, feature_shuffles[i])
                if self.reserve_cls_tokens > 0:
                    mapping = [m + self.reserve_cls_tokens for m in mapping]
                    mapping = list(range(self.reserve_cls_tokens)) + mapping
                embeddings[i] = first_embeddings[mapping]

        return embeddings

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        embed_with_test: bool = False,
        feature_shuffles: Optional[List[List[int]]] = None,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        """Transform input table into embeddings using dual-stream fusion.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H).

        y_train : Tensor
            Target values for training samples of shape (B, train_size).

        d : Optional[Tensor], default=None
            The number of features per dataset of shape (B,).

        embed_with_test : bool, default=False
            If True, inducing points attend to all samples.

        feature_shuffles : Optional[List[List[int]]], default=None
            Feature shuffle patterns for inference.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager.

        Returns
        -------
        Tensor
            Embeddings of shape (B, T, G+C, E).
        """
        if self.training:
            embeddings = self._train_forward(X, y_train, d, embed_with_test)
        else:
            embeddings = self._inference_forward(X, y_train, embed_with_test, feature_shuffles, mgr_config)

        return embeddings

    # ------------------------------------------------------------------
    # KV cache support
    # ------------------------------------------------------------------

    def _compute_embeddings_with_cache(
        self,
        features_local: Tensor,
        features_global: Tensor,
        col_cache: KVCache,
        train_size: Optional[int] = None,
        y_train: Optional[Tensor] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        """Dual-stream fusion with KV cache."""
        src = self.in_linear(features_local) + self.in_linear(features_global)

        src_projected = src

        if not self.target_aware:
            src = self.tf_col.forward_with_cache(
                src, col_cache=col_cache, train_size=train_size, use_cache=use_cache, store_cache=store_cache
            )
        else:
            if store_cache:
                assert y_train is not None, "y_train must be provided when target_aware=True and store_cache=True."

                if self.max_classes > 0:
                    y_emb = self.y_encoder(y_train.float())
                else:
                    y_emb = self.y_encoder(y_train.unsqueeze(-1))
                src[..., :train_size, :] = src[..., :train_size, :] + y_emb

            src = self.tf_col.forward_with_cache(
                src, col_cache=col_cache, train_size=train_size, use_cache=use_cache, store_cache=store_cache
            )

        if self.affine:
            weights = self.ln_w(self.out_w(src))
            biases = self.ln_b(self.out_b(src))
            embeddings = src_projected * weights + biases
        else:
            embeddings = src

        return embeddings

    def forward_with_cache(
        self,
        X: Tensor,
        col_cache: KVCache,
        y_train: Optional[Tensor] = None,
        use_cache: bool = False,
        store_cache: bool = True,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")

        if store_cache:
            assert y_train is not None, "y_train must be provided when store_cache=True"
            if self.target_aware and self.max_classes > 0:
                num_classes = int(y_train.max().item()) + 1
                if num_classes > self.max_classes:
                    raise ValueError(
                        f"KV caching is not supported for classification with more classes "
                        f"({num_classes}) than max_classes ({self.max_classes}). Mixed-radix ensemble "
                        f"requires multiple forward passes which is incompatible with caching."
                    )

        if mgr_config is None:
            mgr_config = InferenceConfig().COL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        if self.feature_group:
            X_local = self.feature_grouping_local(X)
            X_global = self.feature_grouping_global(X)
            if self.reserve_cls_tokens > 0:
                X_local = F.pad(X_local, (0, 0, self.reserve_cls_tokens, 0), value=-100.0)
                X_global = F.pad(X_global, (0, 0, self.reserve_cls_tokens, 0), value=-100.0)
            features_local = X_local.transpose(1, 2)
            features_global = X_global.transpose(1, 2)
        else:
            if self.reserve_cls_tokens > 0:
                X = F.pad(X, (self.reserve_cls_tokens, 0), value=-100.0)
            features_local = X.transpose(1, 2).unsqueeze(-1)
            features_global = features_local

        if store_cache:
            train_size = y_train.shape[1]
            y_train = y_train.unsqueeze(1).expand(-1, features_local.shape[1], -1)
        else:
            train_size = None
            y_train = None

        embeddings = self.inference_mgr(
            self._compute_embeddings_with_cache,
            inputs=OrderedDict(
                [
                    ("features_local", features_local),
                    ("features_global", features_global),
                    ("col_cache", col_cache),
                    ("train_size", train_size),
                    ("y_train", y_train),
                    ("use_cache", use_cache),
                    ("store_cache", store_cache),
                ]
            ),
        )
        return embeddings.transpose(1, 2)
