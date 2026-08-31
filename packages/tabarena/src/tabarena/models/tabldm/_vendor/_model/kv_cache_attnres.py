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
"""
KV cache data structures for AttnRes model variants.

Extends the base KVCache/TabLDMCache with support for the block attention
residual architecture used in attnres_light.py and related modules.

The AttnRes encoder layers produce the same KV projections as the base
encoder, so the cache structure is identical — this module provides a
dedicated subclass for type clarity and potential future extensions
(e.g., caching block boundaries).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .kv_cache import KVCacheEntry, KVCache, TabLDMCache


@dataclass
class TabLDMAttnCache(TabLDMCache):
    """Cache container for TabLDMAttnResLight and related AttnRes models.

    Inherits all functionality from TabLDMCache. The AttnRes encoder produces
    identical KV projections to the base encoder (the AttnRes mechanism only
    affects how residuals are aggregated, not what is cached), so the cache
    layout is the same.

    Attributes
    ----------
    col_cache : Optional[KVCache]
        Cache for ColEmbedding ISAB blocks.

    row_repr : Optional[Tensor]
        Cached row representations from the model.

    icl_cache : Optional[KVCache]
        Cache for ICLearning AttnRes Encoder layers.

    train_shape : Tuple[int, int, int]
        Shape ``(batch_size, train_size, num_features)`` of training data the
        cache was built with.

    num_classes : Optional[int]
        Number of classes in classification tasks (0 for regression).
    """

    def slice_batch(self, start: int, end: int) -> TabLDMAttnCache:
        """Slice this cache along the batch dimension (dim 0).

        Returns
        -------
        TabLDMAttnCache
            New cache with sliced tensors.
        """
        indices = slice(start, end)
        return TabLDMAttnCache(
            col_cache=self.col_cache[indices] if self.col_cache else KVCache(),
            row_repr=self.row_repr[indices] if self.row_repr is not None else None,
            icl_cache=self.icl_cache[indices] if self.icl_cache else KVCache(),
            train_shape=(end - start, self.train_shape[1], self.train_shape[2]),
            num_classes=self.num_classes,
        )

    def to(self, device, dtype=None) -> TabLDMAttnCache:
        """Move all cached tensors to the given device and optionally cast dtype.

        Returns
        -------
        TabLDMAttnCache
            New cache with all tensors on the target device.
        """
        return TabLDMAttnCache(
            col_cache=self.col_cache.to(device, dtype=dtype) if self.col_cache else KVCache(),
            row_repr=self.row_repr.to(device=device, dtype=dtype) if self.row_repr is not None else None,
            icl_cache=self.icl_cache.to(device, dtype=dtype) if self.icl_cache else KVCache(),
            train_shape=self.train_shape,
            num_classes=self.num_classes,
        )

    @staticmethod
    def concat(caches: List[TabLDMAttnCache], dim: int = 0) -> TabLDMAttnCache:
        """Concatenate multiple TabLDMAttnCache objects along the batch dimension.

        Parameters
        ----------
        caches : List[TabLDMAttnCache]
            Caches to concatenate.

        dim : int, default=0
            Dimension to concatenate along (batch dimension).

        Returns
        -------
        TabLDMAttnCache
            New cache with concatenated entries.
        """
        col_caches = [c.col_cache for c in caches if c.col_cache is not None]
        row_reprs = [c.row_repr for c in caches if c.row_repr is not None]
        icl_caches = [c.icl_cache for c in caches if c.icl_cache is not None]

        total_batch = sum(c.train_shape[0] for c in caches)
        train_size = caches[0].train_shape[1]
        n_features = caches[0].train_shape[2]

        return TabLDMAttnCache(
            col_cache=KVCache.concat(col_caches, dim=dim) if col_caches else KVCache(),
            row_repr=torch.cat(row_reprs, dim=dim) if row_reprs else None,
            icl_cache=KVCache.concat(icl_caches, dim=dim) if icl_caches else KVCache(),
            train_shape=(total_batch, train_size, n_features),
            num_classes=caches[0].num_classes,
        )
