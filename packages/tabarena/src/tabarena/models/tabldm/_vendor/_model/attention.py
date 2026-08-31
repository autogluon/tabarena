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
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .rope import RotaryEmbedding
from .kv_cache import KVCacheEntry

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn3

    HAS_FLASH_ATTN3 = True
except ImportError:
    HAS_FLASH_ATTN3 = False

try:
    from flash_attn import flash_attn_varlen_func as flash_attn2

    HAS_FLASH_ATTN2 = True
except ImportError:
    HAS_FLASH_ATTN2 = False

# Whether the FlashAttention fast path is active. Despite the legacy name (kept for
# backward compatibility with the ``--use_flash_attn3`` flag), the selected backend
# may be FlashAttention-3 or FlashAttention-2 (see ``_flash_backend``).
_use_flash_attn3 = True

# Preferred FlashAttention backend: "auto" (FA3 then FA2), "fa3", or "fa2". An
# explicit choice still falls back to the other FlashAttention variant when the
# requested one is unavailable; PyTorch SDPA is always the final fallback.
_flash_backend = "auto"

# Minimum head dimension for the FlashAttention fast path. Flash-attn2's CUDA
# kernels are not reliable for very small head dims (e.g. 16), which can cause
# illegal memory access. The col/row embedders use head_dim=16 (embed_dim=128,
# nhead=8); the ICL predictor uses head_dim=64. With the default threshold of
# 32, only the ICL predictor uses flash, while col/row fall back to SDPA.
_flash_min_head_dim = 32


def set_flash_attn_min_head_dim(min_head_dim: int):
    """Set the minimum head dimension for the FlashAttention fast path."""
    global _flash_min_head_dim
    _flash_min_head_dim = max(1, int(min_head_dim))


def _select_flash_backend() -> Optional[str]:
    """Resolve the FlashAttention backend given availability and preference.

    Returns ``"fa3"``, ``"fa2"``, or ``None`` (fall back to PyTorch SDPA).
    """
    if _flash_backend == "fa2":
        order = ("fa2", "fa3")
    else:  # "auto" and "fa3" both prefer FA3 first
        order = ("fa3", "fa2")
    for backend in order:
        if backend == "fa3" and HAS_FLASH_ATTN3:
            return "fa3"
        if backend == "fa2" and HAS_FLASH_ATTN2:
            return "fa2"
    return None


def resolve_flash_attn_backend() -> Optional[str]:
    """Return the backend that would be selected right now, or ``None`` for SDPA.

    Returns ``None`` when the fast path is disabled. Useful for logging.
    """
    return _select_flash_backend() if _use_flash_attn3 else None


@contextmanager
def flash_attn3_toggle(enabled: bool):
    """Context manager to temporarily enable or disable the FlashAttention path.

    Used by ``InferenceManager._run_forward()`` in ``inference.py`` to control
    whether the FlashAttention fast path is used during each forward pass based on
    the ``use_fa3`` configuration. The selected backend may be FA3 or FA2 (see
    ``set_flash_attn_backend``).
    """
    global _use_flash_attn3
    old = _use_flash_attn3
    _use_flash_attn3 = enabled
    try:
        yield
    finally:
        _use_flash_attn3 = old


def set_flash_attn3_enabled(enabled: bool):
    """Globally enable or disable the FlashAttention fast path for this process.

    Used by the pre-training ``Trainer`` (``--use_flash_attn3``): the fast path runs
    attention in fp16, so the TabLDMv2 recipe enables it only for stages 2 and 3.
    FlashAttention-3 is preferred when installed, falling back to FlashAttention-2.
    """
    global _use_flash_attn3
    _use_flash_attn3 = enabled


def set_flash_attn_backend(backend: str):
    """Set the preferred FlashAttention backend.

    Parameters
    ----------
    backend : {"auto", "fa3", "fa2"}
        ``"auto"`` prefers FA3 then FA2. ``"fa3"``/``"fa2"`` prefer the requested
        variant but still fall back to the other when it is unavailable.
    """
    global _flash_backend
    if backend not in ("auto", "fa3", "fa2"):
        raise ValueError(f"flash_attn_backend must be 'auto', 'fa3', or 'fa2', got {backend!r}")
    _flash_backend = backend


# ---------------------------------------------------------------------------
# Runtime observability
# ---------------------------------------------------------------------------
# Running counters for the flash fast path vs the SDPA fallback, plus a one-time
# first-use log so users can confirm flash-attn actually engaged at runtime.
_flash_call_count = 0
_sdpa_call_count = 0
_first_use_logged = False


def _is_master() -> bool:
    """True on rank 0, or when distributed training is not initialized."""
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def log_flash_attn_status(preference: str = "auto"):
    """Print a one-time summary of FlashAttention availability and config.

    Called by the trainers at model-build time. Reports whether FA3/FA2 are
    installed, the requested preference, and the backend that will be used.
    """
    if not _is_master():
        return
    backend = resolve_flash_attn_backend()
    print(
        f"[FlashAttention] enabled={_use_flash_attn3} | preference={preference} | "
        f"FA3 installed={HAS_FLASH_ATTN3}, FA2 installed={HAS_FLASH_ATTN2} | "
        f"resolved backend={backend or 'sdpa (fallback)'} | "
        f"min_head_dim={_flash_min_head_dim}"
    )


def get_flash_attn_call_stats() -> tuple:
    """Return ``(flash_calls, sdpa_calls)`` accumulated since process start."""
    return _flash_call_count, _sdpa_call_count


def sdpa_with_flattened_batch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    ssmax_layer: Optional[nn.Module] = None,
) -> Tensor:
    """Applies scaled dot-product attention with flattened batch dimensions.

    This function handles arbitrary batch dimensions by flattening them before
    applying PyTorch's ``scaled_dot_product_attention`` and then reshaping the
    output back to the original shape. This flattening is necessary to properly
    trigger Flash Attention.

    Parameters
    ----------
    q : Tensor
        Query tensor of shape (..., nh, tgt_len, hs) where:

        - ... represents arbitrary batch dimensions
        - nh is the number of attention heads
        - tgt_len is the target sequence length
        - hs is the head size (embedding dimension per head)

    k : Tensor
        Key tensor of shape (..., nh, src_len, hs) with matching batch dimensions.

    v : Tensor
        Value tensor of shape (..., nh, src_len, hs) with matching batch dimensions.

    attn_mask : Optional[Tensor], default=None
        Attention mask of shape (..., nh, tgt_len, src_len).

    dropout_p : float, default=0.0
        Dropout probability applied to attention weights.

    ssmax_layer : Optional[nn.Module], default=None
        If provided, applies scalable softmax (SSMax) scaling to queries before
        attention computation.

    Returns
    -------
    Tensor
        Attention output tensor of shape (..., nh, tgt_len, hs) preserving the
        original batch dimensions of the input.
    """

    q_shape = q.shape
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])
    if attn_mask is not None:
        attn_mask = attn_mask.reshape(-1, *attn_mask.shape[-3:])

    if ssmax_layer is not None:
        src_len = k.size(-2)
        q = ssmax_layer(q, src_len)

    # FlashAttention kernels don't support dropout or a custom attention mask.
    global _flash_call_count, _sdpa_call_count, _first_use_logged
    headdim = q.shape[-1]
    flash_backend = None
    if (
        _use_flash_attn3
        and q.is_cuda
        and attn_mask is None
        and dropout_p == 0.0
        and headdim >= _flash_min_head_dim
    ):
        flash_backend = _select_flash_backend()

    if flash_backend is not None:
        # FlashAttention only supports fp16, bf16 (and fp8_e4m3 on FA3). Use bf16
        # (not fp16) for the fp32→flash cast: bf16 has the same exponent range as
        # fp32, so ssmax-scaled queries with long sequences (seqlen up to 60k) won't
        # overflow. fp16 (max 65504) overflows easily here, producing NaN/Inf that
        # corrupts gradients and hangs DDP allreduce.
        orig_dtype = q.dtype
        fa_dtype = orig_dtype if orig_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

        flat_bs, nheads, seqlen_q, headdim = q.shape
        seqlen_k = k.shape[-2]
        q_fa = q.transpose(1, 2).reshape(flat_bs * seqlen_q, nheads, headdim).contiguous().to(fa_dtype)
        k_fa = k.transpose(1, 2).reshape(flat_bs * seqlen_k, nheads, headdim).contiguous().to(fa_dtype)
        v_fa = v.transpose(1, 2).reshape(flat_bs * seqlen_k, nheads, headdim).contiguous().to(fa_dtype)
        cu_seqlens_q = torch.arange(0, (flat_bs + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.arange(0, (flat_bs + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=q.device)
        fa_fn = flash_attn3 if flash_backend == "fa3" else flash_attn2
        _flash_call_count += 1
        if not _first_use_logged and _is_master():
            _first_use_logged = True
            print(
                f"[FlashAttention] kernel first use: backend={flash_backend}, "
                f"nheads={nheads}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, "
                f"headdim={headdim} (>={_flash_min_head_dim}), dtype={fa_dtype}"
            )
        out = fa_fn(q_fa, k_fa, v_fa, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k)
        out = out.view(flat_bs, seqlen_q, nheads, headdim).transpose(1, 2).to(orig_dtype)
    else:
        _sdpa_call_count += 1
        if _use_flash_attn3 and not _first_use_logged and _is_master():
            _first_use_logged = True
            if not q.is_cuda:
                reason = "non-cuda tensor"
            elif attn_mask is not None:
                reason = "attn_mask present"
            elif dropout_p != 0.0:
                reason = "dropout active"
            elif headdim < _flash_min_head_dim:
                reason = f"head_dim={headdim} < {_flash_min_head_dim}"
            elif not (HAS_FLASH_ATTN3 or HAS_FLASH_ATTN2):
                reason = "no flash_attn package installed"
            else:
                reason = f"no backend available for preference={_flash_backend}"
            print(f"[FlashAttention] fast path bypassed -> SDPA (reason={reason})")
        out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)

    return out.view(q_shape)


def multi_head_attention_forward(
    query: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    key: Optional[Tensor] = None,
    value: Optional[Tensor] = None,
    cached_kv: Optional[KVCacheEntry] = None,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    rope: Optional[RotaryEmbedding] = None,
    ssmax_layer: Optional[nn.Module] = None,
    need_kv: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Multi-head attention with support for rotary position embeddings.

    Parameters
    ----------
    query : Tensor
        Query tensor of shape (..., tgt_len, embed_dim).

    num_heads : int
        Number of attention heads.

    in_proj_weight : Tensor
        Combined weight matrix for Q, K, V input projections.

    in_proj_bias : Tensor
        Combined bias vector for input projections.

    dropout_p : float
        Dropout probability applied to attention weights.

    out_proj_weight : Tensor
        Output projection weight matrix.

    out_proj_bias : Tensor
        Output projection bias vector.

    key : Optional[Tensor], default=None
        Key tensor of shape (..., src_len, embed_dim).
        Required when ``cached_kv`` is None.

    value : Optional[Tensor], default=None
        Value tensor of shape (..., src_len, embed_dim).
        Required when ``cached_kv`` is None.

    cached_kv : Optional[KVCacheEntry], default=None
        Pre-computed key and value projections for caching. When provided:

        - key and value parameters are ignored
        - Only query projection is computed
        - cached_kv.key shape: (..., num_heads, src_len, head_dim)
        - cached_kv.value shape: (..., num_heads, src_len, head_dim)
        - RoPE is applied only to queries (keys should already have RoPE applied)

    training : bool, default=True
        Whether the model is in training mode (affects dropout).

    key_padding_mask : Optional[Tensor], default=None
        Mask of shape (..., src_len) that identifies padding elements
        in the key sequence to be ignored:

        - For binary masks: True values indicate positions to ignore.
        - For float masks: Values are directly added to attention scores.

    attn_mask : Optional[Tensor], default=None
        Attention mask of shape (tgt_len, src_len) or
        (..., num_heads, tgt_len, src_len).

    rope : Optional[RotaryEmbedding]
        Rotary positional encoding.

    ssmax_layer : Optional[nn.Module], default=None
        If provided, applies scalable softmax (SSMax) scaling to queries before
        attention computation.

    need_kv : bool, default=False
        If True and ``cached_kv`` is None, also returns the computed K and V
        projections along with the attention output. Useful for caching K/V for
        subsequent calls.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor, Tensor]]
        If ``need_kv`` is False or ``cached_kv`` is provided:
            Attention output tensor of shape (..., tgt_len, embed_dim).
        If ``need_kv`` is True and ``cached_kv`` is None:
            Tuple of (attn_output, k, v) where:

            - attn_output: shape (..., tgt_len, embed_dim)
            - k: shape (..., num_heads, src_len, head_dim)
            - v: shape (..., num_heads, src_len, head_dim)
    """

    # Extract shape information, supporting arbitrary batch dimensions
    *batch_shape, tgt_len, embed_dim = query.shape
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

    if cached_kv is None:
        # Standard: project Q, K, V jointly
        if key is None or value is None:
            raise ValueError("key and value must be provided when cached_kv is None")
        src_len = key.shape[-2]
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)
        k = k.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)
        v = v.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)
        if rope is not None:
            q = rope.rotate_queries_or_keys(q)
            k = rope.rotate_queries_or_keys(k)
    else:
        # Use cached K/V, project Q only
        k, v = cached_kv.key, cached_kv.value
        src_len = k.shape[-2]
        q_proj_weight = in_proj_weight[:embed_dim]
        q_proj_bias = in_proj_bias[:embed_dim] if in_proj_bias is not None else None
        q = F.linear(query, q_proj_weight, q_proj_bias)
        q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)
        if rope is not None:
            q = rope.rotate_queries_or_keys(q)

    # Disable dropout during evaluation
    if not training:
        dropout_p = 0.0

    # Process attention mask
    correct_2d_shape = (tgt_len, src_len)
    correct_nd_shape = (*batch_shape, num_heads, tgt_len, src_len)
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            if attn_mask.shape != correct_2d_shape:
                raise ValueError(f"2D attn_mask should have shape {correct_2d_shape}, but got {attn_mask.shape}")
            attn_mask = attn_mask.expand(*batch_shape, num_heads, tgt_len, src_len)
        elif attn_mask.dim() == len(correct_nd_shape):
            if attn_mask.shape != correct_nd_shape:
                raise ValueError(
                    f"{len(correct_nd_shape)}D attn_mask should have shape {correct_nd_shape}, "
                    f"but got {attn_mask.shape}"
                )
        else:
            raise ValueError(f"attn_mask must be 2D or {len(correct_nd_shape)}D, got {attn_mask.dim()}D")

    # Process key padding mask
    if key_padding_mask is not None:
        if key_padding_mask.shape != (*batch_shape, src_len):
            raise ValueError(
                f"key_padding_mask should have shape {(*batch_shape, src_len)}, but got {key_padding_mask.shape}"
            )
        key_padding_mask = key_padding_mask.view(*batch_shape, 1, 1, src_len).expand(
            *batch_shape, num_heads, tgt_len, src_len
        )

        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    attn_output = sdpa_with_flattened_batch(
        q, k, v, attn_mask, dropout_p, ssmax_layer=ssmax_layer
    )  # (..., nh, tgt_len, hs)

    # Reshape and project output
    attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)  # (batch_shape, tgt_len, E)

    if need_kv and cached_kv is None:
        return attn_output, k, v

    return attn_output
