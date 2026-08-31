"""Speed-optimized Block AttnRes with RMSNorm replacing all LayerNorm.

Identical to ``attnres_light.py`` except every ``nn.LayerNorm`` is replaced
by ``RMSNorm``.  RMSNorm removes the mean-centering step (no bias, no
subtraction of mean) and normalizes by root-mean-square only, which:

1. Is faster (fewer ops, no mean reduction).
2. Matches modern LLM practice (LLaMA, Gemma, TabFM).
3. Removes the ``bias_free_ln`` parameter since RMSNorm never has bias.

The RMSNorm implementation follows the TabFM convention: normalization is
performed in float32 for numerical stability, then cast back to the input
dtype.  This matches ``tabfm/src/pytorch/model.py``.

Changes from ``attnres_light.py``:
- AttnResTransformerLayerLight: ``attn_norm`` and ``mlp_norm`` are RMSNorm.
- RowInteractionAttnResLight: ``out_ln`` is RMSNorm.
- ICLearningAttnResLight: ``self.ln`` is RMSNorm.
- MultiheadAttentionBlock's internal ``norm1``/``norm2`` are replaced via a
  local subclass ``MultiheadAttentionBlockRMSNorm``.
- ``bias_free_ln`` parameter is removed from all constructors (RMSNorm has
  no bias by definition).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union
from collections import OrderedDict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from .layers import MultiheadAttention, MultiheadAttentionBlock
from .kv_cache import KVCacheEntry, KVCache
from .kv_cache_attnres import TabLDMAttnCache
from .inference import InferenceManager
from .inference_config import MgrConfig, InferenceConfig
from .learning import ICLearning
from .tabldm import TabLDM


# ---------------------------------------------------------------------------
# RMSNorm (float32-stable, matching TabFM convention)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (no bias, float32 internal computation)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dt = x.dtype
        xf = x.float()
        v = xf.pow(2).mean(-1, keepdim=True)
        return ((xf * torch.rsqrt(v + self.eps)) * self.weight.float()).to(dt)


# ---------------------------------------------------------------------------
# MultiheadAttentionBlock with RMSNorm (replaces LayerNorm in norm1/norm2)
# ---------------------------------------------------------------------------

class MultiheadAttentionBlockRMSNorm(MultiheadAttentionBlock):
    """MultiheadAttentionBlock with RMSNorm instead of LayerNorm."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        ssmax: Union[bool, str] = False,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=True,
            ssmax=ssmax,
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)


# ---------------------------------------------------------------------------
# Vectorized block_attn_res (GPU-friendly, from attnres.py)
# ---------------------------------------------------------------------------

def block_attn_res(
    blocks: Sequence[Tensor],
    partial_block: Tensor,
    proj: nn.Linear,
    norm: RMSNorm,
) -> Tensor:
    """Inter-block attention residual — vectorized variant.

    Uses ``torch.stack`` + batched RMSNorm + broadcast weighted sum.
    GPU-friendly: no Python loops, all operations are batched.
    """
    V = torch.stack(list(blocks) + [partial_block])  # [N+1, ..., T, D]
    K = norm(V)
    weight = proj.weight.squeeze()  # [D]
    logits = (K * weight).sum(dim=-1)  # [N+1, ..., T]
    attn = logits.softmax(dim=0)
    return (attn.unsqueeze(-1) * V).sum(dim=0)


# ---------------------------------------------------------------------------
# Transformer layer — AttnRes before attention only, standard MLP residual
# ---------------------------------------------------------------------------

class AttnResTransformerLayerLightRMSNorm(nn.Module):
    """Transformer layer with Block AttnRes before attention only (RMSNorm).

    Compared to ``AttnResTransformerLayerLight``:
    - ``attn_norm`` and ``mlp_norm`` use RMSNorm instead of LayerNorm.
    - No ``bias_free_ln`` parameter (RMSNorm has no bias).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: Union[str, Callable[[Tensor], Tensor]] = "gelu",
        norm_first: bool = True,
        ssmax: Union[bool, str] = False,
    ) -> None:
        super().__init__()

        self.attn = MultiheadAttention(d_model, nhead, dropout, ssmax)
        self.attn_norm = RMSNorm(d_model)
        self.mlp_norm = RMSNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        if isinstance(activation, str):
            self.activation = F.gelu if activation == "gelu" else F.relu
        else:
            self.activation = activation

        self.norm_first = norm_first

    def _mlp(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

    def _attn_mlp(
        self,
        h: Tensor,
        residual: Optional[Tensor],
        train_size: Optional[int],
        rope,
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        """Pure tensor-in/tensor-out core: self-attention + MLP residual.

        ``h`` is the (already AttnRes-aggregated) attention input; ``residual``
        is the ``partial_block`` carried into the attention residual (may be
        None at a block boundary). This is the training-path computation and
        is the part wrapped by gradient checkpointing — it holds the large
        attention/MLP activations. It contains no non-Tensor state, so
        ``torch.utils.checkpoint`` can recompute it exactly.
        """
        attn_input = self.attn_norm(h) if self.norm_first else h

        if train_size is not None:
            k = attn_input[..., :train_size, :]
            v = attn_input[..., :train_size, :]
        else:
            k = v = attn_input
        attn_out = self.attn(
            attn_input, key=k, value=v,
            key_padding_mask=key_padding_mask, rope=rope,
        )

        attn_out = self.dropout(attn_out)
        partial_block = attn_out if residual is None else residual + attn_out

        # MLP — standard pre-norm residual (no AttnRes)
        mlp_input = self.mlp_norm(partial_block) if self.norm_first else partial_block
        mlp_out = self.dropout(self._mlp(mlp_input))
        return partial_block + mlp_out

    def forward(
        self,
        blocks: List[Tensor],
        partial_block: Tensor,
        layer_number: int,
        block_size: int,
        train_size: Optional[int] = None,
        rope=None,
        key_padding_mask: Optional[Tensor] = None,
        cached_kv: Optional[KVCacheEntry] = None,
        need_kv: bool = False,
        attn_res_proj: Optional[nn.Linear] = None,
        attn_res_norm: Optional[RMSNorm] = None,
        recompute: bool = False,
    ) -> tuple:
        """Forward pass with optional AttnRes and KV cache.

        Returns
        -------
        tuple
            If need_kv is False: (blocks, partial_block)
            If need_kv is True:  (blocks, partial_block, k_proj, v_proj)
        """
        # AttnRes aggregation (only when proj/norm provided and blocks exist)
        if attn_res_proj is not None and len(blocks) > 0:
            h = block_attn_res(blocks, partial_block, attn_res_proj, attn_res_norm)
        else:
            h = partial_block

        # Block boundary handling
        if block_size > 0 and layer_number % block_size == 0:
            blocks.append(partial_block)
            partial_block = None

        # KV-cache / need_kv paths are inference-only (run under no_grad) and are
        # left exactly as before — checkpointing is a training-time memory trick.
        if cached_kv is not None or need_kv:
            attn_input = self.attn_norm(h) if self.norm_first else h
            k_proj, v_proj = None, None
            if cached_kv is not None:
                attn_out = self.attn(attn_input, cached_kv=cached_kv, rope=rope)
            else:
                if train_size is not None:
                    k = attn_input[..., :train_size, :]
                    v = attn_input[..., :train_size, :]
                else:
                    k = v = attn_input
                attn_result = self.attn(
                    attn_input, key=k, value=v,
                    key_padding_mask=key_padding_mask, rope=rope,
                    need_kv=need_kv,
                )
                if need_kv and isinstance(attn_result, tuple):
                    attn_out, k_proj, v_proj = attn_result
                else:
                    attn_out = attn_result

            attn_out = self.dropout(attn_out)
            partial_block = attn_out if partial_block is None else partial_block + attn_out
            mlp_input = self.mlp_norm(partial_block) if self.norm_first else partial_block
            mlp_out = self.dropout(self._mlp(mlp_input))
            partial_block = partial_block + mlp_out

            if need_kv:
                return blocks, partial_block, k_proj, v_proj
            return blocks, partial_block

        # Standard training path: optionally gradient-checkpoint the attn+MLP core.
        if recompute and self.training:
            partial_block = checkpoint(
                self._attn_mlp, h, partial_block, train_size, rope, key_padding_mask,
                use_reentrant=False,
            )
        else:
            partial_block = self._attn_mlp(h, partial_block, train_size, rope, key_padding_mask)

        return blocks, partial_block


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class AttnResEncoderLightRMSNorm(nn.Module):
    """Stack of AttnResTransformerLayerLightRMSNorm with stride-based AttnRes.

    Parameters
    ----------
    attnres_stride : int, default=2
        Apply AttnRes every ``attnres_stride`` layers. Layers in between use
        standard residual connections.
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        block_size: int = 4,
        use_rope: bool = False,
        rope_base: int = 100000,
        rope_interleaved: bool = True,
        ssmax: Union[bool, str] = False,
        seed_initial_block: bool = True,
        attnres_stride: int = 2,
        recompute: bool = False,
    ) -> None:
        super().__init__()
        self.recompute = recompute
        self.layers = nn.ModuleList(
            [
                AttnResTransformerLayerLightRMSNorm(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                    ssmax=ssmax,
                )
                for _ in range(num_blocks)
            ]
        )
        self.block_size = block_size
        self.seed_initial_block = seed_initial_block
        self.attnres_stride = attnres_stride

        # AttnRes parameters owned by encoder, only for layers that use them.
        self._ar_layer_indices: List[int] = []
        ar_projs = []
        ar_norms = []
        for layer_idx in range(1, num_blocks + 1):
            if layer_idx % attnres_stride == 0:
                self._ar_layer_indices.append(layer_idx)
                ar_projs.append(nn.Linear(d_model, 1, bias=False))
                ar_norms.append(RMSNorm(d_model))
        self.attn_res_projs = nn.ModuleList(ar_projs)
        self.attn_res_norms = nn.ModuleList(ar_norms)
        self._ar_idx_map = {idx: i for i, idx in enumerate(self._ar_layer_indices)}
        self.rope = None
        if use_rope:
            from .rope import RotaryEmbedding

            self.rope = RotaryEmbedding(dim=d_model // nhead, theta=rope_base, interleaved=rope_interleaved)

    def forward(
        self,
        src: Tensor,
        train_size: Optional[int] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        blocks: List[Tensor] = []
        partial = src
        if self.seed_initial_block:
            blocks.append(partial)
        for layer_idx, layer in enumerate(self.layers, start=1):
            ar_i = self._ar_idx_map.get(layer_idx)
            proj = self.attn_res_projs[ar_i] if ar_i is not None else None
            norm = self.attn_res_norms[ar_i] if ar_i is not None else None
            blocks, partial = layer(
                blocks=blocks,
                partial_block=partial,
                layer_number=layer_idx,
                block_size=self.block_size,
                train_size=train_size,
                rope=self.rope,
                key_padding_mask=key_padding_mask,
                attn_res_proj=proj,
                attn_res_norm=norm,
                recompute=self.recompute,
            )
        return partial

    def forward_with_cache(
        self,
        src: Tensor,
        icl_cache: KVCache,
        train_size: Optional[int] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        """Process input through AttnRes layers with KV caching support."""
        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")
        if store_cache and train_size is None:
            raise ValueError("train_size must be provided when store_cache=True")

        blocks: List[Tensor] = []
        partial = src
        if self.seed_initial_block:
            blocks.append(partial)

        for layer_idx, layer in enumerate(self.layers, start=1):
            ar_i = self._ar_idx_map.get(layer_idx)
            proj = self.attn_res_projs[ar_i] if ar_i is not None else None
            norm = self.attn_res_norms[ar_i] if ar_i is not None else None
            if use_cache:
                blocks, partial = layer(
                    blocks=blocks,
                    partial_block=partial,
                    layer_number=layer_idx,
                    block_size=self.block_size,
                    rope=self.rope,
                    cached_kv=icl_cache.kv[layer_idx - 1],
                    attn_res_proj=proj,
                    attn_res_norm=norm,
                )
            else:
                blocks, partial, k_proj, v_proj = layer(
                    blocks=blocks,
                    partial_block=partial,
                    layer_number=layer_idx,
                    block_size=self.block_size,
                    train_size=train_size,
                    rope=self.rope,
                    need_kv=True,
                    attn_res_proj=proj,
                    attn_res_norm=norm,
                )
                icl_cache.kv[layer_idx - 1] = KVCacheEntry(key=k_proj, value=v_proj)

        return partial


# ---------------------------------------------------------------------------
# RowInteraction wrapper
# ---------------------------------------------------------------------------

class RowInteractionAttnResLightRMSNorm(nn.Module):
    """Row interaction module using AttnResEncoderLightRMSNorm.

    Matches the original RowInteraction aggregation strategy:
    - First N-1 layers: full-sequence self-attention (AttnRes version)
    - Last layer: CLS tokens as query, full sequence as key/value
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_cls: int = 4,
        rope_base: float = 100000,
        rope_interleaved: bool = True,
        dropout: float = 0.0,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
        norm_first: bool = True,
        block_size: int = 4,
        attnres_stride: int = 2,
        recompute: bool = False,
    ) -> None:
        super().__init__()
        self.num_cls = num_cls
        self.embed_dim = embed_dim
        self.norm_first = norm_first
        self.cls_tokens = nn.Parameter(torch.empty(num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        assert num_blocks >= 1, "num_blocks must be >=1"
        self.encoder_prefix = AttnResEncoderLightRMSNorm(
            num_blocks=max(0, num_blocks - 1),
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            block_size=block_size,
            use_rope=True,
            rope_base=rope_base,
            rope_interleaved=rope_interleaved,
            ssmax=False,
            seed_initial_block=True,
            attnres_stride=attnres_stride,
            recompute=recompute,
        )
        # Last layer: attention block with RMSNorm (CLS as query, full seq as k/v)
        self.final_block = MultiheadAttentionBlockRMSNorm(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        self.rope = self.encoder_prefix.rope
        self.out_ln = RMSNorm(embed_dim) if norm_first else nn.Identity()
        self.inference_mgr = InferenceManager(enc_name="tf_row", out_dim=embed_dim * num_cls, out_no_seq=True)

    def _aggregate_embeddings(self, embeddings: Tensor, key_mask: Optional[Tensor] = None) -> Tensor:
        """Core computation: encoder prefix + final CLS-query block."""
        hidden = self.encoder_prefix(embeddings, key_padding_mask=key_mask)
        cls_out = self.final_block(
            q=hidden[..., : self.num_cls, :],
            k=hidden, v=hidden,
            key_padding_mask=key_mask, rope=self.rope,
        )
        cls_out = self.out_ln(cls_out)
        return cls_out.flatten(-2)

    def _prepare_embeddings(self, embeddings: Tensor, d: Optional[Tensor] = None):
        """Prepend CLS tokens and build key_mask. Shared by train/inference."""
        B, T, HC, E = embeddings.shape
        cls_tokens = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim)
        embeddings = embeddings.clone()
        embeddings[:, :, : self.num_cls] = cls_tokens.to(embeddings.device)

        key_mask = None
        if d is not None:
            d = d + self.num_cls
            idx = torch.arange(HC, device=embeddings.device).view(1, 1, HC).expand(B, T, HC)
            key_mask = idx >= d.view(B, 1, 1)
        return embeddings, key_mask

    def forward(self, embeddings: Tensor, d: Optional[Tensor] = None, mgr_config: Optional[MgrConfig] = None, **kwargs) -> Tensor:
        embeddings, key_mask = self._prepare_embeddings(embeddings, d)

        if self.training:
            return self._aggregate_embeddings(embeddings, key_mask)

        # Inference: use InferenceManager for AMP / auto-batching
        if mgr_config is None:
            mgr_config = InferenceConfig().ROW_CONFIG
        self.inference_mgr.configure(**mgr_config)
        return self.inference_mgr(
            self._aggregate_embeddings, inputs=OrderedDict([("embeddings", embeddings)]),
        )


# ---------------------------------------------------------------------------
# ICLearning wrapper
# ---------------------------------------------------------------------------

class ICLearningAttnResLightRMSNorm(ICLearning):
    """ICL module using AttnResEncoderLightRMSNorm.

    Inherits hierarchical-classification support (``_grouping``,
    ``_fit_hierarchical``, ``_predict_hierarchical``, ``_predict_standard``,
    ``_inference_forward`` and the ``forward`` dispatcher) from
    :class:`ICLearning`. Those methods hold no parameters — they reuse this
    module's ``tf_icl`` / ``ln`` / ``y_encoder`` / ``decoder`` and its
    ``_icl_predictions`` — so inheriting them adds no state and does not change
    ``state_dict``. Only the AttnRes-specific ``__init__`` and the
    representation-/KV-cache forward paths are overridden below.
    """

    def __init__(
        self,
        max_classes: int,
        out_dim: int,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
        norm_first: bool = True,
        ssmax: Union[bool, str] = False,
        block_size: int = 4,
        attnres_stride: int = 2,
        recompute: bool = False,
    ) -> None:
        # Skip ICLearning.__init__ (it would build a standard Encoder + LayerNorm);
        # this subclass constructs the AttnRes/RMSNorm submodules itself and only
        # inherits ICLearning's parameter-free hierarchical/dispatch methods.
        nn.Module.__init__(self)
        from .learning import OneHotAndLinear
        from .inference import InferenceManager

        self.max_classes = max_classes
        self.norm_first = norm_first

        self.tf_icl = AttnResEncoderLightRMSNorm(
            num_blocks=num_blocks,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            block_size=block_size,
            use_rope=False,
            ssmax=ssmax,
            seed_initial_block=True,
            attnres_stride=attnres_stride,
            recompute=recompute,
        )
        if self.norm_first:
            self.ln = RMSNorm(d_model)

        if max_classes > 0:
            self.y_encoder = OneHotAndLinear(max_classes, d_model)
        else:
            self.y_encoder = nn.Linear(1, d_model)

        self.decoder = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, out_dim))
        self.inference_mgr = InferenceManager(enc_name="tf_icl", out_dim=out_dim)

    def _icl_predictions(self, R: Tensor, y_train: Tensor) -> Tensor:
        train_size = y_train.shape[1]
        if self.max_classes > 0:
            Ry_train = self.y_encoder(y_train.float())
        else:
            Ry_train = self.y_encoder(y_train.unsqueeze(-1))
        R[:, :train_size] = R[:, :train_size] + Ry_train

        src = self.tf_icl(R, train_size=train_size)
        if self.norm_first:
            src = self.ln(src)
        out = self.decoder(src)
        return out

    # ``forward``, ``_predict_standard``, ``_inference_forward`` and the whole
    # hierarchical-classification tree (``_grouping`` / ``_fit_hierarchical`` /
    # ``_predict_hierarchical`` / ``_label_encoding``) are inherited unchanged
    # from ``ICLearning`` — they are parameter-free and only depend on
    # ``self.max_classes``, ``self.inference_mgr`` and ``self._icl_predictions``.
    # This is what enables inference with more than ``max_classes`` classes.

    # ------------------------------------------------------------------
    # Representation cache methods
    # ------------------------------------------------------------------

    def prepare_repr_cache(self, R: Tensor, y_train: Tensor) -> Tensor:
        """Add target embedding to train representations for caching."""
        train_size = y_train.shape[1]
        if self.max_classes > 0:
            Ry_train = self.y_encoder(y_train.float())
        else:
            Ry_train = self.y_encoder(y_train.unsqueeze(-1))
        R[:, :train_size] = R[:, :train_size] + Ry_train
        return R

    def _icl_predictions_repr_cache(self, R: Tensor, train_size: int) -> Tensor:
        """ICL predictions with representation cache (y_train already baked in)."""
        src = self.tf_icl(R, train_size=train_size)
        if self.norm_first:
            src = self.ln(src)
        out = self.decoder(src)
        return out

    def forward_with_repr_cache(
        self,
        R: Tensor,
        train_size: int,
        num_classes: Optional[int] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config=None,
    ) -> Tensor:
        """ICL with representation cache (y_train already baked into R)."""
        if mgr_config is None:
            from .inference_config import InferenceConfig
            mgr_config = InferenceConfig().ICL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        out = self.inference_mgr(
            self._icl_predictions_repr_cache,
            inputs=OrderedDict([("R", R), ("train_size", train_size)]),
        )

        out = out[:, train_size:]
        if self.max_classes > 0:
            assert num_classes is not None, "num_classes must be provided for classification"
            out = out[..., :num_classes]
            if not return_logits:
                out = torch.softmax(out / softmax_temperature, dim=-1)
        return out

    # ------------------------------------------------------------------
    # KV cache methods
    # ------------------------------------------------------------------

    def _icl_predictions_with_cache(
        self,
        R: Tensor,
        icl_cache: KVCache,
        y_train: Optional[Tensor] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        """ICL predictions with KV caching."""
        if store_cache:
            assert y_train is not None, "y_train must be provided when store_cache=True"
            train_size = y_train.shape[1]
            if self.max_classes > 0:
                Ry_train = self.y_encoder(y_train.float())
            else:
                Ry_train = self.y_encoder(y_train.unsqueeze(-1))
            R[:, :train_size] = R[:, :train_size] + Ry_train

        src = self.tf_icl.forward_with_cache(
            R,
            icl_cache=icl_cache,
            train_size=train_size if store_cache else None,
            use_cache=use_cache,
            store_cache=store_cache,
        )
        if self.norm_first:
            src = self.ln(src)
        out = self.decoder(src)
        return out

    def forward_with_cache(
        self,
        R: Tensor,
        icl_cache: KVCache,
        y_train: Optional[Tensor] = None,
        num_classes: Optional[int] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        use_cache: bool = False,
        store_cache: bool = True,
        mgr_config=None,
    ) -> Tensor:
        """ICL with KV caching support."""
        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")

        if store_cache:
            assert y_train is not None, "y_train must be provided when store_cache=True"
            if self.max_classes > 0:
                num_classes = len(torch.unique(y_train[0]))
                if num_classes > self.max_classes:
                    raise ValueError(
                        f"KV caching is not supported for classification with more classes "
                        f"({num_classes}) than max_classes ({self.max_classes})."
                    )
        else:
            assert num_classes is not None, "num_classes must be provided when use_cache=True"

        if mgr_config is None:
            from .inference_config import InferenceConfig
            mgr_config = InferenceConfig().ICL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        out = self.inference_mgr(
            self._icl_predictions_with_cache,
            inputs=OrderedDict([
                ("R", R),
                ("icl_cache", icl_cache),
                ("y_train", y_train),
                ("use_cache", use_cache),
                ("store_cache", store_cache),
            ]),
        )

        if store_cache:
            train_size = y_train.shape[1]
            out = out[:, train_size:]

        if self.max_classes > 0:
            out = out[..., :num_classes]
            if not return_logits:
                out = torch.softmax(out / softmax_temperature, dim=-1)
        return out


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class TabLDMAttnResLightRMSNorm(TabLDM):
    """TabLDM variant using speed-optimized Block AttnRes layers with RMSNorm.

    Identical to TabLDMAttnResLight but replaces all LayerNorm with RMSNorm.
    """

    def __init__(self, *args, block_size: int = 4, attnres_stride: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_cls = TabLDMAttnCache

        # recompute is consumed by the base TabLDM.__init__ (via **kwargs) for the
        # column encoder; read it here (without removing it) so the AttnRes row/ICL
        # encoders, which the base does not build, also honor gradient checkpointing.
        recompute = kwargs.get("recompute", False)

        self.row_interactor = RowInteractionAttnResLightRMSNorm(
            embed_dim=self.embed_dim,
            num_blocks=self.row_num_blocks,
            nhead=self.row_nhead,
            dim_feedforward=self.embed_dim * self.ff_factor,
            num_cls=self.row_num_cls,
            rope_base=self.row_rope_base,
            rope_interleaved=self.row_rope_interleaved,
            dropout=self.dropout,
            activation=self.activation,
            norm_first=self.norm_first,
            block_size=block_size,
            attnres_stride=attnres_stride,
            recompute=recompute,
        )

        icl_dim = self.embed_dim * self.row_num_cls
        self.icl_predictor = ICLearningAttnResLightRMSNorm(
            out_dim=self.max_classes if self.max_classes > 0 else self.num_quantiles,
            max_classes=self.max_classes,
            d_model=icl_dim,
            num_blocks=self.icl_num_blocks,
            nhead=self.icl_nhead,
            dim_feedforward=icl_dim * self.ff_factor,
            dropout=self.dropout,
            activation=self.activation,
            norm_first=self.norm_first,
            ssmax=self.icl_ssmax,
            block_size=block_size,
            attnres_stride=attnres_stride,
            recompute=recompute,
        )
