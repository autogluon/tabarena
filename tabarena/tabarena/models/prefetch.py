"""Standardized pre-fetching of foundation-model weights.

Benchmarking foundation models requires their (often large, Hugging-Face-hosted) weights to be
present locally. Downloading lazily inside ``_fit`` is fine for a single run, but for a benchmark
launched across many parallel compute nodes — some without internet — the weights must be warmed
on the head node *before* the jobs are dispatched.

Each model declares *how* to fetch its own weights via ``ModelInfo.prefetch_weights`` — a zero-arg
``prefetch_weights`` callable living in the model's ``model.py`` (or, for models supported out of the
box via an external class, its ``info.py``). This module is just the thin consumer:
:func:`prefetch_weights` resolves benchmark model names to their ``ModelInfo`` and calls each one's
prefetcher. Models that declare nothing (tree / linear baselines) are skipped.
"""

from __future__ import annotations

from collections.abc import Iterable


def prefetch_weights(model_names: Iterable[str], *, raise_on_error: bool = False) -> None:
    """Ensure the weights of every foundation model in ``model_names`` are present locally.

    Resolves each name to its registry ``ModelInfo`` and calls ``info.prefetch_weights()`` if the
    model declares one, de-duplicating models that share a prefetcher (e.g. TabPFN variants) so each
    runs once. Models that declare no prefetcher are skipped.

    Args:
        model_names: Benchmark model names (``display_name`` or ``method``), e.g. ``["TabPFN-3"]``.
        raise_on_error: If True, re-raise the first error; otherwise log and continue (a missing
            optional dependency or a single failed download won't abort the benchmark).
    """
    from tabarena.models.utils import get_model_info_from_name

    seen: set = set()
    for model_name in model_names:
        try:
            info = get_model_info_from_name(model_name)
        except ValueError as exc:  # unknown model name
            if raise_on_error:
                raise
            print(f"[prefetch] {model_name}: skipped (could not resolve model: {exc})")
            continue

        prefetcher = info.prefetch_weights
        if prefetcher is None:
            print(f"[prefetch] {model_name}: nothing to prefetch (not a foundation model)")
            continue
        if prefetcher in seen:
            print(f"[prefetch] {model_name}: weights already warmed by another variant")
            continue
        seen.add(prefetcher)

        print(f"[prefetch] {model_name} ({info.method_metadata.method}): ensuring weights present...")
        try:
            prefetcher()
        except ImportError as exc:
            print(f"[prefetch] {model_name}: skipped (optional dependency missing: {exc})")
        except Exception as exc:  # one bad download shouldn't abort the whole plan
            if raise_on_error:
                raise
            print(f"[prefetch] {model_name}: FAILED ({type(exc).__name__}: {exc})")
