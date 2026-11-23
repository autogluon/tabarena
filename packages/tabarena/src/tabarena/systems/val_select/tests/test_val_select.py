from __future__ import annotations

from tabarena.systems.val_select.system import (
    LIMIX2_MAX_CELLS,
    LOW_VRAM_LIMIX_MAX_CELLS,
    ValSelectSystemModel,
    _cap_for_vram_gib,
)


def test_cap_for_vram_small_gpu_pinned():
    # 24 GB (RTX 4090) and 16 GB cards sit below the cutoff and get the fast-path cap.
    assert _cap_for_vram_gib(24.0) == LOW_VRAM_LIMIX_MAX_CELLS
    assert _cap_for_vram_gib(15.9) == LOW_VRAM_LIMIX_MAX_CELLS


def test_cap_for_vram_large_gpu_official():
    # A100-40 / H100-80 keep the wrapper's official cap; the cutoff itself is not "small".
    assert _cap_for_vram_gib(32.0) == LIMIX2_MAX_CELLS
    assert _cap_for_vram_gib(79.1) == LIMIX2_MAX_CELLS


def test_cap_for_vram_no_cuda_official():
    assert _cap_for_vram_gib(None) == LIMIX2_MAX_CELLS


def test_pinned_cap_not_overridden_by_device():
    model = ValSelectSystemModel(
        problem_type="regression",
        eval_metric=None,
        limix_max_cells=5_000_000,
    )
    assert model._limix_cap == 5_000_000
    _, _, refit_hps, _ = model._candidates()[0]
    assert refit_hps == {"max_cells": 5_000_000}


def test_auto_cap_default_resolves_at_fit():
    model = ValSelectSystemModel(problem_type="regression", eval_metric=None)
    assert model.limix_max_cells is None
