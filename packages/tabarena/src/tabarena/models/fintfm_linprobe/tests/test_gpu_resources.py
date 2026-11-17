"""FinTFM asks AutoGluon for one GPU when CUDA is present and for none otherwise.

``_fit`` moves the network to CUDA only when ``num_gpus`` > 0, so a zero-GPU default left a CUDA
machine fitting on its CPU (2026-11-18). These tests patch ``torch.cuda.is_available`` and need no
GPU and no checkpoint.
"""

from __future__ import annotations

import pytest

from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel


def _model(tmp_path) -> FinTFMLinearProbeModel:
    """A binary FinTFM-LinProbe model with default hyperparameters."""
    return FinTFMLinearProbeModel(
        path=str(tmp_path), name="fintfm_gpu_test", problem_type="binary", eval_metric="roc_auc"
    )


@pytest.mark.parametrize(("cuda", "expected"), [(True, 1), (False, 0)])
def test_default_gpus_follow_cuda(tmp_path, monkeypatch, cuda: bool, expected: int) -> None:
    """One GPU with CUDA, zero without; the CPU count is the parent's in both cases."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    num_cpus, num_gpus = _model(tmp_path)._get_default_resources()
    assert num_gpus == expected
    assert isinstance(num_cpus, int) and num_cpus >= 1


def test_no_gpu_minimum(tmp_path) -> None:
    """A machine without CUDA can still fit: the model sets no minimum GPU count."""
    assert getattr(_model(tmp_path), "minimum_num_gpus", 0) == 0
