from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tabarena.models.tabswift._vendor.model.learning import ICLearning


def _force_split(mgr, tables_per_batch: int = 1) -> None:
    """Drive the manager's split-and-write-back loop on CPU with ``tables_per_batch`` tables per sub-batch.

    The manager only splits on CUDA, so the device is reported as CUDA while every tensor stays on the CPU,
    the memory estimate is pinned to ``tables_per_batch``, and the output is kept on the CPU through ``offload``.
    """
    configure = mgr.configure_inference

    def configure_on_fake_cuda(device=None, use_amp=True, verbose=False):
        configure(device="cpu", use_amp=False, verbose=verbose)
        mgr.exe_device = torch.device("cuda")

    mgr.configure_inference = configure_on_fake_cuda
    mgr.to_exe_device = lambda tensor: tensor
    mgr.estimate_safe_batch_size = lambda *args, **kwargs: (float("inf"), tables_per_batch)
    mgr.offload = True


N_TABLES, N_ROWS, TRAIN_SIZE, D_MODEL = 3, 9, 6, 16


def _make_icl(register_tokens: int = 64):
    torch.manual_seed(0)
    icl = ICLearning(
        max_classes=10, d_model=D_MODEL, num_blocks=1, nhead=2, dim_feedforward=32, register_tokens=register_tokens
    ).eval()
    for param in icl.parameters():
        torch.nn.init.normal_(param, std=0.02)
    R = torch.randn(N_TABLES, N_ROWS, D_MODEL)
    y_train = torch.randn(N_TABLES, TRAIN_SIZE)
    return icl, R, y_train


@pytest.mark.parametrize("register_tokens", [0, 64])
def test_regression_batched_inference_matches_unbatched(register_tokens: int):
    """Splitting the regression ICL pass over tables gives the single-pass result (issue #618)."""
    icl, R, y_train = _make_icl(register_tokens)
    n_tables, n_rows, train_size = N_TABLES, N_ROWS, TRAIN_SIZE

    expected = icl(R.clone(), y_train, device="cpu", use_amp=False, if_regression=True)
    _force_split(icl.inference_mgr_reg)
    actual = icl(R.clone(), y_train, device="cpu", use_amp=False, if_regression=True)

    assert actual.shape == (n_tables, n_rows - train_size, 1)
    torch.testing.assert_close(actual, expected)


def test_regression_oom_retry_does_not_re_encode_targets():
    """A sub-batch retried after a CUDA OOM sees the caller's ``R`` unchanged and gives the single-pass result."""
    icl, R, y_train = _make_icl()
    expected = icl(R.clone(), y_train, device="cpu", use_amp=False, if_regression=True)

    _force_split(icl.inference_mgr_reg, tables_per_batch=2)
    forward = icl._icl_predictions_reg
    n_calls = 0

    def forward_with_one_oom(*args, **kwargs):
        # The second sub-batch runs out of memory once, after the first one already ran on its view of ``R``.
        nonlocal n_calls
        n_calls += 1
        if n_calls == 2:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return forward(*args, **kwargs)

    icl._icl_predictions_reg = forward_with_one_oom
    R_input = R.clone()
    actual = icl(R_input, y_train, device="cpu", use_amp=False, if_regression=True)

    assert n_calls > 3, "the manager should have retried with a smaller batch size"
    torch.testing.assert_close(R_input, R)
    torch.testing.assert_close(actual, expected)
