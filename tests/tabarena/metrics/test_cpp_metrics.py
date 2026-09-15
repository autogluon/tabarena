from __future__ import annotations

import time

import numpy as np
import pytest

from tabarena.metrics._cpp_metrics import CppMetrics


def test_cpp_metrics_compilation():
    CppMetrics.clean_plugin()
    assert not CppMetrics.plugin_path().exists(), "plugin should have been deleted"

    metrics = CppMetrics()
    assert CppMetrics.plugin_path().exists(), "plugin should have been compiled automatically"

    n_samples = 32
    assert np.isclose(
        metrics.roc_auc_score(
            y_true=np.array([i % 2 == 0 for i in range(n_samples)]),
            y_score=np.arange(n_samples) / n_samples + 1,
        ),
        0.5,
    )
    assert np.isclose(
        metrics.rmse(
            y_true=np.zeros(4),
            y_pred=np.array([1.0, -1.0, 1.0, -1.0]),
        ),
        1.0,
    )


def _compile_and_score(queue) -> None:
    from tabarena.metrics._cpp_metrics import CppMetrics

    try:
        metrics = CppMetrics()
        n_samples = 32
        score = metrics.roc_auc_score(
            y_true=np.array([i % 2 == 0 for i in range(n_samples)]),
            y_score=np.arange(n_samples) / n_samples + 1,
        )
        queue.put(("ok", score))
    except Exception as e:
        queue.put(("error", f"{type(e).__name__}: {e}"))


def test_cpp_metrics_concurrent_compilation(tmp_path, monkeypatch):
    """A process must never load a half-written ``cpp_metrics.so`` from another process's compile.

    Concurrent first-use compiles happen in practice (ray workers on a fresh checkout). A real
    linker writes the 16 KB output too quickly to hit the window reliably here, so the compile
    is routed through a slow ``g++`` wrapper that leaves junk at the output path for a while
    before running the real compiler. A second process constructing ``CppMetrics`` inside that
    window used to ``dlopen`` the junk and fail with "invalid ELF header".
    """
    import multiprocessing
    import os
    import shutil
    import textwrap

    real_gxx = shutil.which("g++")
    if real_gxx is None:
        pytest.skip("g++ not available")
    fake_gxx = tmp_path / "g++"
    fake_gxx.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/sh
            # slow linker: expose a half-written output for a while before finishing
            out=""; prev=""
            for a in "$@"; do if [ "$prev" = "-o" ]; then out="$a"; fi; prev="$a"; done
            printf 'not an ELF file' > "$out"
            sleep 1.5
            exec {real_gxx} "$@"
            """
        )
    )
    fake_gxx.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    assert shutil.which("g++") == str(fake_gxx)

    CppMetrics.clean_plugin()
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    other = ctx.Process(target=_compile_and_score, args=(queue,))
    other.start()
    # wait until the other process's (slow) compile is underway, then compile here too
    deadline = time.time() + 30
    while time.time() < deadline and not list(CppMetrics.plugin_path().parent.glob("cpp_metrics.so*")):
        time.sleep(0.05)
    metrics = CppMetrics()
    assert np.isclose(metrics.rmse(y_true=np.zeros(2), y_pred=np.ones(2)), 1.0)

    status, payload = queue.get(timeout=120)
    other.join(timeout=120)
    assert status == "ok", payload
    assert np.isclose(payload, 0.5)
    leftovers = list(CppMetrics.plugin_path().parent.glob("cpp_metrics.so.tmp-*"))
    assert not leftovers, leftovers
