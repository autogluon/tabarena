"""Every registered model fits and predicts warm (marker ``models``, deselected by default).

One fresh interpreter per model runs ``python -P -m tabarena.tools.audit_warmup`` with the model's
smoke hyperparameters and asserts that the warm-up finished with every step ok and that neither
timed section imported a new third-party top-level package or created the CUDA context.
``ModelSmokeTest.allowed_lazy_imports`` lists the justified exceptions. The audit sees the main
process only: parallel fold workers, code vendored under ``tabarena.models`` and AutoGluon
submodules are invisible to the diff (``timing_audit["scope"]``). Run one model with
``pytest -m models tests/tabarena/models/test_warmup_coverage.py -k TabM``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tabarena.tools.audit_warmup import EXIT_MISSING_DEPENDENCY

from .smoke_configs import registry_or_fail, smoke_for

_REGISTRY = registry_or_fail()


def _cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


_CUDA_AVAILABLE = _cuda_available()


def run_audit_isolated(
    method: str, tmp_path: Path, *extra_args: str, timeout: float = 900
) -> subprocess.CompletedProcess:
    """Run the warm-up audit for one model in a fresh interpreter with a clean ``sys.path``.

    ``-P`` keeps the child's cwd off ``sys.path`` and ``PYTHONSAFEPATH=1`` does the same for the
    plain Python children it starts (loky). Ray puts the driver's cwd on every worker's ``sys.path``
    regardless, so ``cwd=tmp_path`` is what keeps Ray workers (and FitHelper's cwd-relative
    output) away from the repository.
    """
    return subprocess.run(  # noqa: S603
        [sys.executable, "-P", "-m", "tabarena.tools.audit_warmup", "--model", method, *extra_args],
        cwd=tmp_path,
        env={**os.environ, "PYTHONSAFEPATH": "1"},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.mark.models
@pytest.mark.parametrize("method", sorted(_REGISTRY), ids=str)
def test_timed_sections_run_warm(method: str, tmp_path: Path) -> None:
    info = _REGISTRY[method]
    if info.method_metadata.compute == "gpu" and not _CUDA_AVAILABLE:
        pytest.skip(f"{method}: requires a GPU (compute='gpu') and no CUDA device is available")
    if info.superseded:
        pytest.skip(f"{method}: superseded; its pip_extra {info.pip_extra} conflicts with the installed version")

    cfg = smoke_for(method)
    problem_type = (cfg.problem_types or ("binary",))[0]
    json_path = tmp_path / "audit.json"
    proc = run_audit_isolated(
        method,
        tmp_path,
        "--problem-type",
        problem_type,
        "--hyperparameters",
        json.dumps(cfg.hyperparameters),
        "--json",
        str(json_path),
    )
    if proc.returncode == EXIT_MISSING_DEPENDENCY:
        pytest.skip(f"{method}: optional dependency not installed ({proc.stderr.strip().splitlines()[-1]})")
    assert proc.returncode == 0, f"audit failed (exit {proc.returncode})\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"

    data = json.loads(json_path.read_text())
    report = data["warmup_report"]
    assert report["status"] == "ok", f"{report['status']}: {report['failed_steps']} {report['error']}"
    assert not report["failed_steps"]
    for section in ("fit", "predict"):
        audit = data["timing_audit"][section]
        assert audit is not None, f"{section}: environment audit unavailable"
        cold = set(audit["new_packages"]) - set(cfg.allowed_lazy_imports)
        assert not cold, f"{section} imported {sorted(cold)} inside the timer; declare them in warmup_modules"
        assert not (audit["cuda_initialized_before"] is False and audit["cuda_initialized_after"] is True), (
            f"{section} created the CUDA context inside the timer"
        )
