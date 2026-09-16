"""Dry runs of ``submit_template.sh`` with a fake interpreter (no SLURM, no Ray, no model fits).

The fake python records its argv and a few environment values per item and fails for one dataset,
which exercises the continue-on-failure loop, the scratch layout and hygiene, and the optional
weight staging block. Skipped when bash, jq or rsync are missing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("tabflow_slurm.setup", reason="tabflow_slurm is not installed")

from tabflow_slurm.setup.paths import get_submit_script_path

_MISSING = [tool for tool in ("bash", "jq", "rsync") if shutil.which(tool) is None]
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=f"required tools missing: {_MISSING}")

_FAKE_PYTHON = r"""#!/bin/bash
# Records what the harness handed us; fails for dataset ds_fail.
echo "$0 $*" >> "$FAKE_LOG"
{
    echo "TMPDIR=$TMPDIR"
    echo "PWD=$PWD"
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
    echo "AG_BASE=${TABARENA_MODEL_ARTIFACTS_BASE_PATH:-unset}"
    echo "HF_HOME=${HF_HOME:-unset}"
    echo "HF_TOKEN_PATH=${HF_TOKEN_PATH:-unset}"
    echo "TABPFN_DIR=${TABPFN_MODEL_CACHE_DIR:-unset}"
    if [ -n "${HF_HOME:-}" ]; then
        echo "STAGED_BLOB=$(readlink -e "$HF_HOME/hub/models--a--b/snapshots/rev/model.bin" || echo missing)"
        echo "OVERLAY_REPO=$(readlink -e "$HF_HOME/hub/models--c--d" || echo missing)"
    fi
    if [ -n "${TABPFN_MODEL_CACHE_DIR:-}" ]; then
        echo "OVERLAY_CKPT=$(readlink -e "$TABPFN_MODEL_CACHE_DIR/other.ckpt" || echo missing)"
    fi
} >> "$FAKE_ENV_LOG"
for arg in "$@"; do
    if [ "$prev" = "--dataset" ] && [ "$arg" = "ds_fail" ]; then exit 1; fi
    prev="$arg"
done
exit 0
"""


def _write_fake_interpreter(tmp_path: Path) -> Path:
    real = tmp_path / "fake_python.sh"
    real.write_text(_FAKE_PYTHON)
    real.chmod(0o755)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    link = venv_bin / "python"
    link.symlink_to(real)  # the venv symlink the job JSON records
    return link


def _run_template(
    tmp_path: Path, items: list[dict], *, staging: dict | None = None
) -> tuple[subprocess.CompletedProcess, list[str], str]:
    python = _write_fake_interpreter(tmp_path)
    run_script = tmp_path / "run.py"
    run_script.write_text("")
    defaults = {
        "python": str(python),
        "run_script": str(run_script),
        "job_batch_dir": str(tmp_path / "batch"),
        "output_dir": str(tmp_path / "out"),
        "num_cpus": 2,
        "num_gpus": 0,
        "memory_limit": 4,
        "setup_ray_for_slurm_shared_resources_environment": False,
        "ignore_cache": False,
        "offline_weights": False,
        "slurm_log_dir": str(tmp_path / "slurm_out"),
    }
    if staging is not None:
        defaults["staging"] = staging
    job_json = tmp_path / "job.json"
    job_json.write_text(json.dumps({"defaults": defaults, "jobs": [{"items": items}]}))
    fake_log = tmp_path / "argv.log"
    env_log = tmp_path / "env.log"
    scratch_root = tmp_path / "tmp"
    env = {
        **os.environ,
        "SLURM_JOB_ID": "42",
        "SLURM_ARRAY_JOB_ID": "42",
        "SLURM_ARRAY_TASK_ID": "0",
        "TMPDIR": str(scratch_root),
        "OMP_NUM_THREADS": "3",
        "FAKE_LOG": str(fake_log),
        "FAKE_ENV_LOG": str(env_log),
    }
    env.pop("TABARENA_JIT_ROOT", None)
    proc = subprocess.run(  # noqa: S603
        [shutil.which("bash"), str(get_submit_script_path()), str(job_json)],
        check=False,
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    argv_lines = fake_log.read_text().splitlines() if fake_log.exists() else []
    env_text = env_log.read_text() if env_log.exists() else ""
    assert not (scratch_root / "tj_42").exists(), "the EXIT trap must remove the job scratch"
    return proc, argv_lines, env_text


def _item(dataset: str, fold: int = 0) -> dict:
    return {"experiment": "exp", "dataset": dataset, "fold": fold, "repeat": 0}


def test_bash_syntax():
    assert subprocess.run([shutil.which("bash"), "-n", str(get_submit_script_path())], check=False).returncode == 0  # noqa: S603


def test_dry_run_continues_after_a_failing_item(tmp_path):
    items = [_item("ds_1"), _item("ds_fail"), _item("ds_3"), _item("ds_4")]
    proc, argv_lines, env_text = _run_template(tmp_path, items)

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert [line.split("--dataset ")[1].split(" ")[0] for line in argv_lines] == ["ds_1", "ds_fail", "ds_3", "ds_4"]
    assert proc.stdout.count("##### item FAILED (exit 1)") == 1
    assert "dataset=ds_fail" in proc.stdout
    assert "##### bundle summary: ok=3 failed=1 total=4" in proc.stdout

    first = argv_lines[0].split(" ")
    assert first[0] == str(tmp_path / "venv" / "bin" / "python")  # invoked through the venv symlink, not readlink'd
    assert first[1] == "-P"
    assert "--offline_weights false" in argv_lines[0]
    assert f"--ray_temp_root {tmp_path / 'tmp' / 'tj_42' / 'ray'}" in argv_lines[0]

    scratch = tmp_path / "tmp" / "tj_42"
    assert f"TMPDIR={scratch / 'tmp'}" in env_text
    assert f"PWD={scratch}" in env_text
    assert f"AG_BASE={scratch / 'ag'}" in env_text
    assert "OMP_NUM_THREADS=unset" in env_text
    assert "hygiene: unsetting OMP_NUM_THREADS=3" in proc.stdout
    assert "HF_HOME=unset" in env_text  # no staging block requested


def test_staging_copies_repo_dirs_and_overlays_the_rest_and_fails_closed(tmp_path):
    hub = tmp_path / "hf" / "hub"
    (hub / "models--a--b" / "blobs").mkdir(parents=True)
    (hub / "models--a--b" / "blobs" / "sha1").write_bytes(b"weights")
    snap = hub / "models--a--b" / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "model.bin").symlink_to(Path("../../blobs/sha1"))
    (hub / "models--c--d").mkdir()  # not staged, overlaid as a symlink
    (tmp_path / "hf" / "token").write_text("secret-token")
    tabpfn = tmp_path / "tabpfn"
    tabpfn.mkdir()
    (tabpfn / "clf.ckpt").write_bytes(b"c")
    (tabpfn / "other.ckpt").write_bytes(b"o")
    staging = {
        "stage_weights": True,
        "hf_hub_cache_src": str(hub),
        "hf_home_src": str(tmp_path / "hf"),
        "hf_repo_dirs": [str(hub / "models--a--b")],
        "tabpfn_cache_dir_src": str(tabpfn),
        "tabpfn_files": [str(tabpfn / "clf.ckpt")],
        "reserve_bytes": 0,
        "pretouch_libs": True,
        "pretouch_packages": ["ray"],
        "pretouch_paths": [],
        "pretouch_max_bytes": 1,
    }
    proc, argv_lines, env_text = _run_template(tmp_path, [_item("ds_1")], staging=staging)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    scratch = tmp_path / "tmp" / "tj_42"
    assert f"HF_HOME={scratch / 'stage' / 'huggingface'}" in env_text
    assert f"HF_TOKEN_PATH={tmp_path / 'hf' / 'token'}" in env_text
    assert f"STAGED_BLOB={scratch / 'stage' / 'huggingface' / 'hub' / 'models--a--b' / 'blobs' / 'sha1'}" in env_text
    assert f"OVERLAY_REPO={hub / 'models--c--d'}" in env_text
    assert f"TABPFN_DIR={scratch / 'stage' / 'tabpfn'}" in env_text
    assert f"OVERLAY_CKPT={tabpfn / 'other.ckpt'}" in env_text
    assert any("-m tabflow_slurm.node_prep pretouch" in line for line in argv_lines)
    assert "secret-token" not in proc.stdout

    # A missing source fails closed: no cache variable is exported and the item still runs.
    shutil.rmtree(tmp_path / "tmp", ignore_errors=True)
    for name in ("argv.log", "env.log", "job.json", "fake_python.sh"):
        (tmp_path / name).unlink(missing_ok=True)
    shutil.rmtree(tmp_path / "venv")
    staging["hf_repo_dirs"] = [str(hub / "models--missing")]
    proc, _, env_text = _run_template(tmp_path, [_item("ds_1")], staging=staging)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "staging: source missing" in proc.stdout
    assert "HF_HOME=unset" in env_text
    assert "TABPFN_DIR=unset" in env_text
