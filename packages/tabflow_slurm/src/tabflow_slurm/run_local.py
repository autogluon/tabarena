"""Sequential, local runner for a generated benchmark job JSON.

The non-cluster counterpart to ``submit_template.sh``. Where the SLURM submit
script runs one array task (``jobs[SLURM_ARRAY_TASK_ID]``) and loops over that
task's bundled items, this runner flattens *all* jobs and *all* items into a
single sequential loop and runs ``run_tabarena_experiment``'s per-item logic
once per item (one (experiment, dataset, fold, repeat) work unit each).

Two execution modes (``--execution_mode``):
    - ``subprocess`` (default): each item runs in its own fresh subprocess via
      ``run_tabarena_experiment.py`` — so every model fit stays isolated (fresh
      memory / Ray / GPU context), exactly like an independent SLURM array task.
    - ``in_process``: each item runs in this runner's own Python process by
      calling ``run_experiment`` directly. Faster and debugger friendly, but
      fits share global state and a hard crash aborts the whole run.

Invoke it via the command emitted by ``LocalSequentialSetup.get_run_commands``:

    <python> -m tabflow_slurm.run_local <job.json> [--continue_on_error True]
                                                   [--execution_mode in_process]

The job JSON has the same ``{"defaults": {...}, "jobs": [{"items": [...]}, ...]}``
shape produced by ``TabArenaBenchmarkSetup.get_jobs_dict`` — every runtime arg the
per-item runner needs lives in ``defaults``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _str2bool(v: str | bool) -> bool:
    """Parse a CLI boolean (mirrors ``run_tabarena_experiment._str2bool``)."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _build_item_command(defaults: dict, item: dict) -> list[str]:
    """Build the ``run_tabarena_experiment.py`` argv for a single item.

    Mirrors ``submit_template.sh::run_one``. Every value is stringified:
    ``str(True/False)`` -> ``"True"/"False"`` (accepted by the runner's
    ``_str2bool``) and ``str(None)`` -> ``"None"`` (accepted by
    ``_parse_int_or_none`` for ``num_cpus``/``memory_limit``).
    """
    return [
        str(defaults["python"]),
        str(defaults["run_script"]),
        "--job_batch_dir",
        str(defaults["job_batch_dir"]),
        "--experiment",
        str(item["experiment"]),
        "--dataset",
        str(item["dataset"]),
        "--fold",
        str(item["fold"]),
        "--repeat",
        str(item["repeat"]),
        "--output_dir",
        str(defaults["output_dir"]),
        "--num_cpus",
        str(defaults["num_cpus"]),
        "--num_gpus",
        str(defaults["num_gpus"]),
        "--memory_limit",
        str(defaults["memory_limit"]),
        # Local runs never use the SLURM shared-filesystem Ray setup.
        "--setup_ray_for_slurm_shared_resources_environment",
        "False",
        "--ignore_cache",
        str(defaults["ignore_cache"]),
    ]


def _run_item_subprocess(defaults: dict, item: dict, env: dict) -> int:
    """Run one item in its own subprocess; return its exit code (0 == success)."""
    return subprocess.run(_build_item_command(defaults, item), env=env, check=False).returncode  # noqa: S603


def _setup_in_process(defaults: dict) -> None:
    """One-time setup for `in_process` mode: Ray + telemetry env.

    Subprocess mode does this per child via ``run_tabarena_experiment``'s ``__main__``; in-process
    we must do it once before the first fit. We reuse ``setup_slurm_job`` with the SLURM
    shared-resources Ray setup disabled. Cache configuration is not done here — each item's
    ``run_experiment`` applies the ``JobBatch``'s ``cache_config`` before its fit.
    """
    from tabflow_slurm.slurm_utils import setup_slurm_job

    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    setup_slurm_job(
        num_cpus=defaults["num_cpus"],
        num_gpus=defaults["num_gpus"],
        memory_limit=defaults["memory_limit"],
        setup_ray_for_slurm_shared_resources_environment=False,
    )


def _run_item_in_process(defaults: dict, item: dict) -> int:
    """Run one item in this process via `run_experiment`; return 0 on success, 1 on error.

    Only catches Python exceptions — a hard crash (segfault / OOM kill) still
    takes down the whole runner, which is the core trade-off of in-process mode.
    """
    from tabflow_slurm.run_tabarena_experiment import run_experiment

    try:
        run_experiment(
            job_batch_dir=str(defaults["job_batch_dir"]),
            experiment_name=str(item["experiment"]),
            dataset=str(item["dataset"]),
            fold=item["fold"],
            repeat=item["repeat"],
            output_dir=str(defaults["output_dir"]),
            ignore_cache=bool(defaults["ignore_cache"]),
        )
    except Exception as exc:
        print(f"  in-process item raised: {exc!r}", flush=True)
        return 1
    return 0


def run(json_path: str, *, continue_on_error: bool, execution_mode: str = "subprocess") -> int:
    """Run every item in `json_path` sequentially; return 0 iff all succeeded.

    `execution_mode` is "subprocess" (one fresh process per item, isolated) or
    "in_process" (run every item in this process; faster but no isolation).
    """
    with Path(json_path).open() as f:
        jobs_dict = json.load(f)

    defaults = jobs_dict["defaults"]
    items = [item for job in jobs_dict["jobs"] for item in job["items"]]
    total = len(items)
    print(f"Running {total} item(s) sequentially from {json_path} (mode={execution_mode})", flush=True)

    # Subprocess mode: match the env the SLURM submit template exports to each job.
    env = os.environ.copy()
    env["TABPFN_DISABLE_TELEMETRY"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    if execution_mode == "in_process":
        _setup_in_process(defaults)

    failures: list[tuple[int, dict, int]] = []
    completed = 0
    for idx, item in enumerate(items, start=1):
        print(
            f"\n===== [{idx}/{total}] experiment={item['experiment']} dataset={item['dataset']} "
            f"fold={item['fold']} repeat={item['repeat']} =====",
            flush=True,
        )
        if execution_mode == "in_process":
            code = _run_item_in_process(defaults, item)
        else:
            code = _run_item_subprocess(defaults, item, env)
        completed = idx
        if code != 0:
            print(
                f"##### Item [{idx}/{total}] FAILED (exit {code}): "
                f"experiment={item['experiment']} dataset={item['dataset']} "
                f"fold={item['fold']} repeat={item['repeat']}",
                flush=True,
            )
            failures.append((idx, item, code))
            if not continue_on_error:
                print("Stopping (continue_on_error=False).", flush=True)
                break

    succeeded = completed - len(failures)
    print(
        f"\n##### Local run summary: {succeeded}/{total} succeeded, {len(failures)} failed.",
        flush=True,
    )
    for idx, item, code in failures:
        print(
            f"  - [{idx}/{total}] exit {code}: experiment={item['experiment']} dataset={item['dataset']} "
            f"fold={item['fold']} repeat={item['repeat']}",
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a benchmark job JSON locally and sequentially.")
    parser.add_argument("json_path", type=str, help="Path to the generated job JSON file.")
    parser.add_argument(
        "--continue_on_error",
        type=_str2bool,
        default=False,
        help="If True, keep running after a failing item instead of stopping at the first failure.",
    )
    parser.add_argument(
        "--execution_mode",
        choices=["subprocess", "in_process"],
        default="subprocess",
        help="'subprocess' (default): one isolated process per item. "
        "'in_process': run every item in this process (faster, no isolation).",
    )
    args = parser.parse_args()
    sys.exit(run(args.json_path, continue_on_error=args.continue_on_error, execution_mode=args.execution_mode))
