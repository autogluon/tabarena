"""Sequential, local runner for a generated benchmark job JSON.

The non-cluster counterpart to ``submit_template.sh``. Where the SLURM submit
script runs one array task (``jobs[SLURM_ARRAY_TASK_ID]``) and loops over that
task's bundled items, this runner flattens *all* jobs and *all* items into a
single sequential loop and invokes ``run_tabarena_experiment.py`` once per item
as its own subprocess — so each model fit stays isolated (fresh memory / Ray /
GPU context), exactly like an independent SLURM array task.

Invoke it via the command emitted by ``LocalSequentialSetup.get_run_commands``:

    <python> -m tabflow_slurm.run_local <job.json> [--continue_on_error True]

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
    ``_str2bool``), ``str(None)`` -> ``"None"`` (accepted by ``_parse_int_or_none``
    for ``num_cpus``/``memory_limit``), and ``config_index`` is a single int
    (parsed into a one-element list by the runner's ``_parse_int_list``).
    """
    return [
        str(defaults["python"]),
        str(defaults["run_script"]),
        "--task_id",
        str(item["task_id"]),
        "--fold",
        str(item["fold"]),
        "--repeat",
        str(item["repeat"]),
        "--config_index",
        str(item["config_index"]),
        "--configs_yaml_file",
        str(defaults["configs_yaml_file"]),
        "--openml_cache_dir",
        str(defaults["openml_cache_dir"]),
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


def run(json_path: str, *, continue_on_error: bool) -> int:
    """Run every item in `json_path` sequentially; return 0 iff all succeeded."""
    with Path(json_path).open() as f:
        jobs_dict = json.load(f)

    defaults = jobs_dict["defaults"]
    items = [item for job in jobs_dict["jobs"] for item in job["items"]]
    total = len(items)
    print(f"Running {total} item(s) sequentially from {json_path}", flush=True)

    # Match the env the SLURM submit template exports to each job.
    env = os.environ.copy()
    env["TABPFN_DISABLE_TELEMETRY"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    failures: list[tuple[int, dict, int]] = []
    completed = 0
    for idx, item in enumerate(items, start=1):
        print(
            f"\n===== [{idx}/{total}] task_id={item['task_id']} fold={item['fold']} "
            f"repeat={item['repeat']} config_index={item['config_index']} =====",
            flush=True,
        )
        result = subprocess.run(_build_item_command(defaults, item), env=env, check=False)  # noqa: S603
        completed = idx
        if result.returncode != 0:
            print(
                f"##### Item [{idx}/{total}] FAILED (exit {result.returncode}): "
                f"task_id={item['task_id']} fold={item['fold']} "
                f"repeat={item['repeat']} config_index={item['config_index']}",
                flush=True,
            )
            failures.append((idx, item, result.returncode))
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
            f"  - [{idx}/{total}] exit {code}: task_id={item['task_id']} fold={item['fold']} "
            f"repeat={item['repeat']} config_index={item['config_index']}",
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
    args = parser.parse_args()
    sys.exit(run(args.json_path, continue_on_error=args.continue_on_error))
