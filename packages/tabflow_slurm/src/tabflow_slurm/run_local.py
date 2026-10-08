"""Sequential, local runner for a generated benchmark job JSON.

The non-cluster counterpart to ``submit_template.sh``. Where the SLURM submit
script runs one array task (``jobs[SLURM_ARRAY_TASK_ID]``) and loops over that
task's bundled items, this runner flattens *all* jobs and *all* items into a
single sequential loop and runs ``run_tabarena_experiment``'s per-item logic
once per item (one (experiment, dataset, fold, repeat) work unit each).

``--num_workers N`` keeps up to N items in flight at once (subprocess mode only), each writing
its output to its own file under ``--item_log_dir``. That is for systems whose compute is remote (a
hosted API): the node only waits on the provider, so several items can share it without sharing
compute. Never run a local model with N > 1, its fits would compete for the node's CPUs and the
recorded times would be wrong.

Two execution modes (``--execution_mode``):
    - ``subprocess`` (default): each item runs in its own fresh subprocess via
      ``run_tabarena_experiment.py`` — so every model fit stays isolated (fresh
      memory / Ray / GPU context), exactly like an independent SLURM array task.
    - ``in_process``: each item runs in this runner's own Python process by
      calling ``run_experiment`` directly. Faster and debugger friendly, but
      fits share global state and a hard crash aborts the whole run.

Invoke it via the command emitted by ``LocalSequentialSetup.get_run_commands``:

    <python> -P -m tabflow_slurm.run_local <job.json> [--continue_on_error True]
                                                      [--execution_mode in_process]
                                                      [--num_workers N --item_log_dir DIR]

Subprocess mode mirrors the SLURM template's environment: ``python -P``, the thread-count
variables unset (see ``HYGIENE_ENV_VARS``) and ``--offline_weights`` from the job defaults.

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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tabflow_slurm.run_tabarena_experiment import _str2bool

#: Thread-pool and OpenMP placement variables a login shell may carry; the models and Ray size
#: their pools themselves, so the item processes start without them (same as ``submit_template.sh``).
HYGIENE_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
)


def _build_item_command(defaults: dict, item: dict) -> list[str]:
    """Build the ``run_tabarena_experiment.py`` argv for a single item.

    Mirrors ``submit_template.sh::run_one``. Every value is stringified:
    ``str(True/False)`` -> ``"True"/"False"`` (accepted by the runner's
    ``_str2bool``) and ``str(None)`` -> ``"None"`` (accepted by
    ``_parse_int_or_none`` for ``num_cpus``/``memory_limit``).
    """
    return [
        str(defaults["python"]),
        "-P",
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
        "--offline_weights",
        str(bool(defaults.get("offline_weights", False))),
        # Local machines rarely confine the process to num_cpus CPUs; report the mismatch instead.
        "--cpu_budget_check",
        "warn",
        "--require_warmup",
        str(bool(defaults.get("require_warmup", True))),
    ]


def _run_item_subprocess(defaults: dict, item: dict, env: dict, log_path: Path | None = None) -> int:
    """Run one item in its own subprocess; return its exit code (0 == success).

    With ``log_path`` the item's stdout and stderr go to that file instead of this process's output.
    """
    command = _build_item_command(defaults, item)
    if log_path is None:
        return subprocess.run(command, env=env, check=False).returncode  # noqa: S603
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        return subprocess.run(command, env=env, check=False, stdout=log, stderr=subprocess.STDOUT).returncode  # noqa: S603


def _item_label(item: dict) -> str:
    return f"experiment={item['experiment']} dataset={item['dataset']} fold={item['fold']} repeat={item['repeat']}"


def _item_log_path(log_dir: Path, idx: int, item: dict) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"{item['experiment']}__{item['dataset']}")
    return log_dir / f"{idx:05d}__{safe}__r{item['repeat']}f{item['fold']}.log"


def _run_parallel(
    items: list[dict],
    *,
    defaults: dict,
    env: dict,
    num_workers: int,
    item_log_dir: Path,
    continue_on_error: bool,
) -> tuple[list[tuple[int, dict, int]], int]:
    """Run ``items`` with up to ``num_workers`` subprocesses in flight; return ``(failures, completed)``.

    Without ``continue_on_error`` no new item starts after the first failure; the ones in flight finish.
    """
    total = len(items)
    failures: list[tuple[int, dict, int]] = []
    completed = 0
    stop = False
    print(f"Item logs: {item_log_dir}", flush=True)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        pending = iter(enumerate(items, start=1))
        futures: dict = {}

        def submit_next() -> None:
            for idx, item in pending:
                log_path = _item_log_path(item_log_dir, idx, item)
                print(f"===== [{idx}/{total}] START {_item_label(item)} log={log_path.name}", flush=True)
                futures[pool.submit(_timed_item, defaults, item, env, log_path)] = (idx, item)
                return

        for _ in range(num_workers):
            submit_next()
        while futures:
            future = next(as_completed(futures))
            idx, item = futures.pop(future)
            code, seconds = future.result()
            completed += 1
            status = "OK" if code == 0 else f"FAILED (exit {code})"
            print(f"===== [{idx}/{total}] {status} after {seconds:.0f}s {_item_label(item)}", flush=True)
            if code != 0:
                failures.append((idx, item, code))
                if not continue_on_error:
                    stop = True
            if not stop:
                submit_next()
    if stop:
        print("Stopped starting new items (continue_on_error=False).", flush=True)
    return failures, completed


def _timed_item(defaults: dict, item: dict, env: dict, log_path: Path) -> tuple[int, float]:
    start = time.monotonic()
    code = _run_item_subprocess(defaults, item, env, log_path)
    return code, time.monotonic() - start


def _setup_in_process(defaults: dict) -> None:
    """One-time setup for `in_process` mode: Ray + telemetry env.

    Subprocess mode does this per child via ``run_tabarena_experiment``'s ``__main__``; in-process
    we must do it once before the first fit. We reuse ``setup_slurm_job`` with the SLURM
    shared-resources Ray setup disabled. Cache configuration is not done here — each item's
    ``run_experiment`` applies the ``JobBatch``'s ``cache_config`` before its fit.
    """
    from tabflow_slurm.slurm_utils import apply_offline_weights_env, setup_slurm_job

    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    apply_offline_weights_env(bool(defaults.get("offline_weights", False)))
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
            cleanup_on_failure=False,
            cpu_budget_check="warn",
            require_warmup=bool(defaults.get("require_warmup", True)),
        )
    except Exception as exc:
        print(f"  in-process item raised: {exc!r}", flush=True)
        return 1
    return 0


def run(
    json_path: str,
    *,
    continue_on_error: bool,
    execution_mode: str = "subprocess",
    num_workers: int = 1,
    item_log_dir: str | None = None,
) -> int:
    """Run every item in `json_path`; return 0 iff all succeeded.

    `execution_mode` is "subprocess" (one fresh process per item, isolated) or
    "in_process" (run every item in this process; faster but no isolation). `num_workers` > 1
    keeps that many subprocess items in flight, each logging to its own file under `item_log_dir`
    (default: `<json stem>_item_logs` next to the JSON).
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1, got {num_workers}")
    if num_workers > 1 and execution_mode != "subprocess":
        raise ValueError(
            "num_workers > 1 needs execution_mode='subprocess' (in-process items would share one process)."
        )
    with Path(json_path).open() as f:
        jobs_dict = json.load(f)

    defaults = jobs_dict["defaults"]
    items = [item for job in jobs_dict["jobs"] for item in job["items"]]
    total = len(items)
    how = "sequentially" if num_workers == 1 else f"with {num_workers} in flight"
    print(f"Running {total} item(s) {how} from {json_path} (mode={execution_mode})", flush=True)

    # Subprocess mode: match the env the SLURM submit template exports to each job.
    env = os.environ.copy()
    env["TABPFN_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    for name in HYGIENE_ENV_VARS:
        env.pop(name, None)

    if execution_mode == "in_process":
        _setup_in_process(defaults)

    if num_workers > 1:
        log_dir = Path(item_log_dir) if item_log_dir else Path(json_path).with_name(f"{Path(json_path).stem}_item_logs")
        failures, completed = _run_parallel(
            items,
            defaults=defaults,
            env=env,
            num_workers=num_workers,
            item_log_dir=log_dir,
            continue_on_error=continue_on_error,
        )
        return _summarize(total, completed, failures)

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
    return _summarize(total, completed, failures)


def _summarize(total: int, completed: int, failures: list[tuple[int, dict, int]]) -> int:
    """Print the run summary; return 1 when any item failed, else 0."""
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Items in flight at once (subprocess mode). Only for systems whose compute is remote (a hosted API).",
    )
    parser.add_argument(
        "--item_log_dir",
        type=str,
        default=None,
        help="Per-item log files when --num_workers > 1 (default: <json stem>_item_logs next to the JSON).",
    )
    args = parser.parse_args()
    sys.exit(
        run(
            args.json_path,
            continue_on_error=args.continue_on_error,
            execution_mode=args.execution_mode,
            num_workers=args.num_workers,
            item_log_dir=args.item_log_dir,
        ),
    )
