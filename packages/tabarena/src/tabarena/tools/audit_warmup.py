"""Audit what the timed fit and predict of a benchmarked model still pay for.

Usage::

    python -P -m tabarena.tools.audit_warmup --model NAME [--problem-type TYPE] [--num-cpus N]
        [--num-gpus F] [--hyperparameters JSON] [--json PATH]
    python -P -m tabarena.tools.audit_warmup --results DIR [--json PATH]

``--model`` builds the registry model's ``AGSingleWrapper`` the way one benchmark item does, runs
its ``warmup_fn`` through :func:`tabarena.models.warmup.run_warmup_fn`, then fits and predicts on
synthetic frames through ``fit_custom`` and prints the warm-up report (status, failed steps,
preloaded weights, dummy fits) and the timing audit (packages imported and CUDA state inside each
timer). The hyperparameters default to the model's smoke configuration in
``tests/tabarena/models/smoke_configs.py`` when this is a repository checkout, else to the model's
own defaults. Exit code 0 when the audit ran, 3 when the model's optional dependency is not
installed, 1 on any other failure. ``tests/tabarena/models/test_warmup_coverage.py`` runs this
per registered model.

``--results`` summarizes ``experiment_metadata["warmup_report"]`` and ``["timing_audit"]`` over the
``results.pkl`` files of a benchmark run: warm-up status counts, failed steps and the packages the
timed sections imported, per method.

Run it with ``python -P`` from the repository root: a directory named ``autogluon/`` on
``sys.path[0]`` shadows the installed AutoGluon and empties the registry (see
:func:`tabarena.models._registry.assert_autogluon_resolves`, which runs first).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from tabarena.models._registry import assert_autogluon_resolves

#: Exit code when the model's optional dependency is not installed.
EXIT_MISSING_DEPENDENCY = 3
PROBLEM_TYPES = ("binary", "multiclass", "regression")
_SMOKE_CONFIGS = Path(__file__).resolve().parents[5] / "tests" / "tabarena" / "models" / "smoke_configs.py"


def smoke_config(method: str) -> tuple[dict, tuple[str, ...] | None]:
    """The smoke ``(hyperparameters, problem_types)`` of ``method`` from the checkout's test config.

    Returns ``({}, None)`` when this is not a repository checkout or the test module cannot be
    imported (it needs pytest).
    """
    if not _SMOKE_CONFIGS.is_file():
        return {}, None
    try:
        spec = importlib.util.spec_from_file_location("_tabarena_smoke_configs", _SMOKE_CONFIGS)
        module = importlib.util.module_from_spec(spec)
        # Registered before execution: the module's dataclass looks itself up in sys.modules.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cfg = module.smoke_for(method)
    except Exception as exc:
        print(f"Smoke config unavailable ({exc!r}); using the model's default hyperparameters.")
        return {}, None
    return dict(cfg.hyperparameters), cfg.problem_types


def audit_model(
    method: str,
    *,
    problem_type: str,
    num_cpus: int,
    num_gpus: float,
    hyperparameters: dict,
) -> dict[str, Any]:
    """Warm up, fit and predict ``method`` once on synthetic data; return the JSON-able audit.

    Mirrors ``ExperimentRunner._run`` for one item: the exec model is ``AGSingleWrapper`` with the
    registry model class, ``raise_on_model_failure=True`` (so a missing library surfaces as its
    ``ImportError``), ``run_warmup_fn`` around ``warmup_fn``, then ``fit_custom`` on
    :func:`tabarena.utils.synthetic_data.make_synthetic_frames` (seed 1, so the frames differ from
    the warm-up's dummy fit). The predictor artifacts live in a temporary directory removed at the
    end. ``ImportError`` propagates; the caller maps it to :data:`EXIT_MISSING_DEPENDENCY`.
    """
    from autogluon.core.metrics import get_metric

    from tabarena.benchmark.exec_models.autogluon import AGSingleWrapper
    from tabarena.benchmark.task.metrics import default_eval_metric
    from tabarena.models import get_model_registry
    from tabarena.models.warmup import WarmupReport, run_warmup_fn
    from tabarena.utils.synthetic_data import make_synthetic_frames

    info = get_model_registry()[method]
    workdir = tempfile.mkdtemp(prefix="tabarena_audit_")
    model = AGSingleWrapper(
        model_cls=info.model_cls,
        model_hyperparameters=hyperparameters,
        init_kwargs={"default_base_path": workdir, "verbosity": 0},
        fit_kwargs={"num_cpus": num_cpus, "num_gpus": num_gpus, "raise_on_model_failure": True},
        problem_type=problem_type,
        eval_metric=get_metric(default_eval_metric(problem_type), problem_type=problem_type),
    )
    out: dict[str, Any] = {
        "method": method,
        "model_cls": info.model_cls.__name__,
        "problem_type": problem_type,
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
        "hyperparameters": hyperparameters,
    }
    try:
        warmup_fn = model.warmup_fn
        report = (
            WarmupReport(status="none", label=method) if warmup_fn is None else run_warmup_fn(warmup_fn, label=method)
        )
        out["warmup_report"] = report.to_dict()
        X, y, X_test = make_synthetic_frames(problem_type, n_rows=160, seed=1)
        fitted = model.fit_custom(X, y, X_test, split_seed=0)
        out["timing_audit"] = fitted["timing_audit"]
        out["time_train_s"] = fitted["time_train_s"]
        out["time_infer_s"] = fitted["time_infer_s"]
        metadata = model.get_metadata()
        out["method_metadata"] = {
            key: metadata.get(key)
            for key in ("persisted_models", "prepared_for_inference", "shared_weights", "per_child_test_source")
        }
    finally:
        try:
            model.cleanup()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    return out


def _section_lines(name: str, audit: dict | None) -> list[str]:
    if audit is None:
        return [f"  {name}: audit unavailable"]
    cuda = f"{audit.get('cuda_initialized_before')} -> {audit.get('cuda_initialized_after')}"
    ray = f"{audit.get('ray_initialized_before')} -> {audit.get('ray_initialized_after')}"
    return [
        f"  {name}: new_packages={audit.get('new_packages')} new_modules={audit.get('new_modules')}",
        f"    new_submodule_packages={audit.get('new_submodule_packages')}",
        f"    cuda_initialized {cuda}; ray_initialized {ray}",
    ]


def format_model_audit(data: dict[str, Any]) -> str:
    """Human-readable summary of an :func:`audit_model` result."""
    report = data["warmup_report"]
    lines = [
        f"{data['method']} ({data['model_cls']}, {data['problem_type']}, num_cpus={data['num_cpus']}, "
        f"num_gpus={data['num_gpus']})",
        f"warm-up: status={report['status']} duration_s={report['duration_s']} "
        f"imported={report['imported_modules']} cuda_initialized={report['cuda_initialized']}",
        f"  failed_steps={report['failed_steps']}",
    ]
    if report.get("error"):
        lines.append(f"  error={report['error']}")
    for entry in report["weights_preloaded"]:
        lines.append(
            f"  weights: {entry.get('library')}/{entry.get('variant')}@{entry.get('device')}:{entry.get('dtype')} "
            f"loaded_by={entry.get('loaded_by')} load_time_s={entry.get('load_time_s')} n_bytes={entry.get('n_bytes')}"
        )
    for record in report["dummy_fits"]:
        lines.append(
            f"  dummy_fit: {record.get('model_cls')} ran={record.get('ran')} duration_s={record.get('duration_s')} "
            f"skipped_reason={record.get('skipped_reason')} error={record.get('error')} "
            f"torch_globals_changed={record.get('torch_globals_changed')}"
        )
    if report.get("ray"):
        lines.append(f"  ray={report['ray']}")
    audit = data.get("timing_audit") or {}
    lines.append(f"timed sections: time_train_s={data.get('time_train_s')} time_infer_s={data.get('time_infer_s')}")
    lines.extend(_section_lines("fit", audit.get("fit")))
    lines.extend(_section_lines("predict", audit.get("predict")))
    metadata = data.get("method_metadata") or {}
    lines.append(
        f"inference: persisted_models={metadata.get('persisted_models')} "
        f"prepared_for_inference={metadata.get('prepared_for_inference')} "
        f"per_child_test_source={metadata.get('per_child_test_source')}"
    )
    shared = (metadata.get("shared_weights") or {}).get("registry") or {}
    if shared.get("entries"):
        lines.append(f"  shared_weights: stats={shared.get('stats')} entries={len(shared['entries'])}")
    return "\n".join(lines)


def summarize_results(paths: list[Path]) -> dict[str, Any]:
    """Aggregate the warm-up and timing-audit blocks of ``results.pkl`` files, per method.

    Every field is read with ``.get`` so results written before these blocks existed count under
    ``missing``. Returns ``{method: {...}}`` with ``n``, ``missing_warmup_report``,
    ``missing_timing_audit``, ``warmup_status`` counts, ``failed_steps`` counts,
    ``time_warmup_s_mean``, and per timed section the ``new_packages`` counts and how many results
    created the CUDA context inside the timer.
    """
    from tabarena.utils.pickle_utils import load_pickle

    summary: dict[str, dict[str, Any]] = {}
    for path in paths:
        result = load_pickle(path)
        method = str(result.get("framework", "unknown"))
        entry = summary.setdefault(
            method,
            {
                "n": 0,
                "missing_warmup_report": 0,
                "missing_timing_audit": 0,
                "warmup_status": Counter(),
                "failed_steps": Counter(),
                "time_warmup_s": [],
                "fit": {"new_packages": Counter(), "cuda_initialized_inside": 0},
                "predict": {"new_packages": Counter(), "cuda_initialized_inside": 0},
            },
        )
        entry["n"] += 1
        experiment_metadata = result.get("experiment_metadata") or {}
        report = experiment_metadata.get("warmup_report")
        if report is None:
            entry["missing_warmup_report"] += 1
        else:
            entry["warmup_status"][str(report.get("status"))] += 1
            entry["failed_steps"].update(report.get("failed_steps") or [])
        duration = experiment_metadata.get("time_warmup_s")
        if duration is not None:
            entry["time_warmup_s"].append(float(duration))
        audit = experiment_metadata.get("timing_audit")
        if audit is None:
            entry["missing_timing_audit"] += 1
            continue
        for section in ("fit", "predict"):
            block = audit.get(section)
            if not block:
                continue
            entry[section]["new_packages"].update(block.get("new_packages") or [])
            if block.get("cuda_initialized_before") is False and block.get("cuda_initialized_after") is True:
                entry[section]["cuda_initialized_inside"] += 1
    for entry in summary.values():
        durations = entry.pop("time_warmup_s")
        entry["time_warmup_s_mean"] = sum(durations) / len(durations) if durations else None
        entry["warmup_status"] = dict(entry["warmup_status"])
        entry["failed_steps"] = dict(entry["failed_steps"].most_common())
        for section in ("fit", "predict"):
            entry[section]["new_packages"] = dict(entry[section]["new_packages"].most_common())
    return summary


def format_results_summary(summary: dict[str, Any]) -> str:
    """Human-readable form of :func:`summarize_results`."""
    lines = []
    for method, entry in sorted(summary.items()):
        lines.append(
            f"{method}: n={entry['n']} warmup_status={entry['warmup_status']} "
            f"time_warmup_s_mean={entry['time_warmup_s_mean']} "
            f"missing(warmup_report={entry['missing_warmup_report']}, timing_audit={entry['missing_timing_audit']})"
        )
        if entry["failed_steps"]:
            lines.append(f"  failed_steps={entry['failed_steps']}")
        for section in ("fit", "predict"):
            block = entry[section]
            lines.append(
                f"  {section}: new_packages={block['new_packages']} "
                f"cuda_initialized_inside={block['cuda_initialized_inside']}"
            )
    return "\n".join(lines) if lines else "No results found."


def _default_num_gpus(method: str) -> float:
    from tabarena.models import get_model_registry
    from tabarena.utils.resources import detect_num_gpus

    if get_model_registry()[method].method_metadata.compute != "gpu":
        return 0
    return min(1, detect_num_gpus())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -P -m tabarena.tools.audit_warmup",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--model", help="Registry method name (MethodMetadata.method), for example TabPFN-3.")
    target.add_argument("--results", type=Path, help="Directory of results.pkl files to summarize.")
    parser.add_argument(
        "--problem-type",
        choices=PROBLEM_TYPES,
        help="Default: the first problem type of the model's smoke config, else binary.",
    )
    parser.add_argument("--num-cpus", type=int, help="Default: the CPUs this process may run on.")
    parser.add_argument(
        "--num-gpus", type=float, help="Default: 1 for a compute='gpu' model when a GPU is detected, else 0."
    )
    parser.add_argument(
        "--hyperparameters",
        help="JSON dict of model hyperparameters. Default: the model's smoke config, else the model defaults.",
    )
    parser.add_argument("--json", type=Path, help="Write the full audit (or summary) as JSON to this path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    assert_autogluon_resolves()
    args = _build_parser().parse_args(argv)

    if args.results is not None:
        from tabarena.benchmark.result.raw_loading import fetch_raw_result_paths

        summary = summarize_results(fetch_raw_result_paths(args.results))
        print(format_results_summary(summary))
        if args.json is not None:
            args.json.write_text(json.dumps(summary, indent=2, default=str))
        return 0

    from tabarena.models import get_model_registry
    from tabarena.utils.resources import detect_num_cpus

    registry = get_model_registry()
    if args.model not in registry:
        print(f"Unknown model {args.model!r}. Registered: {', '.join(sorted(registry))}", file=sys.stderr)
        return 1
    smoke_hyperparameters, smoke_problem_types = smoke_config(args.model)
    hyperparameters = json.loads(args.hyperparameters) if args.hyperparameters else smoke_hyperparameters
    problem_type = args.problem_type or (smoke_problem_types[0] if smoke_problem_types else "binary")
    num_cpus = args.num_cpus if args.num_cpus is not None else detect_num_cpus()
    num_gpus = args.num_gpus if args.num_gpus is not None else _default_num_gpus(args.model)

    try:
        data = audit_model(
            args.model,
            problem_type=problem_type,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            hyperparameters=hyperparameters,
        )
    except ImportError as exc:
        print(f"{args.model}: optional dependency not installed ({exc})", file=sys.stderr)
        return EXIT_MISSING_DEPENDENCY
    except Exception:
        traceback.print_exc()
        return 1
    print(format_model_audit(data))
    if args.json is not None:
        args.json.write_text(json.dumps(data, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
