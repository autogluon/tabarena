"""Measure how much a model's pickle grows per training row.

Usage::

    python -P -m tabarena.tools.audit_pickle --model NAME [--problem-type TYPE] [--rows N] [--num-gpus F]
        [--hyperparameters JSON] [--json PATH]

Fits the registry model class twice, on synthetic frames of ``N`` and ``2N`` training rows, saves each
fit and reports the size of ``model.pkl``, the other files beside it, and the in-memory pickle
AutoGluon makes to size the model (``_get_pickled_size``), each with its time. AutoGluon pickles a
model inside the timed fit, so a fitted state that grows with the training rows and sits in the
pickle (an in-context model's KV cache or encoded context) is copied on every dataset: it inflates
the measured training time everywhere and doubles the RAM on large tables. ``FLAG`` means
``model.pkl`` grows by more than ``16x`` the table's own bytes per row and by more than 1 KiB per row,
and tensors make up most of that growth; the fix is to keep that state in its own file
(:class:`tabarena.models.context_cache.ContextCacheFileMixin`). Storing the training table itself
(TabPFN, KNN) stays below the threshold. A pickle that grows without tensors behind it (the nodes of
boosted trees or forests, which grow with the data) is reported as ``NOTE``: it costs the same copy,
but the model's own structure is not something to move out of the pickle.

The hyperparameters default to the model's smoke configuration (see ``audit_warmup``). Exit code 0
on ``OK`` and ``NOTE``, 2 on ``FLAG``, 3 when the model's optional dependency is not installed, 1 on
any other failure. Run it with ``python -P`` from the repository root.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any

from tabarena.models._registry import assert_autogluon_resolves

EXIT_FLAG = 2
EXIT_MISSING_DEPENDENCY = 3
#: ``model.pkl`` may grow by at most this multiple of the training table's bytes per row ...
MAX_TABLE_MULTIPLE = 16
#: ... unless it grows by less than this many bytes per row.
MIN_FLAG_BYTES_PER_ROW = 1024


def measure_fit(
    model_cls,
    *,
    problem_type: str,
    n_rows: int,
    hyperparameters: dict,
    num_gpus: float = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit ``model_cls`` on ``n_rows`` synthetic rows, then size and save it; return the measurements."""
    from autogluon.core.metrics import get_metric

    from tabarena.benchmark.task.metrics import default_eval_metric
    from tabarena.models.context_cache import tensor_nbytes
    from tabarena.utils.synthetic_data import make_synthetic_frames

    X, y, _ = make_synthetic_frames(problem_type, n_rows=n_rows, seed=seed)
    workdir = tempfile.mkdtemp(prefix="tabarena_audit_pickle_")
    try:
        model = model_cls(
            path=workdir,
            name="m",
            problem_type=problem_type,
            eval_metric=get_metric(default_eval_metric(problem_type), problem_type=problem_type),
            hyperparameters=hyperparameters,
        )
        model.fit(X=X, y=y, num_gpus=num_gpus)
        # The tensors model.pkl carries: a cache that ContextCacheFileMixin saves apart is left out.
        without_cache = getattr(model, "_without_context_cache", contextlib.nullcontext)
        with without_cache():
            pickled_tensor_bytes = tensor_nbytes(model)
        start = time.perf_counter()
        size_estimate = model._get_pickled_size()
        size_estimate_s = time.perf_counter() - start
        start = time.perf_counter()
        path = model.save()
        save_s = time.perf_counter() - start
        files = {name: os.path.getsize(os.path.join(path, name)) for name in os.listdir(path)}
        pkl = files.pop("model.pkl", 0)
        return {
            "n_rows": n_rows,
            "table_bytes": int(X.memory_usage(deep=True).sum() + y.memory_usage(deep=True)),
            "model_pkl_bytes": pkl,
            "pickled_tensor_bytes": pickled_tensor_bytes,
            "other_files_bytes": sum(size for name, size in files.items() if os.path.isfile(os.path.join(path, name))),
            "pickled_size_estimate": size_estimate,
            "pickled_size_estimate_s": size_estimate_s,
            "save_s": save_s,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def audit_pickle(
    model_cls,
    *,
    problem_type: str,
    hyperparameters: dict,
    n_rows: int = 200,
    num_gpus: float = 0,
) -> dict[str, Any]:
    """Fit ``model_cls`` on ``n_rows`` and ``2 * n_rows`` rows and compare how much each part grows per row."""
    small = measure_fit(
        model_cls, problem_type=problem_type, n_rows=n_rows, hyperparameters=hyperparameters, num_gpus=num_gpus
    )
    large = measure_fit(
        model_cls, problem_type=problem_type, n_rows=2 * n_rows, hyperparameters=hyperparameters, num_gpus=num_gpus
    )
    extra = large["n_rows"] - small["n_rows"]
    per_row = {
        key: (large[key] - small[key]) / extra
        for key in (
            "table_bytes",
            "model_pkl_bytes",
            "pickled_tensor_bytes",
            "other_files_bytes",
            "pickled_size_estimate",
        )
    }
    limit = max(MAX_TABLE_MULTIPLE * per_row["table_bytes"], MIN_FLAG_BYTES_PER_ROW)
    grows = per_row["model_pkl_bytes"] > limit
    from_tensors = per_row["pickled_tensor_bytes"] >= per_row["model_pkl_bytes"] / 2
    return {
        "model_cls": model_cls.__name__,
        "problem_type": problem_type,
        "fits": [small, large],
        "bytes_per_row": per_row,
        "limit_bytes_per_row": limit,
        "flag": grows and from_tensors,
        "note": grows and not from_tensors,
    }


def format_audit(data: dict[str, Any]) -> str:
    """Human-readable report of :func:`audit_pickle`."""
    per_row = data["bytes_per_row"]
    lines = [f"{data['model_cls']} ({data['problem_type']})"]
    for fit in data["fits"]:
        lines.append(
            f"  {fit['n_rows']:>5} rows: model.pkl {fit['model_pkl_bytes']:>12,d} B, beside it "
            f"{fit['other_files_bytes']:>12,d} B, size estimate {fit['pickled_size_estimate']:>12,d} B "
            f"({fit['pickled_size_estimate_s']:.2f} s), save {fit['save_s']:.2f} s"
        )
    lines.append(
        f"  per training row: table {per_row['table_bytes']:,.0f} B, model.pkl {per_row['model_pkl_bytes']:,.0f} B "
        f"(tensors {per_row['pickled_tensor_bytes']:,.0f} B), "
        f"beside it {per_row['other_files_bytes']:,.0f} B (limit for model.pkl {data['limit_bytes_per_row']:,.0f} B)"
    )
    if data["flag"]:
        lines.append(
            "FLAG: model.pkl grows with the training rows, mostly through tensors (a context or KV cache). AutoGluon "
            "pickles the model inside the timed fit, so this state is copied on every dataset; keep it in its own "
            "file (tabarena.models.context_cache.ContextCacheFileMixin)."
        )
    elif data["note"]:
        lines.append(
            "NOTE: model.pkl grows with the training rows, but not through tensors (tree nodes, a stored table). "
            "AutoGluon copies it inside the timed fit like any pickle; nothing for ContextCacheFileMixin to move."
        )
    else:
        lines.append("OK: nothing that scales with the training rows sits in model.pkl.")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -P -m tabarena.tools.audit_pickle",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, help="Registry method name (MethodMetadata.method), e.g. Kumo-Tabular."
    )
    parser.add_argument("--problem-type", choices=("binary", "multiclass", "regression"))
    parser.add_argument(
        "--rows", type=int, default=200, help="Training rows of the smaller fit (the larger has twice)."
    )
    parser.add_argument("--num-gpus", type=float, default=0, help="GPUs for the fits (default 0: the CPU path).")
    parser.add_argument("--hyperparameters", help="JSON dict of hyperparameters. Default: the model's smoke config.")
    parser.add_argument("--json", help="Write the measurements as JSON to this path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    assert_autogluon_resolves()
    args = _build_parser().parse_args(argv)

    from tabarena.models import get_model_registry
    from tabarena.tools.audit_warmup import smoke_config

    registry = get_model_registry()
    if args.model not in registry:
        print(f"Unknown model {args.model!r}. Registered: {', '.join(sorted(registry))}", file=sys.stderr)
        return 1
    smoke_hyperparameters, smoke_problem_types = smoke_config(args.model)
    hyperparameters = json.loads(args.hyperparameters) if args.hyperparameters else smoke_hyperparameters
    problem_type = args.problem_type or (smoke_problem_types[0] if smoke_problem_types else "binary")
    try:
        data = audit_pickle(
            registry[args.model].model_cls,
            problem_type=problem_type,
            hyperparameters=hyperparameters,
            n_rows=args.rows,
            num_gpus=args.num_gpus,
        )
    except ImportError as exc:
        print(f"{args.model}: optional dependency not installed ({exc})", file=sys.stderr)
        return EXIT_MISSING_DEPENDENCY
    except Exception:
        traceback.print_exc()
        return 1
    print(f"{args.model}: {format_audit(data)}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2, default=str)
    return EXIT_FLAG if data["flag"] else 0


if __name__ == "__main__":
    sys.exit(main())
