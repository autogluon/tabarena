"""Check that a model predicts inductively: a row's prediction depends only on the training data and that row.

Usage::

    python -P -m tabarena.tools.audit_transduction --model NAME [--problem-type TYPE] [--num-gpus F]
        [--hyperparameters JSON] [--tolerance T] [--json PATH]

Fits the registry model once per problem type on a synthetic training table, then predicts the same probe rows in
several settings and compares each with their prediction inside the full query set (the reference):

``repeat``   the reference once more (any difference is non-determinism, which every other probe inherits)
``alone``    each probe row on its own
``batch``    the probe rows as a batch of their own
``size``     the probe rows next to decoy rows, in a batch of another size than the reference
``content``  the probe rows next to decoy rows, in a batch of the reference's size
``twin``     the probe rows after an identical copy of themselves, in a batch of the reference's size
``order``    the probe rows at the end of the batch instead of the start

The tables reach the inputs that transductive code hid in during past reviews (autogluon/tabarena#720): string
categories and integer values seen only in the query rows, a column constant on the training rows, a column that
holds one value plus NaN on the training rows, a column that is all-NaN in the query rows, a large-scale numeric
column and missing values. The query rows keep both train-constant columns at their training value; the decoys
shift every numeric column, carry an unseen category and give those columns other values.

Reading a ``FAIL``:

- ``content``: the values of other rows reach a prediction (a statistic, encoder, column filter or attention over
  the rows being predicted, in the preprocessing or inside the network).
- ``size`` with ``content`` clean: the prediction depends on how many rows are predicted together, not on their
  values, for example a random draw sized by the rows that advances a global RNG used afterwards.
- ``twin`` with ``content`` clean: duplicate query rows change each other, for example per-row hashing whose
  collisions are resolved across the query rows.
- ``order``: the position in the batch matters.
- ``alone`` and ``batch`` combine the causes above.

Run it on the CPU (the default, ``--num-gpus 0``): GPU kernels and float16 or bfloat16 autocast drift by 1e-4 to
1e-3 between batch sizes, which hides the effects above. Classification compares probabilities; regression
compares predictions divided by the standard deviation of the reference predictions. The hyperparameters default
to the model's smoke configuration (see ``audit_warmup``). Exit code 0 when every probe is within the tolerance,
2 on a ``FAIL``, 3 when the model's optional dependency is not installed, 1 on any other failure. Run it with
``python -P`` from the repository root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from tabarena.models._registry import assert_autogluon_resolves

EXIT_FAIL = 2
EXIT_MISSING_DEPENDENCY = 3
#: Largest difference a probe may show, in probability (classification) or reference standard deviations
#: (regression). CPU float32 rounding between batch shapes stays near 1e-6.
DEFAULT_TOLERANCE = 1e-5
N_TRAIN, N_QUERY, N_PROBE = 300, 200, 16
#: The batch size of the ``size`` probe, different from ``N_QUERY``.
N_SIZE_PROBE = 100
PROBES = ("repeat", "alone", "batch", "size", "content", "twin", "order")
_CATEGORIES = ("a", "b", "c", "d", "e", "f", "g", "h")


def _table(rng: np.random.Generator, n: int, *, query: bool) -> pd.DataFrame:
    """Training rows (``query=False``) or query rows of the probe table."""
    X = pd.DataFrame({f"num{i}": rng.normal(size=n) for i in range(4)})
    X["num_large"] = rng.normal(1000, 300, size=n)
    levels = ["a", "b", "c", "d", "e"] + (["f", "g"] if query else [])
    X["cat"] = pd.Categorical(rng.choice(levels, size=n), categories=_CATEGORIES)
    X["int_lowcard"] = rng.integers(0, 10 if query else 4, size=n).astype(float)
    X["const_train"] = 0.0
    const_nan = np.full(n, 5.0)
    if not query:
        const_nan[rng.random(n) < 0.3] = np.nan
    X["const_nan_train"] = const_nan
    missing = rng.normal(size=n)
    missing[rng.random(n) < 0.3] = np.nan
    X["some_missing"] = missing
    X["nan_in_query"] = np.full(n, np.nan) if query else rng.normal(size=n)
    return X


def make_probe_tables(
    problem_type: str, *, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """``(X_train, y_train, X_query, X_decoy)`` for ``problem_type``; the probe rows are ``X_query[:N_PROBE]``.

    ``X_decoy`` has ``N_QUERY`` rows: numeric columns shifted far from the training range, the unseen category
    ``h``, integer values above every query value, and the train-constant columns varying.
    """
    rng = np.random.default_rng(seed)
    X_train = _table(rng, N_TRAIN, query=False)
    signal = (
        X_train["num0"].to_numpy()
        + 0.5 * X_train["num1"].to_numpy()
        - 0.002 * (X_train["num_large"].to_numpy() - 1000)
        + 0.3 * X_train["cat"].cat.codes.to_numpy()
        + 0.3 * rng.normal(size=N_TRAIN)
    )
    if problem_type == "regression":
        y_train = pd.Series(10 * signal + 3, name="target")
    else:
        n_classes = 2 if problem_type == "binary" else 3
        edges = np.quantile(signal, np.linspace(0, 1, n_classes + 1)[1:-1])
        y_train = pd.Series(np.digitize(signal, edges), name="target")
    X_query = _table(rng, N_QUERY, query=True)
    X_decoy = _table(rng, N_QUERY, query=True)
    for col in ("num0", "num1", "num2", "num3"):
        X_decoy[col] = 5 * X_decoy[col] + 50
    X_decoy["num_large"] = 20 * X_decoy["num_large"]
    X_decoy["cat"] = pd.Categorical(["h"] * N_QUERY, categories=_CATEGORIES)
    X_decoy["int_lowcard"] = rng.integers(20, 40, size=N_QUERY).astype(float)
    X_decoy["const_train"] = rng.normal(100, 50, size=N_QUERY)
    X_decoy["const_nan_train"] = rng.normal(5, 3, size=N_QUERY)
    X_decoy["nan_in_query"] = rng.normal(size=N_QUERY)
    return X_train, y_train, X_query, X_decoy


def run_probes(
    predict: Callable[[pd.DataFrame], np.ndarray], X_query: pd.DataFrame, X_decoy: pd.DataFrame
) -> dict[str, float]:
    """The largest absolute difference of each probe from the reference, for the probe rows ``X_query[:N_PROBE]``.

    ``predict`` maps a frame to an array with one row per input row (probabilities, or regression predictions).
    """

    def pred(frame: pd.DataFrame) -> np.ndarray:
        out = np.asarray(predict(frame.reset_index(drop=True)), dtype=np.float64)
        return out.reshape(len(frame), -1)

    probe = X_query.iloc[:N_PROBE]
    rest = X_query.iloc[N_PROBE:]
    reference = pred(X_query)[:N_PROBE]
    variants = {
        "repeat": pred(X_query)[:N_PROBE],
        "alone": np.concatenate([pred(probe.iloc[[i]]) for i in range(N_PROBE)]),
        "batch": pred(probe),
        "size": pred(pd.concat([probe, X_decoy.iloc[: N_SIZE_PROBE - N_PROBE]]))[:N_PROBE],
        "content": pred(pd.concat([probe, X_decoy.iloc[: N_QUERY - N_PROBE]]))[:N_PROBE],
        "twin": pred(pd.concat([probe, probe, rest.iloc[N_PROBE:]]))[N_PROBE : 2 * N_PROBE],
        "order": pred(pd.concat([rest, probe]))[-N_PROBE:],
    }
    return {name: float(np.max(np.abs(values - reference))) for name, values in variants.items()} | {
        "reference_std": float(np.std(reference))
    }


def audit_transduction(
    model_cls,
    *,
    problem_type: str,
    hyperparameters: dict,
    num_gpus: float = 0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Fit ``model_cls`` on the probe table and run :func:`run_probes` on its predictions."""
    from autogluon.core.metrics import get_metric

    from tabarena.benchmark.task.metrics import default_eval_metric

    X_train, y_train, X_query, X_decoy = make_probe_tables(problem_type)
    workdir = tempfile.mkdtemp(prefix="tabarena_audit_transduction_")
    try:
        model = model_cls(
            path=workdir,
            name="m",
            problem_type=problem_type,
            eval_metric=get_metric(default_eval_metric(problem_type), problem_type=problem_type),
            hyperparameters=hyperparameters,
        )
        model.fit(X=X_train, y=y_train, num_gpus=num_gpus)
        predict = model.predict if problem_type == "regression" else model.predict_proba
        diffs = run_probes(predict, X_query, X_decoy)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    scale = max(diffs.pop("reference_std"), 1e-12) if problem_type == "regression" else 1.0
    scaled = {name: diff / scale for name, diff in diffs.items()}
    return {
        "model_cls": model_cls.__name__,
        "problem_type": problem_type,
        "tolerance": tolerance,
        "max_abs_diff": scaled,
        "failed": [name for name in PROBES if scaled[name] > tolerance],
    }


def diagnose(failed: list[str]) -> str:
    """One line on what the failing probes point at."""
    if not failed:
        return "OK: every probe row gets the same prediction in every setting."
    if "repeat" in failed:
        return "FAIL: the model is not deterministic on a repeated call; fix that first, the other probes inherit it."
    causes = []
    if "content" in failed:
        causes.append("other rows' values reach a prediction (statistics, encoders, filters or attention over them)")
    if "size" in failed and "content" not in failed:
        causes.append("the prediction depends on the number of rows predicted together (an RNG draw sized by rows?)")
    if "twin" in failed and "content" not in failed:
        causes.append("duplicate query rows change each other (per-row hashing or counting over the query rows?)")
    if "order" in failed:
        causes.append("the position in the batch changes the prediction")
    if not causes:
        causes.append("a row predicted alone differs from the same row in a batch")
    return "FAIL: " + "; ".join(causes) + "."


def format_audit(data: dict[str, Any]) -> str:
    """Human-readable report of :func:`audit_transduction`."""
    unit = "reference std" if data["problem_type"] == "regression" else "probability"
    lines = [f"{data['model_cls']} ({data['problem_type']}, max abs diff in {unit}, tolerance {data['tolerance']:g})"]
    for name in PROBES:
        verdict = "FAIL" if name in data["failed"] else "ok"
        lines.append(f"  {name:<8} {data['max_abs_diff'][name]:.2e}  {verdict}")
    lines.append(diagnose(data["failed"]))
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -P -m tabarena.tools.audit_transduction",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, help="Registry method name (MethodMetadata.method), e.g. LimiX-2-1016."
    )
    parser.add_argument(
        "--problem-type",
        action="append",
        choices=("binary", "multiclass", "regression"),
        help="Repeatable. Default: the model's smoke problem types, else all three it supports.",
    )
    parser.add_argument("--num-gpus", type=float, default=0, help="GPUs for the fit (default 0: the CPU path).")
    parser.add_argument("--hyperparameters", help="JSON dict of hyperparameters. Default: the model's smoke config.")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--json", help="Write the results as JSON to this path.")
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
    model_cls = registry[args.model].model_cls
    smoke_hyperparameters, smoke_problem_types = smoke_config(args.model)
    hyperparameters = json.loads(args.hyperparameters) if args.hyperparameters else smoke_hyperparameters
    problem_types = args.problem_type or list(smoke_problem_types or model_cls.supported_problem_types())
    results = []
    for problem_type in problem_types:
        try:
            data = audit_transduction(
                model_cls,
                problem_type=problem_type,
                hyperparameters=hyperparameters,
                num_gpus=args.num_gpus,
                tolerance=args.tolerance,
            )
        except ImportError as exc:
            print(f"{args.model}: optional dependency not installed ({exc})", file=sys.stderr)
            return EXIT_MISSING_DEPENDENCY
        except Exception:
            traceback.print_exc()
            return 1
        print(f"{args.model}: {format_audit(data)}", flush=True)
        results.append(data)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
    return EXIT_FAIL if any(data["failed"] for data in results) else 0


if __name__ == "__main__":
    sys.exit(main())
