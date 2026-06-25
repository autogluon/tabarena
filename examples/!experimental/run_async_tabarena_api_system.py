"""Standalone, single-process example of TabArena's async / fan-out API — driving a *system*.

The fan-out backend is :class:`LocalBackend`, a single-process stand-in for a distributed,
future-based backend (e.g. one backed by ``torch.distributed``). It has the same surface
(``run(...)`` -> a future, ``wait_for_futures([...])`` -> the results), so the code around it —
build the jobs, fan them out, gather, register once, score on rank 0 — is written exactly as it
would be against a real distributed backend; only the backend object swaps.

The demo system, ``AutoGluonSystemModel``, fits a ``TabularPredictor`` (a real, if small, fit per
task), so a run over a subset takes a while — narrow ``--subset`` to go faster.

Run it:

    python "examples/!experimental/run_async_tabarena_api_system.py" --subset lite small
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tabarena.benchmark.exec_models import ExternalSystemModel
from tabarena.benchmark.experiment import Job, TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import SystemConfigGenerator

if TYPE_CHECKING:
    import pandas as pd
    from autogluon.core.metrics import Scorer

    from tabarena.benchmark.task.metadata import ValidationMetadata

# Prefix TabArena tags our registered (in-memory) method with, so we can pick its row out
# of the leaderboard.
NEW_RESULT_PREFIX = "[New] "


class AutoGluonSystemModel(ExternalSystemModel):
    """Demo system: fit an AutoGluon ``TabularPredictor`` on all the data (see ``ExternalSystemModel``)."""

    def __init__(self, *, preset: str = "medium_quality", path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.preset = preset
        self.path = path
        self._predictor = None
        self._predictor_path: str | None = None

    def _fit_system(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        target_name: str,  # name for the label column (the task's real target name when known)
        problem_type: str,  # "binary" | "multiclass" | "regression"
        eval_metric: Scorer,  # the metric the task is scored with
        validation_metadata: ValidationMetadata,  # split metadata (group_on / time_on / ...)
        num_cpus: int | None,  # CPU budget for the fit
        num_gpus: int | None,  # GPU budget for the fit
        memory_limit: float | None,  # memory budget in GB
        time_limit: float | None,  # wall-clock budget in seconds
    ):
        """Fit a ``TabularPredictor`` on all the data.

        See the parent ``ExternalSystemModel._fit_system`` docstring for the full argument
        contract — types, semantics, and the in-place / no-validation guarantees.
        """
        from autogluon.tabular import TabularPredictor

        # Materialize the label as a column named `target_name` — the task's real target name when
        # known — so its semantic meaning is preserved. `X` is ours to edit in place.
        X[target_name] = y

        self._predictor_path = self.path or tempfile.mkdtemp(prefix="tabarena_system_")
        self._predictor = TabularPredictor(
            label=target_name,
            problem_type=problem_type,
            eval_metric=eval_metric,
            path=self._predictor_path,
            verbosity=0,
        ).fit(
            X,
            presets=self.preset,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory_limit=memory_limit,
            time_limit=time_limit,
        )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        return self._predictor.predict(X)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._predictor.predict_proba(X)

    def cleanup(self):
        import shutil

        if self._predictor_path:
            shutil.rmtree(self._predictor_path, ignore_errors=True)


@dataclass(frozen=True)
class _LocalFuture:
    """Opaque handle returned by :meth:`LocalBackend.run`; its work runs when waited on."""

    thunk: Callable[[], Any]


class LocalBackend:
    """Single-process stand-in for a distributed, future-based backend.

    Mirrors the future-based surface a real backend fans jobs across — :meth:`run` submits a
    unit of work and returns a future, :meth:`wait_for_futures` blocks for them in submission
    order — but runs everything in this process (no ``torch.distributed``, no ranks). Swap this
    for a real distributed backend and the surrounding orchestration distributes unchanged.

    ``world_size`` / ``global_rank`` / ``cpu_barrier`` are accepted to match a typical distributed
    backend's constructor; the defaults describe the only configuration a single process can be
    (rank 0 of 1, a no-op barrier).
    """

    def __init__(
        self,
        *,
        world_size: int = 1,
        global_rank: int = 0,
        cpu_barrier: Callable[[], None] | None = None,
    ) -> None:
        self.world_size = world_size
        self.global_rank = global_rank
        self.cpu_barrier = cpu_barrier or (lambda: None)

    def run(self, fn: Callable[..., Any], *args: Any, cost_estimate: int = 1) -> _LocalFuture:
        """Submit ``fn(*args)`` and return a future.

        ``cost_estimate`` is the round-robin load-balancing weight a real backend uses to
        spread work across ranks (≈ rows × features); a single process ignores it.
        """
        return _LocalFuture(thunk=lambda: fn(*args))

    def wait_for_futures(self, futures: list[_LocalFuture]) -> list[Any]:
        """Block for ``futures`` and return their results in submission order."""
        return [future.thunk() for future in futures]


def evaluate_tabarena(
    *,
    preset: str = "medium_quality",
    time_limit: int = 60,
    subset: str | list[str] = "lite",
    output_dir: str | Path | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run the demo system over ``subset`` and score it on the TabArena leaderboard.

    Args:
        preset: AutoGluon preset for the demo system's internal ``TabularPredictor``.
        time_limit: Per-fit wall-clock budget (seconds), forwarded to the system.
        subset: TabArena task-subset predicate(s) (e.g. ``"lite"``, ``["lite", "small"]``).
        output_dir: where ``compare`` writes its leaderboard / figures (a temp dir if omitted).

    Returns:
        Our model's single leaderboard row (a ``pd.Series``) and the per-split results frame.
    """
    context = TabArenaContext()

    # The candidate under eval — the demo system at one preset. `system_experiments=True` fits it
    # directly on the training data (no bagging); `time_limit` is forwarded to the system's fit.
    generator = SystemConfigGenerator(
        model_cls=AutoGluonSystemModel,
        name="DemoAutoGluonSystem",
        manual_configs=[{"preset": preset}],
    )
    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(generator, 0)],
        system_experiments=True,
    ).build_experiments(time_limit=time_limit)

    # Build the work units once. pre_materialize=True fetches each task's data now, so the
    # fanned-out jobs don't each re-download it (as on an offline cluster); drop it to fetch lazily.
    jobs = context.build_jobs(experiments, subset=subset, pre_materialize=True)
    print("N jobs:", len(jobs))
    # Pre-compute each job's backend load-balancing cost (≈ train rows × features) from its task
    # metadata, then zip it back onto the jobs. Each `meta` carries exactly the job's one split.
    costs = []
    for task_meta in context.metadata_for_jobs(jobs):
        split_meta = task_meta.split_metadata
        costs.append(max(int(split_meta.num_instances_train * split_meta.num_features_train), 1))
    jobs_costs = zip(jobs, costs, strict=True)

    # Fan the jobs across the backend and gather the raw per-split results. LocalBackend runs them
    # in this process; swapping in a real distributed backend distributes the same code.
    backend = LocalBackend(world_size=1, global_rank=0, cpu_barrier=lambda: None)

    def _run(job: Job) -> list[dict]:
        """Run one job in-process; register=False so the whole gather is registered once below.
        Each job is independent with nothing to hold or clean up here.
        """
        return context.run_job(job, expname=None, register=False, debug_mode=True)

    futures = [backend.run(_run, job, cost_estimate=cost) for job, cost in jobs_costs]
    raw_results = [r for batch in backend.wait_for_futures(futures) for r in batch]

    # Only rank 0 holds the full gather and scores it (here world_size=1, so always rank 0).
    context.register(raw_results, new_result_prefix=NEW_RESULT_PREFIX)
    # One call gives our model's single leaderboard row (a pd.Series) + its per-split results
    # frame. return_single asserts exactly one registered model; elo/rank are still computed
    # against all baselines.
    return context.compare(
        output_dir=output_dir,
        subset=subset,
        return_results=True,
        return_single=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--subset",
        nargs="+",
        default=["lite", "small"],
        help="TabArena subset predicate(s), AND-composed (e.g. lite small classification).",
    )
    parser.add_argument("--preset", default="medium_quality", help="AutoGluon preset for the demo system.")
    parser.add_argument("--time-limit", type=int, default=60, help="Per-fit time limit in seconds.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where compare() writes the leaderboard / figures (temp dir if omitted).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    leaderboard_row, per_split = evaluate_tabarena(
        preset=args.preset,
        time_limit=args.time_limit,
        subset=args.subset,
        output_dir=args.output_dir,
    )
    print("\n=== per-split results ===")
    print(per_split[["dataset", "fold", "method", "metric_error"]].to_markdown(index=False))
    print("\n=== leaderboard row (our registered method) ===")
    print(leaderboard_row.to_markdown())
