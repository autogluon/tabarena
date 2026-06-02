"""Job candidates and the Ray-side cache/constraint filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tabarena.benchmark.experiment.experiment_constructor import resolve_class
from tabarena.benchmark.experiment.experiment_utils import check_cache_hit
from tabarena.benchmark.models.model_registry import infer_model_cls
from tabarena.benchmark.task.user_task import UserTask
from tabarena.utils.cache import CacheFunctionPickle

if TYPE_CHECKING:
    from tabarena.benchmark.experiment import ModelConstraints


@dataclass(frozen=True)
class JobCandidate:
    """A single (task split x config) work unit.

    Carries the identifying tuple (`task_id`, `dataset_name`, `fold`,
    `repeat`, `config_index`), the resolved `config` dict, and the dataset
    shape inputs needed by the cache/constraint filter on the Ray side
    (`should_run_job`). Used end-to-end: enumeration -> Ray filtering ->
    scheduler bundling. Only `task_id`/`fold`/`repeat`/`config_index` survive
    into the per-array-task JSON consumed by the runner script.
    """

    task_id: str
    dataset_name: str
    fold: int
    repeat: int
    config_index: int
    config: dict
    n_features: int
    n_classes: int
    n_samples_train_per_fold: int
    problem_type: str


def should_run_job_batch(*, candidates: list[JobCandidate], **kwargs) -> list[bool]:
    """Batched version for Ray."""
    return [should_run_job(candidate=c, **kwargs) for c in candidates]


def _resolve_ag_key(config: dict) -> str:
    """Resolve the AutoGluon model key from a serialized experiment config.

    `model_cls` (AGModelExperiment) and `method_cls` (plain Experiment) both
    carry the class identifier as a dotted import path. Resolve back to the
    class and use `ag_key` so the lookup matches the AG-key-based
    `model_constraints` dict. Fall back to "AutoGluon" for the full-pipeline
    AutoGluon experiments (which expose neither field).
    """
    raw_cls = config.get("model_cls") or config.get("method_cls")
    if raw_cls is not None:
        try:
            cls_obj = resolve_class(raw_cls, registry_resolver=infer_model_cls)
            return getattr(cls_obj, "ag_key", None) or raw_cls
        except (ImportError, AttributeError, ValueError, TypeError):
            return raw_cls
    if config.get("name", "").startswith("AutoGluon"):
        return "AutoGluon"
    return config.get("name", "")


def should_run_job(
    *,
    candidate: JobCandidate,
    output_dir: str,
    model_constraints: dict[str, ModelConstraints],
    ignore_cache: bool,
) -> bool:
    """Decide whether a candidate's job should run (skip on cache hit / constraint violation).

    Module-level so Ray workers can pickle it; reads everything it needs off
    the `JobCandidate` dataclass.
    """
    # Normalize task_id into the cache directory key. This MUST match the key the
    # writer uses in `run_experiments_new` (`task.slug` for UserTasks, the int task
    # id otherwise) — otherwise cache hits are looked up under the wrong path and the
    # benchmark needlessly re-runs already-cached jobs.
    try:
        task_id = int(candidate.task_id)
    except ValueError:
        task_id = UserTask.from_task_id_str(candidate.task_id).slug

    constraints = model_constraints.get(_resolve_ag_key(candidate.config))
    if constraints is not None and not constraints.applies(
        n_features=candidate.n_features,
        n_classes=candidate.n_classes,
        n_samples_train_per_fold=candidate.n_samples_train_per_fold,
        problem_type=candidate.problem_type,
    ):
        return False

    if ignore_cache:
        return True

    return not check_cache_hit(
        result_dir=output_dir,
        method_name=candidate.config["name"],
        task_id=task_id,
        fold=candidate.fold,
        repeat=candidate.repeat,
        cache_cls=CacheFunctionPickle,
        cache_cls_kwargs={"include_self_in_call": True},
        mode="local",
    )
