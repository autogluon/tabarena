from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from autogluon.common.loaders import load_pd

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.benchmark.task.metadata import TaskMetadataCollection


def _get_n_repeats(n_instances: int, tabarena_lite: bool = False) -> int:
    """Get the number of n_repeats for the full benchmark run based on the 2025 paper.

    Parameters
    ----------
    n_instances: int
        Important: these are the number of training samples!

    Returns:
    -------
    n_repeats: int
    """
    if tabarena_lite:
        return 1

    if n_instances < 2_500:
        tabarena_repeats = 10
    elif n_instances > 250_000:
        tabarena_repeats = 1
    else:
        tabarena_repeats = 3
    return tabarena_repeats


def _get_problem_type_from_n_classes(n_classes: int) -> str:
    if n_classes == 0:
        return "regression"
    if n_classes == 2:
        return "binary"
    if n_classes > 2:
        return "multiclass"
    raise ValueError(f"Invalid n_classes: {n_classes}")


def add_extra_task_metadata_info(task_metadata: pd.DataFrame) -> pd.DataFrame:
    task_metadata["n_features"] = (task_metadata["NumberOfFeatures"] - 1).astype(int)

    task_metadata["n_classes"] = task_metadata["NumberOfClasses"].astype(int)
    task_metadata["problem_type"] = task_metadata["n_classes"].apply(_get_problem_type_from_n_classes)

    task_metadata["dataset"] = task_metadata["name"]
    return task_metadata


def enrich_legacy_task_metadata(task_metadata: pd.DataFrame) -> pd.DataFrame:
    """Add the TabArena split columns (``n_folds``/``n_repeats``/``n_samples_*_per_fold``) and the
    derived schema columns (``n_features``/``n_classes``/``problem_type``/``dataset``) that the
    legacy ``task_metadata`` format — and :meth:`TaskMetadataCollection.from_legacy_df` — expect,
    starting from a raw OpenML-style frame (needs ``NumberOfInstances``/``NumberOfFeatures``/
    ``NumberOfClasses``/``name``). Used to make the OpenML ``generate_task_metadata`` fallback
    convertible to a ``TaskMetadataCollection`` (see :func:`task_metadata_collection_from_openml`).
    """
    task_metadata["n_folds"] = 3
    task_metadata["n_repeats"] = task_metadata["NumberOfInstances"].apply(_get_n_repeats)
    task_metadata["n_samples_test_per_fold"] = (task_metadata["NumberOfInstances"] / task_metadata["n_folds"]).astype(
        int
    )
    task_metadata["n_samples_train_per_fold"] = (
        task_metadata["NumberOfInstances"] - task_metadata["n_samples_test_per_fold"]
    ).astype(int)
    return add_extra_task_metadata_info(task_metadata=task_metadata)


def task_metadata_collection_from_openml(tids: list[int], *, verbose: bool = True) -> TaskMetadataCollection:
    """Build a (lossy) ``TaskMetadataCollection`` for a list of OpenML task ids.

    Prefers TabArena's cached committed task metadata; for any id not in it, falls back to
    live OpenML (via :func:`~tabarena.nips2025_utils.method_processor.generate_task_metadata`,
    enriched to the legacy schema by :func:`enrich_legacy_task_metadata`). The legacy frame is
    wrapped via :meth:`TaskMetadataCollection.from_legacy_df`, which is lossy (rich fields become
    ``None`` and per-fold sizes are the recorded averages) but sufficient for running and
    comparing models on arbitrary OpenML tasks.

    Note: when *every* requested id is already cached, the full cached collection is returned;
    filter it to the requested tasks with ``.subset_tasks(task_ids=tids)``.
    """
    from tabarena.benchmark.task.metadata import TaskMetadataCollection, default_task_metadata_collection

    log = print if verbose else (lambda *a, **k: None)
    cached = default_task_metadata_collection()
    tids_cached = set(cached.to_legacy_df()["tid"].unique())

    tids_missing = [tid for tid in tids if tid not in tids_cached]
    if not tids_missing:
        return cached

    from tabarena.nips2025_utils.method_processor import generate_task_metadata

    log(f"Note: Missing {len(tids_missing)} tasks in the cached task_metadata...")
    log("\tFetching task_metadata from OpenML... (this may take ~1 minute)")
    task_metadata = enrich_legacy_task_metadata(generate_task_metadata(tids=tids))
    return TaskMetadataCollection.from_legacy_df(task_metadata)


def load_curated_task_metadata() -> pd.DataFrame:
    """Load the curated metadata for the TabArena datasets.

    Original file (and future version), can be found here: https://github.com/TabArena/tabarena_dataset_curation/tree/main/dataset_creation_scripts/metadata

    TODO: align the ``num_features`` convention with the schema upstream in
    ``tabarena_dataset_curation`` (define it to exclude the target, or rename it to e.g.
    ``num_columns``) so consumers don't each have to drop the target column. Until then,
    the conversion to ``TabArenaTaskMetadata`` happens at the boundary in
    ``benchmark.task.metadata.sources.tabarena_v0pt1.load_tabarena_v0_1_task_metadata``.

    The metadata requires the following columns per task (per row) to schedule tasks:
        "tabarena_num_repeats": int
            The number of repeats for the task based on the protocol from TabArena.
            See tabarena.nips2025_utils.fetch_metadata._get_n_repeats for details.
        "num_folds": int
            The number of folds for the task.
        "task_id": str
            The task ID for the task as an int.
            If a local task, we assume this to be `UserTask.task_id_str`.
        "num_instances": int
            The number of instances/samples in the dataset.
        "num_features" : int
            The number of features in the dataset, **including the target column**
            (OpenML ``NumberOfFeatures`` convention). Subtract 1 for the true feature
            count; ``TabArenaTaskMetadata.num_features`` excludes the target.
        "num_classes": int
            The number of classes in the dataset. For regression tasks, this value is
            ignored.
        "problem_type": str
            The problem type of the task. Options: "binary", "regression", "multiclass"
    """
    path = str(Path(__file__).parent.resolve() / "metadata" / "curated_tabarena_dataset_metadata.csv")

    return load_pd.load(path=path)
