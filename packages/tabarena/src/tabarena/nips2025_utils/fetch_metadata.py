from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from autogluon.common.loaders import load_pd

import tabarena

if TYPE_CHECKING:
    import pandas as pd


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


def load_task_metadata(paper: bool = True, subset: str | None = None, path: str | None = None) -> pd.DataFrame:
    """Load the task metadata for all datasets in the TabArena benchmark.

    Parameters
    ----------
    paper: bool, default True
        If True, returns the task_metadata for the 51 NeurIPS 2025 TabArena paper datasets
        If False, returns the task_metadata for 61 datasets prior to filtering down to 51 datasets for the paper.
    subset: {None, "TabPFNv2", "TabICL"}, default None
        If None, returns all tasks.
        If "TabPFNv2", filters to tasks compatible with TabPFNv2:
            <=10k samples, <=500 features, <=10 classes
            33/51 datasets
        If "TabICL", filters to tasks compatible with TabICL:
            <=100k samples, <=500 features, classification
            36/51 datasets

    Returns:
    -------
    task_metadata: pd.DataFrame
        Metadata about each dataset in the benchmark.

    """
    if path is None:
        tabrepo_root = str(Path(tabarena.__file__).parent.parent)
        if paper:
            path = f"{tabrepo_root}/tabarena/nips2025_utils/metadata/task_metadata_tabarena51.csv"
        else:
            raise ValueError("paper == True is required")
    task_metadata = load_pd.load(path=path)
    task_metadata = enrich_legacy_task_metadata(task_metadata)

    return subset_task_metadata(task_metadata=task_metadata, subset=subset)


def subset_task_metadata(task_metadata, subset: str | None = None) -> pd.DataFrame:
    if subset is None:
        pass
    elif subset == "TabPFNv2":
        task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] <= 10000]
        task_metadata = task_metadata[task_metadata["n_features"] <= 500]
        task_metadata = task_metadata[task_metadata["NumberOfClasses"] <= 10]
    elif subset == "TabICL":
        task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] <= 100000]
        task_metadata = task_metadata[task_metadata["n_features"] <= 500]
        task_metadata = task_metadata[task_metadata["NumberOfClasses"] > 0]
    else:
        raise AssertionError(f"Unknown subset: {subset}")
    return task_metadata


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
    ``NumberOfClasses``/``name``). Used by :func:`load_task_metadata` and to make the OpenML
    ``generate_task_metadata`` fallback convertible to a ``TaskMetadataCollection``.
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
