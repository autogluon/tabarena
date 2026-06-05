"""Task metadata source for the built-in TabArena v0.1 benchmark suite.

Aligned with the Data Foundry sources: the source loads a *committed* reference CSV
(per-task x split, the :meth:`TabArenaTaskMetadata.to_dataframe` format) shipped as
package data, instead of rebuilding it from the curated metadata at runtime.

:func:`load_tabarena_v0_1_task_metadata` (the on-the-fly rebuild) is kept for
backward compatibility and is what :func:`generate_tabarena_v0_1_reference_metadata`
(used by a maintainer script) runs to (re)produce the committed CSV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabarena.benchmark.task.metadata.schema import SplitMetadata, TabArenaTaskMetadata, derive_task_type
from tabarena.benchmark.task.metadata.sources.base import committed_metadata_path as _committed_metadata_path
from tabarena.benchmark.task.metadata.sources.openml import OpenMLTaskMetadataSource

#: Surface suite name used for the committed reference CSV and the source registry.
TABARENA_V0PT1_NAME = "TabArena-v0.1"


def committed_metadata_path() -> Path:
    """Path to the git-committed v0.1 reference-metadata CSV (package data)."""
    return _committed_metadata_path(TABARENA_V0PT1_NAME)


def load_tabarena_v0_1_task_metadata(curated_metadata: pd.DataFrame) -> list[TabArenaTaskMetadata]:
    """Convert curated TabArena v0.1 metadata into the current TabArenaTaskMetadata format.

    Produces one entry per task with splits unrolled. This is the backward-compatible
    rebuild; prefer the committed CSV via :class:`TabArenaV0pt1TaskMetadataSource`. It is
    also what :func:`generate_tabarena_v0_1_reference_metadata` runs to (re)build the CSV.

    Args:
        curated_metadata: The curated v0.1 metadata table (one row per task), e.g. the
            output of ``tabarena.nips2025_utils.fetch_metadata.load_curated_task_metadata``.
            Must expose the columns accessed below (``num_classes``, ``num_instances``,
            ``tabarena_num_repeats``, ``num_folds``, ``problem_type``, ``task_id``, ...).
    """
    print("Converting TabArena v0.1 curated metadata to the new TabArenaTaskMetadata format...")

    metric_map = {
        "binary": "roc_auc",
        "multiclass": "log_loss",
        "regression": "rmse",
    }

    task_metadata: list[TabArenaTaskMetadata] = []
    for row in curated_metadata.itertuples():
        # Regression uses the schema's -1 convention (the curated table leaves num_classes
        # unset for regression); classification keeps the curated class count, cast to int
        # (it loads as float). Drives both the static field and the per-split counts.
        num_classes = -1 if row.problem_type == "regression" else int(row.num_classes)
        num_instances = row.num_instances
        # Curated `num_features` is OpenML's NumberOfFeatures, which counts the target
        # column and loads as float. TabArenaTaskMetadata.num_features is an int feature
        # count excluding the target (matching legacy `n_features = NumberOfFeatures - 1`),
        # so cast to int and drop the target here.
        num_features = int(row.num_features) - 1

        n_repeats = row.tabarena_num_repeats
        n_folds = row.num_folds

        eval_metric = metric_map[row.problem_type]

        # Warehouse-level metadata, present in the curated table but optional for
        # other curated sources — fall back to None when a column is absent.
        domain = getattr(row, "domain", None)
        year = getattr(row, "year", None)
        data_source = getattr(row, "data_source", None)
        dataset_year = None if year is None or pd.isna(year) else str(year)

        for repeat_i in range(n_repeats):
            for fold_i in range(n_folds):
                split_index = SplitMetadata.get_split_index(repeat_i=repeat_i, fold_i=fold_i)
                splits_metadata = {
                    split_index: SplitMetadata(
                        repeat=repeat_i,
                        fold=fold_i,
                        num_instances_train=num_instances * 2 / 3,
                        num_instances_test=num_instances * 1 / 3,
                        num_instance_groups_train=num_instances * 2 / 3,
                        num_instance_groups_test=num_instances * 1 / 3,
                        num_classes_train=num_classes,
                        num_classes_test=num_classes,
                        num_features_train=num_features,
                        num_features_test=num_features,
                    ),
                }

                task_metadata.append(
                    TabArenaTaskMetadata(
                        task_id_str=row.task_id,
                        dataset_name=row.dataset_name,
                        tabarena_task_name=row.dataset_name,
                        problem_type=row.problem_type,
                        is_classification=row.is_classification,
                        target_name=row.target_feature,
                        stratify_on=row.target_feature if row.is_classification else None,
                        split_time_horizon=None,
                        split_time_horizon_unit=None,
                        time_on=None,
                        group_on=None,
                        group_time_on=None,
                        group_labels=None,
                        multiclass_max_n_classes_over_splits=num_classes,
                        multiclass_min_n_classes_over_splits=num_classes,
                        class_consistency_over_splits=True,
                        num_instances=num_instances,
                        num_features=num_features,
                        num_instance_groups=num_instances,
                        num_classes=num_classes,
                        splits_metadata=splits_metadata,
                        eval_metric=eval_metric,
                        # Warehouse-level metadata available from the curated table.
                        # v0.1 is IID, and dataset-derived dtype stats (text / cardinality /
                        # cols-after-preprocessing / missing fraction) are not in the curated
                        # table, so they remain None (no dataset is loaded here).
                        task_type=derive_task_type(time_on=None, group_on=None),
                        domain=domain,
                        dataset_year=dataset_year,
                        source=data_source,
                    ),
                )
    return task_metadata


def generate_tabarena_v0_1_reference_metadata(
    out_path: str | Path | None = None,
    *,
    curated_metadata: pd.DataFrame | None = None,
) -> Path:
    """Write the committed v0.1 reference CSV by running the rebuild (maintainer tool).

    Args:
        out_path: Where to write the CSV. Defaults to :func:`committed_metadata_path`.
        curated_metadata: Curated v0.1 metadata to convert. Defaults to the built-in
            ``tabarena.nips2025_utils.fetch_metadata.load_curated_task_metadata()``.

    Returns:
        The path the CSV was written to.
    """
    if curated_metadata is None:
        from tabarena.nips2025_utils.fetch_metadata import load_curated_task_metadata

        curated_metadata = load_curated_task_metadata()

    task_metadata = load_tabarena_v0_1_task_metadata(curated_metadata)
    metadata_df = pd.concat([ttm.to_dataframe() for ttm in task_metadata], ignore_index=True)
    out_path = Path(out_path) if out_path is not None else committed_metadata_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(metadata_df)} reference-metadata rows to {out_path}.")
    metadata_df.to_csv(out_path, index=False)
    return out_path


class TabArenaV0pt1TaskMetadataSource(OpenMLTaskMetadataSource):
    """Source for the built-in TabArena v0.1 curated task metadata.

    Loads the committed reference CSV (package data); falls back to rebuilding from
    the curated metadata if that file is missing. ``materialize`` (inherited from
    :class:`OpenMLTaskMetadataSource`) pre-caches each task's OpenML dataset + splits,
    since every v0.1 task is identified by an integer OpenML task id.
    """

    def load(self, *, verbose: bool = False) -> list[TabArenaTaskMetadata]:
        """Load v0.1 task metadata from the committed CSV (or rebuild as a fallback)."""
        path = committed_metadata_path()
        if path.exists():
            if verbose:
                print(f"Loading committed TabArena v0.1 reference metadata from {path}.")
            metadata_df = pd.read_csv(path)
            return [TabArenaTaskMetadata.from_row(row) for _, row in metadata_df.iterrows()]

        from tabarena.nips2025_utils.fetch_metadata import load_curated_task_metadata

        return load_tabarena_v0_1_task_metadata(load_curated_task_metadata())
