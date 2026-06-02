"""Convert Data Foundry curated containers into TabArena UserTasks.

This module is the TabArena-side bridge for the ``data_foundry`` package. It takes
:class:`data_foundry.curation_container.CuratedContainer` objects (typically obtained
by iterating a :class:`data_foundry.collections.DatasetCollection`) and produces
TabArena :class:`UserTask` instances together with a metadata DataFrame describing them.

Caching, downloading, and source-resolution (Hugging Face vs. local warehouse) are
delegated to ``data_foundry``: the collection's :class:`~data_foundry.collections.DataSource`
handles all of that. This adapter only owns the TabArena-specific conversion logic:

* problem-type / target-dtype validation,
* eval-metric fallback against TabArena's supported metric set per problem type,
* assembling the per-task metadata into a single DataFrame keyed by ``data_foundry_uri``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger
from tqdm import tqdm

from tabarena.benchmark.task.user_task import UserTask

if TYPE_CHECKING:
    from data_foundry.collections import DatasetCollection
    from data_foundry.curation_container import CuratedContainer

DEFAULT_EVAL_METRICS: dict[str, list[str]] = {
    "binary_classification": ["roc_auc"],
    "multiclass_classification": ["log_loss"],
    "regression": ["root_mean_squared_error"],
}
"""Allowed TabArena eval metrics per Data Foundry ``problem_type``.

If a curated container's ``objective_metric_name`` is not in the allowed list for its
problem type, the first metric in the list is used as a fallback. Override by passing
``evaluation_metrics=...`` to the adapter or converter.
"""


class DataFoundryAdapter:
    """Convert every container in a Data Foundry collection into a TabArena UserTask.

    The adapter is a thin driver around a :class:`DatasetCollection`: it walks the
    collection (downloading via its ``source`` when needed) and converts each
    :class:`CuratedContainer` into a local OpenML task via :class:`UserTask`. The
    resulting ``UserTask`` is persisted in the OpenML cache (see
    :meth:`UserTask.save_local_openml_task`) and one metadata row per task is collected
    into a DataFrame.

    Example:
        >>> from data_foundry.collections import BEYOND_ARENA
        >>> adapter = DataFoundryAdapter(collection=BEYOND_ARENA)
        >>> metadata_df = adapter.to_tabarena_user_tasks()

        Offline use with a pre-populated local warehouse::

            from data_foundry.collections import DatasetCollection, LocalWarehouseSource

            collection = DatasetCollection.from_relative_paths(
                name="beyond_iid_benchmark_2026",
                description="...",
                relative_paths=["musk/<uuid>", "mercedes_benz_.../<uuid>", ...],
                source=LocalWarehouseSource(base_dir=Path("/.../local-data-warehouse")),
            )
            adapter = DataFoundryAdapter(collection=collection)
            metadata_df = adapter.to_tabarena_user_tasks()
    """

    def __init__(
        self,
        *,
        collection: DatasetCollection,
        evaluation_metrics: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            collection: The Data Foundry collection to convert. Must have a
                :class:`~data_foundry.collections.DataSource` configured (either a
                ``HuggingFaceSource`` for online access, a ``LocalWarehouseSource``
                pointing at a pre-populated directory, or any custom source).
            evaluation_metrics: Mapping from Data Foundry ``problem_type`` to the list of
                eval metrics TabArena is willing to use for that problem type. If a
                container's ``objective_metric_name`` is not in the allowed list, the
                first metric in the list is used as a fallback. ``None`` uses
                :data:`DEFAULT_EVAL_METRICS`.
        """
        if collection.source is None:
            raise ValueError(
                f"Collection {collection.name!r} has no `source` configured. "
                "Wire a HuggingFaceSource (online), LocalWarehouseSource (offline), "
                "or other DataSource subclass before passing the collection to the adapter.",
            )
        self.collection = collection
        self.evaluation_metrics = deepcopy(DEFAULT_EVAL_METRICS) if evaluation_metrics is None else evaluation_metrics

    def to_tabarena_user_tasks(
        self,
        *,
        cache_dir: str | None = None,
        force_download: bool = False,
        show_sample: bool = False,
    ) -> pd.DataFrame:
        """Convert every container in the collection into a TabArena UserTask.

        Args:
            cache_dir: Optional override for the data_foundry cache directory used by the
                collection's :class:`DataSource`. Forwarded as-is to
                :meth:`DatasetCollection.iter_containers`.
            force_download: When ``True``, bypass any cached copy and re-fetch every
                container from its source. Forwarded to ``iter_containers``.
            show_sample: When ``True``, prints a 5-row sample of the resulting metadata
                DataFrame for quick visual inspection.

        Returns:
            A DataFrame with one row per ``(task, split)`` combination, containing all
            fields from :class:`TabArenaTaskMetadata`/:class:`SplitMetadata` plus a
            ``data_foundry_uri`` column with the relative path used to identify the
            container in the source collection.
        """
        rows: list[pd.DataFrame] = []

        iterator = self.collection.iter_containers(
            cache_dir=cache_dir,
            force_download=force_download,
        )
        for entry, container in tqdm(
            zip(self.collection.entries, iterator, strict=True),
            total=len(self.collection),
            desc=f"Converting {self.collection.name} containers to TabArena UserTasks",
        ):
            user_task = convert_curated_container_to_user_task(
                container=container,
                evaluation_metrics=self.evaluation_metrics,
            )
            oml_task = user_task.load_local_openml_task()

            task_metadata = oml_task.compute_metadata(
                tabarena_task_name=user_task.tabarena_task_name,
                task_id_str=user_task.task_id_str,
            )
            task_metadata.data_foundry_uri = entry.relative_path.as_posix()
            # Warehouse-level metadata only available from the Data Foundry container.
            task_metadata.domain = container.dataset_metadata.domain_str
            task_metadata.dataset_year = container.dataset_metadata.dataset_year
            task_metadata.source = container.dataset_metadata.dataset_source
            rows.append(task_metadata.to_dataframe())

        metadata_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

        if show_sample and not metadata_df.empty:
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                None,
                "display.max_colwidth",
                None,
            ):
                print(metadata_df.sample(min(5, len(metadata_df)), random_state=42))

        return metadata_df


def convert_curated_container_to_user_task(  # noqa: C901 — linear problem-type validation
    *,
    container: CuratedContainer,
    evaluation_metrics: dict[str, list[str]] | None = None,
) -> UserTask:
    """Convert a single Data Foundry curated container into a TabArena UserTask.

    Validates the target column matches the declared ``problem_type``, resolves the
    eval metric against the allowed list (falling back to the first allowed metric
    when the container's metric is not allowed), and persists the resulting
    ``UserTask`` to the OpenML cache so it can be reloaded later via
    :meth:`UserTask.load_local_openml_task`.

    Args:
        container: An already-loaded :class:`CuratedContainer` (e.g. yielded by
            :meth:`DatasetCollection.iter_containers` or returned by
            :meth:`DatasetCollection.get_dataset`).
        evaluation_metrics: Mapping from Data Foundry ``problem_type`` to the list of
            eval metrics TabArena is willing to use. ``None`` skips fallback and uses
            whatever the container declares.

    Returns:
        The persisted TabArena :class:`UserTask`.
    """
    target_name = container.task_metadata.target_column_name
    y: pd.Series = container.dataset[target_name]
    df_problem_type = container.task_metadata.problem_type
    unique_name = container.dataset_metadata.unique_name

    if df_problem_type == "regression":
        problem_type = "regression"
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(
                f"Target column {target_name!r} is not numeric for regression problem. ({unique_name})",
            )
    elif df_problem_type == "binary_classification":
        problem_type = "classification"
        if not isinstance(y.dtype, pd.CategoricalDtype):
            raise ValueError(
                f"Target column {target_name!r} is not categorical for classification problem. ({unique_name})",
            )
        if y.nunique() != 2:
            raise ValueError(
                f"Target column {target_name!r} has {y.nunique()} classes, "
                f"but expected 2 for binary classification problem. ({unique_name})",
            )
    elif df_problem_type == "multiclass_classification":
        problem_type = "classification"
        if not isinstance(y.dtype, pd.CategoricalDtype):
            raise ValueError(
                f"Target column {target_name!r} is not categorical for classification problem. ({unique_name})",
            )
        if y.nunique() < 3:
            raise ValueError(
                f"Target column {target_name!r} has {y.nunique()} classes, "
                f"but expected at least 3 for multiclass classification "
                f"problem. ({unique_name})",
            )
    else:
        raise ValueError(f"Unknown problem type {df_problem_type!r}")

    # Eval-metric fallback against TabArena's allowed set.
    eval_metric = container.task_metadata.objective_metric_name
    if evaluation_metrics is not None:
        allowed = evaluation_metrics[df_problem_type]
        if eval_metric not in allowed:
            fallback = allowed[0]
            logger.debug(
                f"Objective metric {eval_metric!r} not allowed for problem type "
                f"{problem_type!r}. Falling back to {fallback!r}.",
            )
            eval_metric = fallback

    user_task = UserTask(task_name=container.unique_name)
    oml_task = user_task.create_local_openml_task(
        dataset=container.dataset,
        target_feature=target_name,
        problem_type=problem_type,
        splits=container.experiment_metadata.splits,
        eval_metric=eval_metric,
        stratify_on=container.task_metadata.stratify_on,
        group_on=container.task_metadata.group_on,
        time_on=container.task_metadata.time_on,
        group_time_on=container.task_metadata.group_time_on,
        dataset_name=container.dataset_metadata.unique_name,
        group_labels=container.task_metadata.group_labels,
        split_time_horizon=container.experiment_metadata.time_horizon,
        split_time_horizon_unit=container.experiment_metadata.time_horizon_unit,
    )
    user_task.save_local_openml_task(oml_task)

    # The container ships its semantic-text embedding cache as an extra artifact; copy it into the
    # canonical tabarena cache so the fit-time loader finds it (no-op when the container has none).
    from tabarena.benchmark.task.data_foundry.text_cache import import_text_cache_from_container

    import_text_cache_from_container(container, user_task.slug)

    return user_task
