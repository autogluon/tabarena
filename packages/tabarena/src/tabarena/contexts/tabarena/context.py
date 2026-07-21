from __future__ import annotations

import copy
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import pandas as pd

from tabarena.benchmark.task.subset_predicate import SubsetPredicate
from tabarena.contexts import AbstractArenaContext
from tabarena.contexts.tabarena.methods import tabarena_method_metadata_collection
from tabarena.nips2025_utils.eval_all import evaluate_all

if TYPE_CHECKING:
    from pathlib import Path

    from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
    from tabarena.caching import CacheConfig
    from tabarena.models._method_metadata import MethodMetadata

#: Inclusive boundary on ``percentage_cat_features`` (the fraction of a dataset's features that are
#: categorical) separating the ``low_cats`` bucket (at or below it) from ``high_cats`` (above it).
#: 50 splits the TabArena v0.1 suite into 36 low-cat and 15 high-cat datasets.
CAT_FRACTION_THRESHOLD = 50.0


@lru_cache(maxsize=1)
def _datasets_by_cat_fraction() -> tuple[frozenset[str], frozenset[str]]:
    """``(low_cat, high_cat)`` dataset-name sets, split on ``percentage_cat_features``.

    TabArena v0.1's native task grid carries no categorical-feature counts (the warehouse
    ``num_*_cats`` fields are ``None`` for v0.1 tasks), so the split is sourced from the committed
    curated metadata CSV and joined to the grid on the ``dataset`` name (the curated
    ``dataset_name`` matches the grid ``dataset`` 1:1). Loaded once on first use.
    """
    from tabarena.benchmark.task.metadata.fetch_metadata import load_curated_task_metadata

    df = load_curated_task_metadata()
    is_low = df["percentage_cat_features"] <= CAT_FRACTION_THRESHOLD
    low = frozenset(df.loc[is_low, "dataset_name"].astype(str))
    high = frozenset(df.loc[~is_low, "dataset_name"].astype(str))
    return low, high


@lru_cache(maxsize=1)
def _numerical_only_datasets() -> frozenset[str]:
    """Dataset names with a purely numerical feature space (``percentage_cat_features == 0``).

    Like :func:`_datasets_by_cat_fraction`, sourced from the committed curated metadata CSV
    because the v0.1 task grid carries no categorical-feature counts. Loaded once on first use.
    """
    from tabarena.benchmark.task.metadata.fetch_metadata import load_curated_task_metadata

    df = load_curated_task_metadata()
    return frozenset(df.loc[df["percentage_cat_features"] == 0, "dataset_name"].astype(str))


def _numerical_only_predicate() -> SubsetPredicate:
    """A :class:`SubsetPredicate` keeping grid rows whose dataset has only numerical features.

    Data-dependent like :func:`_cat_fraction_predicate`: closes over the curated-metadata
    selection, so it only requires the always-present ``dataset`` column.
    """

    def _mask(df: pd.DataFrame) -> pd.Series:
        return df["dataset"].astype(str).isin(_numerical_only_datasets())

    return SubsetPredicate(_mask, ("dataset",))


def _cat_fraction_predicate(*, low: bool) -> SubsetPredicate:
    """A :class:`SubsetPredicate` keeping grid rows whose dataset is in the low/high cat bucket.

    Data-dependent (like :func:`~tabarena.benchmark.task.subset_predicate.tasks_in_frame`): it
    closes over the curated-metadata split rather than reading a grid column, so it only requires
    the always-present ``dataset`` column.
    """

    def _mask(df: pd.DataFrame) -> pd.Series:
        low_set, high_set = _datasets_by_cat_fraction()
        keep = low_set if low else high_set
        return df["dataset"].astype(str).isin(keep)

    return SubsetPredicate(_mask, ("dataset",))


class TabArenaContext(AbstractArenaContext):
    """Reference arena context: TabArena v0.1 task/method presets + the paper workflow.

    Implements the :class:`AbstractArenaContext` hooks against the committed TabArena v0.1
    suite and the paper's method metadata (so :meth:`load_results` loads the paper baseline
    results), and adds the paper's ``evaluate_all`` reproduction workflow.
    """

    benchmark_name: str = "TabArena"

    SUBSET_PREDICATES: dict[str, SubsetPredicate] = {
        "all": SubsetPredicate(lambda df: pd.Series(True, index=df.index)),
        # problem_type
        "binary": SubsetPredicate(lambda df: df["problem_type"] == "binary", ("problem_type",)),
        "multiclass": SubsetPredicate(lambda df: df["problem_type"] == "multiclass", ("problem_type",)),
        "classification": SubsetPredicate(
            lambda df: df["problem_type"].isin(["binary", "multiclass"]), ("problem_type",)
        ),
        "regression": SubsetPredicate(lambda df: df["problem_type"] == "regression", ("problem_type",)),
        # size buckets keyed on training rows
        "medium": SubsetPredicate(lambda df: df["max_train_rows"].between(10_001, 100_000), ("max_train_rows",)),
        "small": SubsetPredicate(lambda df: df["max_train_rows"] <= 10_000, ("max_train_rows",)),
        "tiny": SubsetPredicate(lambda df: df["max_train_rows"] <= 2_000, ("max_train_rows",)),
        # foundation-model compatibility (operates on tabarena task_metadata columns)
        "tabpfn": SubsetPredicate(
            lambda df: (df["max_train_rows"] <= 10_000) & (df["n_features"] <= 500) & (df["n_classes"] <= 10),
            ("max_train_rows", "n_features", "n_classes"),
        ),
        "tabicl": SubsetPredicate(
            lambda df: (df["max_train_rows"] <= 100_000) & (df["n_features"] <= 500) & (df["n_classes"] > 0),
            ("max_train_rows", "n_features", "n_classes"),
        ),
        # categorical-feature fraction: the percentage of features that are categorical, taken from
        # the curated metadata (the v0.1 grid carries no cat counts natively) and keyed on the
        # dataset name. Complementary split at CAT_FRACTION_THRESHOLD (50%): low_cats has 36
        # datasets, high_cats 15.
        "low_cats": _cat_fraction_predicate(low=True),
        "high_cats": _cat_fraction_predicate(low=False),
        # purely numerical feature space (percentage_cat_features == 0); a subset of low_cats.
        "numerical": _numerical_only_predicate(),
        # split-level filter: keeps split 0 == (fold 0, repeat 0). Evaluated on the task grid's
        # "split" column (see TaskMetadataCollection.task_grid); a results frame's "fold" is the
        # split, so this maps to fold == 0 there.
        "lite": SubsetPredicate(lambda df: df["split"] == 0, ("split",)),
    }

    def __init__(
        self,
        methods: str | list[MethodMetadata] = "tabarena",
        task_metadata: str | TaskMetadataCollection = "tabarena",
        *,
        extra_methods: list[MethodMetadata] | None = None,
        backend: Literal["ray", "native"] = "ray",
        fillna_method: str | None = "RF (default)",
        calibration_method: str | None = "RF (default)",
        only_valid_tasks: bool = False,
        cache_config: CacheConfig | None = None,
    ):
        super().__init__(
            methods=methods,
            task_metadata=task_metadata,
            extra_methods=extra_methods,
            backend=backend,
            fillna_method=fillna_method,
            calibration_method=calibration_method,
            only_valid_tasks=only_valid_tasks,
            cache_config=cache_config,
        )

    def _resolve_task_metadata_preset(self, name: str) -> TaskMetadataCollection:
        if name != "tabarena":
            raise ValueError(f"Unknown task_metadata preset {name!r}; expected 'tabarena'.")
        # Native default: the committed TabArena v0.1 suite (metadata only, no downloads).
        from tabarena.benchmark.task.metadata import TaskMetadataCollection

        return TaskMetadataCollection.from_preset("TabArena-v0.1")

    def _resolve_methods_preset(self, name: str) -> list[MethodMetadata]:
        if name != "tabarena":
            raise ValueError(f"Unknown methods preset '{name}'.")
        # `tabarena_method_metadata_collection` already holds exactly the paper method set.
        return copy.deepcopy(tabarena_method_metadata_collection.method_metadata_lst)

    @property
    def _default_subsets(self):
        return [
            [],
            ["tiny"],
            ["small"],
            ["medium"],
            ["binary"],
            ["multiclass"],
            ["classification"],
            ["regression"],
        ]

    def evaluate_all(
        self,
        save_path: str | Path,
        df_results: pd.DataFrame = None,
        df_results_cpu: pd.DataFrame = None,
        configs_hyperparameters: dict[str, dict] | None = None,
        include_portfolio: bool = False,
        elo_bootstrap_rounds: int = 200,
        use_latex: bool = False,
        fillna_method: str | None = "auto",
        use_website_folder_names: bool = False,
        evaluator_kwargs: dict | None = None,
        engine: str = "auto",
        progress_bar: bool = True,
    ):
        if df_results is None:
            df_results = self.load_results(download_results="auto")

        if fillna_method == "auto":
            fillna_method = self.fillna_method
        if fillna_method is not None:
            df_results = self.fillna_metrics(
                df_to_fill=df_results,
                df_fillna=df_results[df_results["method"] == fillna_method],
            )

        evaluate_all(
            tabarena_context=self,  # FIXME: Don't do this in future, clean up
            df_results=df_results,
            # configs_hyperparameters=configs_hyperparameters,  # TODO: Add back later
            eval_save_path=save_path,
            elo_bootstrap_rounds=elo_bootstrap_rounds,
            use_latex=use_latex,
            use_website_folder_names=use_website_folder_names,
            evaluator_kwargs=evaluator_kwargs,
            engine=engine,
            progress_bar=progress_bar,
        )
