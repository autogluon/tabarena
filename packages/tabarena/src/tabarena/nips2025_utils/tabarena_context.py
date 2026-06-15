from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Literal

import pandas as pd

from tabarena.nips2025_utils.abstract_arena_context import AbstractArenaContext
from tabarena.nips2025_utils.artifacts import tabarena_method_metadata_collection
from tabarena.nips2025_utils.eval_all import evaluate_all
from tabarena.nips2025_utils.subset_predicate import SubsetPredicate

if TYPE_CHECKING:
    from pathlib import Path

    from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
    from tabarena.models._method_metadata import MethodMetadata

_methods_paper = [
    "AutoGluon_v140_bq_4h8c",
    # "AutoGluon_v140_eq_4h8c",
    "AutoGluon_v150_eq_4h8c",
    # "Portfolio-N200-4h",
    "CatBoost",
    # "Dummy",
    "ExplainableBM",
    "ExtraTrees",
    "KNeighbors",
    "LightGBM",
    "LinearModel",
    # "ModernNCA",
    "NeuralNetFastAI",
    "NeuralNetTorch",
    "RandomForest",
    # "RealMLP",
    # "TabM",
    "XGBoost",
    "Mitra_GPU",
    "ModernNCA_GPU",
    "RealMLP_GPU",
    "TabDPT_GPU",
    "TabICL_GPU",
    "TabM_GPU",
    "TabPFNv2_GPU",
    "xRFM_GPU",
    "BetaTabPFN_GPU",
    "TabFlex_GPU",
    "RealTabPFN-v2.5",
    "SAP-RPT-OSS",
    "TabICLv2",
    "TabSTAR",
    "PerpetualBooster",
    "TabPFN-v2.6",
    "LimiX",
    "OrionMSP",
    "TabPFN-3",
    "iLTM",
]


class TabArenaContext(AbstractArenaContext):
    """Reference arena context: TabArena v0.1 task/method presets + the paper workflow.

    Implements the :class:`AbstractArenaContext` hooks against the committed TabArena v0.1
    suite and the paper's method metadata (so :meth:`load_results` loads the paper baseline
    results), and adds the paper's ``evaluate_all`` reproduction workflow.
    """

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
    ):
        super().__init__(
            methods=methods,
            task_metadata=task_metadata,
            extra_methods=extra_methods,
            backend=backend,
            fillna_method=fillna_method,
            calibration_method=calibration_method,
            only_valid_tasks=only_valid_tasks,
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
        methods = copy.deepcopy(_methods_paper)
        method_metadata_lst: list[MethodMetadata] = copy.deepcopy(
            tabarena_method_metadata_collection.method_metadata_lst,
        )
        method_metadata_lst = [m for m in method_metadata_lst if m.method in methods]
        method_metadata_name_set = {m.method for m in method_metadata_lst}
        methods = [m for m in methods if m in method_metadata_name_set]
        return [m for m in method_metadata_lst if m.method in set(methods)]

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
