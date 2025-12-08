from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path
import re

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import ticker
from tueplots import bundles, fonts, fontsizes
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

from autogluon.common.savers import save_pd

from tabarena.utils.normalized_scorer import NormalizedScorer
from tabarena.nips2025_utils.fetch_metadata import load_task_metadata
from bencheval.tabarena import TabArena
from tabarena.paper.paper_utils import get_framework_type_method_names, get_method_rename_map, get_f_map_suffix_plots
from tabarena.plot.dataset_analysis import plot_train_time_deep_dive
from tabarena.plot.plot_ens_weights import create_heatmap
from tabarena.plot.plot_pareto_frontier import plot_pareto as _plot_pareto, plot_pareto_aggregated

matplotlib.rcParams.update(fontsizes.neurips2024())
matplotlib.rcParams.update({
    'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{xcolor}'
})


def darken_color(color_str, amount=0.5):
    # Convert color string to RGB tuple (values between 0 and 1)
    rgb = mcolors.to_rgb(color_str)
    # Interpolate with black (0, 0, 0)
    darker_rgb = tuple((1 - amount) * c for c in rgb)
    return darker_rgb


# FIXME: ensemble weights can get if including `config_hyperparameters` as input
class TabArenaEvaluator:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        task_metadata: pd.DataFrame | None = None,
        config_types: dict[str, str] = None,
        method_col: str = "method",
        error_col: str = "metric_error",
        methods: list[str] | None = None,
        folds: list[int] | None = None,
        datasets: list[str] | None = None,
        problem_types: list[str] | None = None,
        banned_model_types: list[str] | None = None,
        banned_pareto_methods: list[str] | None = None,
        elo_bootstrap_rounds: int = 200,
        elo_ymin: float = 800,
        keep_best: bool = False,
        figure_file_type: str = "pdf",
        use_latex: bool = False,
        tabarena_context=None,  # FIXME: Remove this and refactor after leaderboard v0.2 upload, this is purely to get things working fast
    ):
        """

        Parameters
        ----------
        methods
            filter methods
        folds
            filter folds
        datasets
            filter datasets
        problem_types
            filter problem_types
        elo_bootstrap_rounds
            10 = toy
            100 = paper
        kwargs
        """
        if task_metadata is None:
            task_metadata = load_task_metadata()
        if banned_pareto_methods is None:
            banned_pareto_methods = []
        self.output_dir: Path = Path(output_dir)
        self.task_metadata = task_metadata
        self.method_col = method_col
        self.error_col = error_col
        self.config_types = config_types
        self.figure_file_type = figure_file_type
        self.banned_pareto_methods = banned_pareto_methods

        self.datasets = datasets
        self.problem_types = problem_types
        self.methods = methods
        self.folds = folds
        self.elo_bootstrap_rounds = elo_bootstrap_rounds
        self.elo_ymin = elo_ymin
        self.banned_model_types = banned_model_types
        self.keep_best = keep_best

        self.use_latex = use_latex
        if self.use_latex:
            matplotlib.rcParams.update(bundles.neurips2024())
            matplotlib.rcParams.update(fonts.neurips2024_tex())
            self.rc_context_params = {
                                    'font.family': 'serif',
                                    "text.usetex": True,
                                } | fontsizes.neurips2024(default_smaller=0)
        else:
            self.rc_context_params = {}

        self.style_order = [
            "Default",
            "Tuned",
            "Tuned + Ens.",
            "Baseline",
            "Best",
            "Default, Holdout",
            "Tuned, Holdout",
            "Tuned + Ens., Holdout",
        ]

        self.style_markers = {
            "Default": "o",
            "Tuned": "s",
            "Tuned + Ens.": "X",
            "Baseline": "D",
            "Best": "*",
            "Default, Holdout": "^",
            "Tuned, Holdout": "<",
            "Tuned + Ens., Holdout": ">",
        }

        if tabarena_context is not None:
            self.method_metadata_info = tabarena_context.method_metadata_collection.info()
            self.method_metadata_info = self.method_metadata_info.rename(columns={
                "method": "ta_name",
                "artifact_name": "ta_suite",
            })
            self.method_metadata_info = self.method_metadata_info.drop(columns=["method_type"])
        else:
            self.method_metadata_info = None

    def compute_normalized_error_dynamic(self, df_results: pd.DataFrame) -> pd.DataFrame:
        df_results = df_results.copy(deep=True)
        df_results_og = df_results.copy(deep=True)

        df_results = df_results.drop(columns=["normalized-error-dataset", "normalized-error-task"], errors="ignore")

        method_col = self.method_col

        df_results_per_dataset = df_results.groupby([method_col, "dataset"])[self.error_col].mean().reset_index(
            drop=False)

        # Alternative, this also incorporates Portfolios and HPO into the normalized scoring. This makes normalized-error dependent on what simulations we run.
        # This is unbiased against very strong simulation results because the best method defines what is `0.0` on a dataset.
        normalized_scorer_dataset = NormalizedScorer(
            df_results_per_dataset,
            tasks=list(df_results_per_dataset["dataset"].unique()),
            baseline=None,
            task_col="dataset",
            framework_col=method_col,
            metric_error_col=self.error_col,
        )

        all_tasks = df_results[["dataset", "fold"]].drop_duplicates().values.tolist()
        all_tasks = [tuple(task) for task in all_tasks]

        normalized_scorer_task = NormalizedScorer(
            df_results,
            tasks=all_tasks,
            baseline=None,
            task_col=["dataset", "fold"],
            framework_col=method_col,
        )

        df_results["normalized-error-task"] = [normalized_scorer_task.rank(task=(dataset, fold), error=error) for
                                               (dataset, fold, error) in
                                               zip(df_results["dataset"], df_results["fold"],
                                                   df_results[self.error_col])]

        df_results_per_dataset["normalized-error-dataset"] = [
            normalized_scorer_dataset.rank(task=dataset, error=error) for (dataset, error) in
            zip(df_results_per_dataset["dataset"], df_results_per_dataset[self.error_col])
        ]

        df_results_per_dataset = df_results_per_dataset.set_index(["dataset", method_col], drop=True)[
            "normalized-error-dataset"]
        df_results = df_results.merge(df_results_per_dataset, left_on=["dataset", method_col], right_index=True)

        df_results_og["normalized-error-dataset"] = df_results["normalized-error-dataset"]
        df_results_og["normalized-error-task"] = df_results["normalized-error-task"]
        return df_results_og

    @classmethod
    def _get_config_types(cls, df_results: pd.DataFrame) -> list[str]:
        config_types = sorted([
            config_type for config_type in df_results["config_type"].unique()
            if config_type is not None and isinstance(config_type, str)
        ])
        return config_types

    # TODO: Remove the need for this, have the original results have the correct name to begin with
    @classmethod
    def _rename_dict(cls) -> dict:
        return {}

    def eval(
        self,
        df_results: pd.DataFrame,
        include_norm_score: bool = False,
        use_gmean: bool = False,
        imputed_names: list[str] | None = None,
        baselines: list[str] | str | None = "auto",
        baseline_colors: list[str] | None = None,
        plot_tune_types: list[str] | None = None,
        plot_times: bool = False,
        plot_extra_barplots: bool = False,
        plot_cdd: bool = True,
        plot_runtimes: bool = False,
        plot_pareto: bool = True,
        plot_other: bool = False,
        calibration_framework: str | None = "auto",
        average_seeds: bool = True,
        tmp_treat_tasks_independently: bool = False,  # FIXME: Need to make a weighted elo logic
        leaderboard_kwargs: dict | None = None,
        plot_with_baselines: bool = False,
    ) -> pd.DataFrame:
        if leaderboard_kwargs is None:
            leaderboard_kwargs = {}
        leaderboard_kwargs = leaderboard_kwargs.copy()
        if calibration_framework is not None and calibration_framework == "auto":
            calibration_framework = "RF (default)"
        df_results = df_results.copy(deep=True)
        if "method_metadata" in df_results.columns:
            # currently no need to use this column
            df_results.drop(columns=["method_metadata"], inplace=True)
        if "imputed" not in df_results.columns:
            df_results["imputed"] = False
        df_results["imputed"] = df_results["imputed"].astype(int).fillna(0).astype(bool)

        # rename methods
        _rename_dict = self._rename_dict()
        df_results[self.method_col] = df_results[self.method_col].map(_rename_dict).fillna(df_results[self.method_col])
        if calibration_framework is not None:
            calibration_framework = _rename_dict.get(calibration_framework, calibration_framework)

        if imputed_names is None:
            # TODO: This is a hack
            from tabarena.nips2025_utils.compare import get_imputed_names
            imputed_names = get_imputed_names(df_results=df_results, method_col=self.method_col)

        self.assert_no_duplicates(df_results=df_results)
        self.assert_no_nan_methods(df_results=df_results)

        df_results = self.filter_results(df_results=df_results)

        if isinstance(baselines, list):
            baselines = [_rename_dict.get(b, b) for b in baselines]
        baselines, baseline_colors = self._process_baselines(
            df_results=df_results,
            baselines=baselines,
            baseline_colors=baseline_colors,
        )

        framework_types = self._get_config_types(df_results=df_results[~df_results["method"].isin(baselines)])

        df_results_rank_compare = copy.deepcopy(df_results)

        f_map, f_map_type, f_map_inverse, f_map_type_name = self.get_framework_type_method_names(
            framework_types=framework_types,
        )

        df_results_rank_compare = df_results_rank_compare[(~df_results_rank_compare[self.method_col].map(f_map_type).isna()) | (df_results_rank_compare[self.method_col].isin(baselines))]

        # ----- end removing unused methods -----

        method_info: pd.DataFrame = self.get_method_info(df=df_results_rank_compare)
        if self.method_metadata_info is not None:
            method_info_full = pd.merge(left=method_info.reset_index(), right=self.method_metadata_info, on=["ta_name", "ta_suite"])
            save_pd.save(path=self.output_dir / "method_info.csv", df=method_info_full)
        save_pd.save(path=self.output_dir / "results_per_split.csv", df=df_results_rank_compare)

        # ----- add times per 1K samples -----
        # FIXME: Hack to work in cases where this is not available
        if "n_samples_train_per_fold" not in self.task_metadata.columns or "n_samples_test_per_fold" not in self.task_metadata.columns:
            print(f"WARNING: Missing n_samples_train_per_fold and/or n_samples_test_per_fold in task_metadata. Using dummy values...")
            dataset_to_n_samples_train = self.task_metadata.set_index("name")["NumberOfInstances"].to_dict()
            dataset_to_n_samples_test = self.task_metadata.set_index("name")["NumberOfInstances"].to_dict()
        else:
            dataset_to_n_samples_train = self.task_metadata.set_index("name")["n_samples_train_per_fold"].to_dict()
            dataset_to_n_samples_test = self.task_metadata.set_index("name")["n_samples_test_per_fold"].to_dict()

        df_results_rank_compare['time_train_s_per_1K'] = df_results_rank_compare['time_train_s'] * 1000 / df_results_rank_compare["dataset"].map(
            dataset_to_n_samples_train)
        df_results_rank_compare['time_infer_s_per_1K'] = df_results_rank_compare['time_infer_s'] * 1000 / df_results_rank_compare["dataset"].map(
            dataset_to_n_samples_test)

        if plot_times:
            self.plot_tabarena_times(df=df_results_rank_compare, output_dir=self.output_dir, show=False)

        # TODO: Move this into the `.leaderboard` call
        if "normalized-error-dataset" not in df_results_rank_compare.columns:
            df_results_rank_compare = self.compute_normalized_error_dynamic(df_results=df_results_rank_compare)
        assert "normalized-error-dataset" in df_results_rank_compare.columns, f"Run `self.compute_normalized_error_dynamic(df_results)` first to get normalized-error."
        df_results_rank_compare["normalized-error"] = df_results_rank_compare["normalized-error-dataset"]

        if include_norm_score:
            self.plot_tuning_impact(
                df=df_results_rank_compare,
                framework_types=framework_types,
                save_prefix=f"{self.output_dir}",
                use_gmean=use_gmean,
                baselines=baselines,
                baseline_colors=baseline_colors,
                use_score=True,
                name_suffix="-normscore-dataset-horizontal",
                imputed_names=imputed_names,
                plot_tune_types=plot_tune_types,
                show=False,
                use_y=True
            )

            if plot_extra_barplots:
                self.plot_tuning_impact(
                    df=df_results_rank_compare,
                    framework_types=framework_types,
                    save_prefix=f"{self.output_dir}",
                    use_gmean=use_gmean,
                    baselines=baselines,
                    baseline_colors=baseline_colors,
                    use_score=True,
                    name_suffix="-normscore-dataset",
                    imputed_names=imputed_names,
                    plot_tune_types=plot_tune_types,
                    show=False,
                )

                self.plot_tuning_impact(
                    df=df_results_rank_compare,
                    framework_types=framework_types,
                    save_prefix=f"{self.output_dir}",
                    use_gmean=use_gmean,
                    baselines=baselines,
                    baseline_colors=baseline_colors,
                    use_score=True,
                    metric="normalized-error-task",
                    name_suffix="-normscore-task",
                    imputed_names=imputed_names,
                    plot_tune_types=plot_tune_types,
                    show=False,
                )

        elo_kwargs = dict(
            calibration_framework=calibration_framework,
            calibration_elo=1000,
            BOOTSTRAP_ROUNDS=self.elo_bootstrap_rounds,
        )

        if tmp_treat_tasks_independently:
            df_results_rank_compare["dataset"] = df_results_rank_compare["dataset"] + "_" + df_results_rank_compare["fold"].astype(str)
            df_results_rank_compare["fold"] = 0

        tabarena = TabArena(
            method_col=self.method_col,
            task_col="dataset",
            seed_column="fold",
            error_col=self.error_col,
            columns_to_agg_extra=[
                "time_train_s",
                "time_infer_s",
                "time_train_s_per_1K",
                "time_infer_s_per_1K",
                "normalized-error",
                "normalized-error-task",
                "imputed",
            ],
            groupby_columns=[
                "metric",
                "problem_type",
            ],
        )

        leaderboard_kwargs.setdefault("include_elo", True)
        leaderboard_kwargs.setdefault("include_winrate", True)
        leaderboard_kwargs.setdefault("include_mrr", True)
        leaderboard_kwargs.setdefault("include_rank_counts", True)
        leaderboard_kwargs.setdefault("baseline_method", calibration_framework)
        leaderboard_kwargs.setdefault("elo_kwargs", elo_kwargs)
        leaderboard_kwargs.setdefault("average_seeds", average_seeds)

        leaderboard = tabarena.leaderboard(
            data=df_results_rank_compare,
            **leaderboard_kwargs,
        )
        elo_map = leaderboard["elo"]
        leaderboard = leaderboard.reset_index(drop=False)
        save_pd.save(path=f"{self.output_dir}/tabarena_leaderboard.csv", df=leaderboard)

        self.create_leaderboard_latex(leaderboard, framework_types=framework_types, save_dir=self.output_dir)

        print(
            f"Evaluating with {len(df_results_rank_compare[tabarena.task_col].unique())} datasets... | problem_types={self.problem_types}, folds={self.folds}")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print(leaderboard)

        # horizontal elo barplot
        self.plot_tuning_impact(
            df=df_results_rank_compare,
            df_elo=leaderboard,
            framework_types=framework_types,
            save_prefix=f"{self.output_dir}",
            use_gmean=use_gmean,
            baselines=baselines,
            baseline_colors=baseline_colors,
            name_suffix="-elo-horizontal",
            imputed_names=imputed_names,
            plot_tune_types=plot_tune_types,
            use_y=True,
            show=False
        )

        # vertical elo barplot
        self.plot_tuning_impact(
            df=df_results_rank_compare,
            df_elo=leaderboard,
            framework_types=framework_types,
            save_prefix=f"{self.output_dir}",
            use_gmean=use_gmean,
            baselines=baselines,
            baseline_colors=baseline_colors,
            name_suffix="-elo",
            imputed_names=imputed_names,
            plot_tune_types=plot_tune_types,
            show=False
        )

        results_per_task = tabarena.compute_results_per_task(data=df_results_rank_compare)
        results_per_split = tabarena.compute_results_per_task(data=df_results_rank_compare, include_seed_col=True)

        # TODO: Consider adding the metadata to the saved `results_per_split.csv` file?
        # assert len(results_per_split) == len(df_results_rank_compare)
        # groupby_columns = tabarena._get_groupby_cols(results=results_per_split)
        # extra_cols = [c for c in df_results_rank_compare.columns if c not in results_per_split.columns]
        # results_per_split_w_metadata = results_per_split.merge(df_results_rank_compare[[*groupby_columns, *extra_cols]], on=groupby_columns)
        # assert len(results_per_split) == len(results_per_split_w_metadata)

        # FIXME: Is critical diagram incorrect?
        if plot_cdd:
            def rename_model(name: str):
                parts = name.split(" ")
                if parts[0] in f_map_type_name:
                    parts[0] = f_map_type_name[parts[0]]
                name = " ".join(parts)
                return name.replace('(tuned + ensemble)', '(T+E)')

            # use tuned+ensembled version if available, and default otherwise
            tune_methods = results_per_task[self.method_col].map(method_info["method_subtype"])
            method_types = results_per_task[self.method_col].map(method_info["config_type"]).fillna(results_per_task[self.method_col])

            tuned_ens_types = method_types[tune_methods == 'tuned_ensemble']
            per_task_filter = (tune_methods == 'tuned_ensemble') | ((tune_methods == 'default') & ~method_types.isin(tuned_ens_types))

            tune_methods_split = results_per_split[self.method_col].map(method_info["method_subtype"])
            method_types_split = results_per_split[self.method_col].map(method_info["config_type"]).fillna(results_per_split[self.method_col])
            tuned_ens_types_split = method_types_split[tune_methods_split == 'tuned_ensemble']
            per_split_filter = (tune_methods_split == 'tuned_ensemble') | ((tune_methods_split == 'default') & ~method_types_split.isin(tuned_ens_types_split))

            if plot_with_baselines:
                per_task_filter = per_task_filter | results_per_task[self.method_col].isin(baselines)
                per_split_filter = per_split_filter | results_per_split[self.method_col].isin(baselines)

            results_te_per_task = results_per_task[per_task_filter]
            results_te_per_split = results_per_split[per_split_filter]

            results_te_per_task.loc[:, self.method_col] = results_te_per_task[self.method_col].map(rename_model)
            results_te_per_split.loc[:, self.method_col] = results_te_per_split[self.method_col].map(rename_model)

            if average_seeds:
                _results_to_use_winrate_matrix = results_te_per_task.copy()
            else:
                _results_to_use_winrate_matrix = results_te_per_split.copy()

            if len(_results_to_use_winrate_matrix) != 0:
                winrate_matrix = tabarena.compute_winrate_matrix(results_per_task=_results_to_use_winrate_matrix)
                try:
                    tabarena.plot_winrate_matrix(
                        winrate_matrix=winrate_matrix,
                        save_path=str(Path(self.output_dir / f"winrate_matrix.{self.figure_file_type}")),
                    )
                except RuntimeError as e:
                    print(
                        "Warning: RuntimeError encountered during winrate matrix plotting. "
                        "This likely means the CLI does not have access to the correct Chromium version..."
                    )

            if len(results_te_per_task) != 0:
                try:
                    tabarena.plot_critical_diagrams(
                        results_per_task=results_te_per_task,
                        save_path=f"{self.output_dir}/figures/critical-diagram.{self.figure_file_type}",
                        show=False,
                    )
                except ValueError as e:
                    print(
                        f"Warning: ValueError encountered during critical diagram plotting. "
                        f"This likely means there is too little data to compute critical diagrams. Skipping ..."
                    )

        if plot_runtimes:
            self.generate_runtime_plot(df_results=df_results_rank_compare)

        if plot_pareto and (framework_types or plot_with_baselines):
            self.plot_pareto(
                leaderboard=leaderboard,
                framework_types=framework_types,
                with_baselines=plot_with_baselines,
            )

        if plot_other:
            try:
                import autogluon_benchmark
            except:
                print(f"WARNING: autogluon_benchmark failed to import... skipping extra figure generation")
            else:
                results_per_task_ag_benchmark = results_per_task.rename(columns={
                    "improvability": "bestdiff",
                })
                self.run_autogluon_benchmark_logic(
                    results_per_task=results_per_task_ag_benchmark,
                    elo_map=elo_map,
                    tabarena=tabarena,
                    calibration_framework=calibration_framework,
                )

        return leaderboard

    def assert_no_duplicates(self, df_results: pd.DataFrame):
        # don't allow duplicate results
        dupes = df_results[df_results.duplicated(
            subset=["dataset", "fold", self.method_col],
            keep=False,
        )]
        if not dupes.empty:
            dupes = dupes.sort_values(by=[self.method_col, "dataset", "fold"])
            duplicated_methods = dupes["method"].value_counts()
            raise ValueError(
                "Duplicate rows detected on keys [dataset, fold, "
                f"{self.method_col}].\n"
                f"The following {len(duplicated_methods)} methods were duplicated (w/ counts):\n"
                f"{duplicated_methods.to_string()}\n"
                f"The following {len(dupes)} rows are duplicates:\n"
                f"{dupes.to_string(index=False)}"
            )

    def assert_no_nan_methods(self, df_results: pd.DataFrame):
        if df_results[self.method_col].isna().any():
            missing_count = df_results[self.method_col].isna().sum()
            missing_percent = missing_count / len(df_results)
            raise AssertionError(
                f"Found NaN values in '{self.method_col}' column: "
                f"{missing_count}/{len(df_results)} ({missing_percent * 100:.1f}%) were NaN."
            )

    def filter_results(self, df_results: pd.DataFrame):
        if self.datasets is not None:
            df_results = df_results[df_results["dataset"].isin(self.datasets)]
        if self.folds is not None:
            df_results = df_results[df_results["fold"].isin(self.folds)]
        if self.methods is not None:
            df_results = df_results[df_results[self.method_col].isin(self.methods)]
        if self.problem_types is not None:
            df_results = df_results[df_results["problem_type"].isin(self.problem_types)]
        if not self.keep_best:
            df_results = df_results[df_results["method_subtype"] != "best"]
        if self.banned_model_types:
            df_results = df_results[~df_results["config_type"].isin(self.banned_model_types)]
        return df_results

    def get_method_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Verify that each method has exactly one unique value for method_type,
        method_subtype, config_type, ta_name, and ta_suite. Return a mapping dataframe.

        Raises
        ------
        ValueError
            If any method has non-unique values in any of the checked columns.

        Returns
        -------
        pd.DataFrame
            A dataframe indexed by "method" with
            the unique values for each of method_type, method_subtype, config_type, ta_name, ta_suite.
        """

        group_cols = self.method_col
        value_cols = ["method_type", "method_subtype", "config_type", "ta_name", "ta_suite"]

        grouped = df.groupby(group_cols)[value_cols]

        # Compute sets of unique values per group
        unique_vals = grouped.nunique(dropna=False)

        # Identify problematic groups
        bad = (unique_vals > 1).any(axis=1)
        if bad.any():
            msg = (
                "Found groups with multiple unique values in one or more columns.\n"
                f"Columns checked: {value_cols}\n\n"
                "Offending groups:\n"
                f"{unique_vals[bad]}"
            )
            raise ValueError(msg)

        # Safe: each value is unique → extract the scalar values
        mapping = grouped.first()  # identical to .agg("first") but faster

        return mapping

    def plot_pareto(self, leaderboard: pd.DataFrame, framework_types: list[str], with_baselines: bool = True):
        f_map, f_map_type, f_map_inverse, f_map_type_name = self.get_framework_type_method_names(
            framework_types=framework_types,
        )
        leaderboard_pareto = leaderboard.copy()
        leaderboard_pareto["Method"] = leaderboard_pareto[self.method_col].map(f_map_type).fillna(
            leaderboard_pareto[self.method_col])

        if self.banned_pareto_methods:
            leaderboard_pareto = leaderboard_pareto[~leaderboard_pareto["Method"].isin(self.banned_pareto_methods)]

        leaderboard_pareto["Type"] = leaderboard_pareto[self.method_col].map(f_map_inverse).fillna("baseline")
        leaderboard_pareto["Method"] = leaderboard_pareto["Method"].map(f_map_type_name).fillna(
            leaderboard_pareto["Method"])
        f_map_suffix = get_f_map_suffix_plots()
        leaderboard_pareto["suffix"] = leaderboard_pareto["Type"].map(f_map_suffix).fillna("")
        leaderboard_pareto[self.method_col] = leaderboard_pareto["Method"] + leaderboard_pareto["suffix"]
        fig_rename_dict = {
            "baseline": "Baseline",
            "default": "Default",
            "tuned": "Tuned",
            "tuned_ensembled": "Tuned + Ens.",
            "best": "Best",
            "holdout": "Default, Holdout",
            "holdout_tuned": "Tuned, Holdout",
            "holdout_tuned_ensembled": "Tuned + Ens., Holdout",
        }
        leaderboard_pareto["Type"] = leaderboard_pareto["Type"].map(fig_rename_dict).fillna(
            leaderboard_pareto["Type"]
        )

        if not with_baselines:
            leaderboard_pareto = leaderboard_pareto[leaderboard_pareto["Type"] != "Baseline"]

        self.plot_pareto_elo_vs_time_infer(leaderboard=leaderboard_pareto)
        self.plot_pareto_elo_vs_time_train(leaderboard=leaderboard_pareto)
        self.plot_pareto_improvability_vs_time_infer(leaderboard=leaderboard_pareto)
        self.plot_pareto_improvability_vs_time_train(leaderboard=leaderboard_pareto)

    def run_autogluon_benchmark_logic(self, results_per_task: pd.DataFrame, elo_map: dict, tabarena: TabArena,
                                      calibration_framework: str):
        """
        Requires autogluon_benchmark installed:

        Parameters
        ----------
        results_per_task
        elo_map
        tabarena
        calibration_framework

        Returns
        -------

        """
        results_per_task_rename = results_per_task.rename(columns={
            tabarena.method_col: "framework",
            tabarena.task_col: "dataset",
            tabarena.error_col: "metric_error",
        })

        results_per_task_rename["elo"] = results_per_task_rename["framework"].map(elo_map)

        # Nick: Comment out this code if you don't have autogluon_benchmark
        from autogluon_benchmark.plotting.plotter import Plotter
        plotter = Plotter(
            results_ranked_df=results_per_task_rename,
            results_ranked_fillna_df=results_per_task_rename,
            save_dir=f"{self.output_dir}/figures/plotter",
            show=False,
        )

        # FIXME: Nick: This isn't yet merged, as I haven't made it nice yet
        # plotter.plot_pareto_time_infer_elo(data=results_per_task_rename)
        # plotter.plot_pareto_time_train_elo(data=results_per_task_rename)

        plotter.plot_all(
            calibration_framework=calibration_framework,
            calibration_elo=1000,
            BOOTSTRAP_ROUNDS=self.elo_bootstrap_rounds,
        )

    def plot_pareto_elo_vs_time_train(self, leaderboard: pd.DataFrame):
        save_prefix = Path(self.output_dir)
        save_path = str(save_prefix / f"pareto_front_elo_vs_time_train.{self.figure_file_type}")
        y_name = "Elo"
        x_name = "Train time per 1K samples (s) (median)"
        title = "Elo vs Train Time"

        data = leaderboard.copy()
        data[x_name] = data["median_time_train_s_per_1K"]
        data[y_name] = data["elo"]

        _plot_pareto(
            data=data,
            x_name=x_name,
            y_name=y_name,
            max_X=False,
            max_Y=True,
            sort_y=True,
            hue="Method",  # color by family
            style_col="Type",  # marker by run type
            style_order=self.style_order,
            style_markers=self.style_markers,
            label_col=self.method_col,  # annotate with full method name
            title=title,
            save_path=save_path,
            show=False,
            aspect=4 / 3,
        )

    def plot_pareto_elo_vs_time_infer(self, leaderboard: pd.DataFrame):
        save_prefix = Path(self.output_dir)
        save_path = str(save_prefix / f"pareto_front_elo_vs_time_infer.{self.figure_file_type}")
        y_name = "Elo"
        x_name = "Inference time per 1K samples (s) (median)"
        title = f"Elo vs Inference Time"

        data = leaderboard.copy()
        data[x_name] = data["median_time_infer_s_per_1K"]
        data[y_name] = data["elo"]

        _plot_pareto(
            data=data,
            x_name=x_name,
            y_name=y_name,
            max_X=False,
            max_Y=True,
            sort_y=True,
            hue="Method",  # <-- same color for same method_type
            style_col="Type",  # <-- different marker per run_type
            style_order=self.style_order,
            style_markers=self.style_markers,
            label_col=self.method_col,  # <-- annotate with full method name
            title=title,
            save_path=save_path,
            show=False,
            aspect=4 / 3,
        )

    def plot_pareto_improvability_vs_time_infer(self, leaderboard: pd.DataFrame):
        save_prefix = Path(self.output_dir)
        save_path = str(save_prefix / f"pareto_front_improvability_vs_time_infer.{self.figure_file_type}")
        y_name = "Improvability (%)"
        x_name = "Inference time per 1K samples (s) (median)"
        title = "Improvability vs Inference Time"

        data = leaderboard.copy()
        data[x_name] = data["median_time_infer_s_per_1K"]
        data[y_name] = data["improvability"] * 100

        _plot_pareto(
            data=data,
            x_name=x_name,
            y_name=y_name,
            max_X=False,
            max_Y=False,
            sort_y=True,
            hue="Method",  # color by family
            style_col="Type",  # marker by run type
            style_order=self.style_order,
            style_markers=self.style_markers,
            label_col=self.method_col,  # annotate with full method name
            # ylim=(0, None),
            title=title,
            save_path=save_path,
            show=False,
            aspect=4 / 3,
        )

    def plot_pareto_improvability_vs_time_train(self, leaderboard: pd.DataFrame):
        save_prefix = Path(self.output_dir)
        save_path = str(save_prefix / f"pareto_front_improvability_vs_time_train.{self.figure_file_type}")
        y_name = "Improvability (%)"
        x_name = "Train time per 1K samples (s) (median)"
        title = "Improvability vs Train Time"

        data = leaderboard.copy()
        data[x_name] = data["median_time_train_s_per_1K"]
        data[y_name] = data["improvability"] * 100

        _plot_pareto(
            data=data,
            x_name=x_name,
            y_name=y_name,
            max_X=False,
            max_Y=False,
            sort_y=True,
            hue="Method",  # color by family
            style_col="Type",  # marker by run type
            style_order=self.style_order,
            style_markers=self.style_markers,
            label_col=self.method_col,  # annotate with full method name
            # ylim=(0, None),
            title=title,
            save_path=save_path,
            show=False,
            aspect=4 / 3,
        )

    def get_method_rename_map(self) -> dict[str, str]:
        return get_method_rename_map()  # FIXME: Avoid hardcoding

    def plot_portfolio_ensemble_weights_barplot(self, df_ensemble_weights: pd.DataFrame):
        import seaborn as sns
        import matplotlib.pyplot as plt
        from pathlib import Path
        import matplotlib.colors as mcolors
        import numpy as np

        fig, ax = plt.subplots(1, 1,
                               figsize=(3.5, 3)
                               )

        df_ensemble_weights = df_ensemble_weights.copy(deep=True)
        _method_rename_map = self.get_method_rename_map()
        columns_new = [_method_rename_map.get(c, c) for c in df_ensemble_weights.columns]
        df_ensemble_weights.columns = columns_new

        df_long = df_ensemble_weights.melt(var_name="Model", value_name="Weight")
        model_order = list(df_ensemble_weights.columns)

        pastel_palette = sns.color_palette("pastel")
        deep_palette = sns.color_palette("deep")

        # Define gradient from pastel and deep separately
        pastel_start = mcolors.to_rgb(pastel_palette[2])
        pastel_end = mcolors.to_rgb(pastel_palette[0])
        deep_start = mcolors.to_rgb(deep_palette[2])
        deep_end = mcolors.to_rgb(deep_palette[0])

        # Create pastel gradient for bars
        bar_colors = [mcolors.to_hex(c) for c in np.linspace(pastel_start, pastel_end, len(model_order))]

        # Create deep gradient for error bars
        error_colors = [mcolors.to_hex(c) for c in np.linspace(deep_start, deep_end, len(model_order))]

        # Alphas for bars
        alphas = np.linspace(1.0, 1.0, len(model_order))[::-1]  # Keep at 1.0 for now

        # Create barplot
        barplot = sns.barplot(
            data=df_long,
            x="Weight",
            y="Model",
            hue="Model",
            legend=False,
            ax=ax,
            order=model_order,
            palette=bar_colors,
        )

        # Apply alpha to bar colors
        for patch, alpha in zip(barplot.patches, alphas):
            r, g, b = patch.get_facecolor()[:3]
            patch.set_facecolor((r, g, b, alpha))

        # Update error bar colors manually
        for i, line in enumerate(ax.lines):
            # Seaborn/matplotlib adds error bar lines in a certain order.
            # Each bar usually has 2 lines: one vertical bar and one cap on top.
            # Here, we assume two lines per error bar, so divide i by 2.
            color_index = i // 2
            if color_index < len(error_colors):
                line.set_color(error_colors[color_index])

        barplot.set_xlabel("Average weight in TabArena ensemble")
        barplot.set_ylabel("")

        fig_name = f"portfolio-weight-barplot.{self.figure_file_type}"
        fig_prefix = Path(self.output_dir) / "figures"
        fig_prefix.mkdir(parents=True, exist_ok=True)

        fig_save_path = fig_prefix / fig_name
        plt.savefig(fig_save_path)

    def create_leaderboard_latex(self, df: pd.DataFrame, framework_types, save_dir):
        df = df.copy(deep=True)
        f_map, f_map_type, f_map_inverse, f_map_type_name = self.get_framework_type_method_names(
            framework_types=framework_types,
        )

        def rename_model(name: str):
            parts = name.split(" ")
            if parts[0] in f_map_type_name:
                parts[0] = f_map_type_name[parts[0]]
            name = " ".join(parts)
            name = name.replace('(default)', '(D)')
            name = name.replace('(tuned)', '(T)')
            name = name.replace('(tuned + ensemble)', '(T+E)')
            return name

        df = df.sort_values(by="elo", ascending=False)

        df_new = pd.DataFrame()

        df_new[r"Model"] = df[self.method_col].map(rename_model)
        # do the more annoying way {}_{...} so that \textbf{} affects the main number
        df_new[r"Elo ($\uparrow$)"] = [f'{round(elo)}' + r'${}_{' + f'-{math.ceil(elom)},+{math.ceil(elop)}' + r'}$'
                                       for elo, elom, elop in zip(df["elo"], df["elo-"], df["elo+"])]
        df_new[r"Norm." + "\n" + r"score ($\uparrow$)"] = [f'{1-err:5.3f}' for err in df["normalized-error"]]
        df_new[r"Avg." + "\n" + r"rank ($\downarrow$)"] = [f'{rank:.1f}' for rank in df["rank"]]
        df_new["Harm.\nmean\n" + r"rank ($\downarrow$)"] = [f'{1/val:.1f}' for val in df["mrr"]]
        df_new[r"\#wins ($\uparrow$)"] = [str(cnt) for cnt in df["rank=1_count"]]
        df_new[f"Improva-\n" + r"bility ($\downarrow$)"] = [f'{100*val:.1f}\\%' for val in df["improvability"]]
        df_new[r"Train time" + "\n" + r"per 1K [s]"] = [f'{t:.2f}' for t in df["median_time_train_s_per_1K"]]
        df_new[r"Predict time" + "\n" + r"per 1K [s]"] = [f'{t:.2f}' for t in df["median_time_infer_s_per_1K"]]

        # ----- highlight best and second-best numbers per column -----

        # first, convert the strings back to floats
        def extract_first_float(s):
            """
            Extracts the first sequence of digits (including decimal point) from the input string
            and returns it as a float. Returns None if no valid number is found.
            """
            match = re.search(r'\d+(\.\d+)?', s)
            if match:
                return float(match.group())
            return None

        def find_smallest_and_second_smallest_indices(numbers):
            if len(numbers) < 2:
                return [], []

            # Find the smallest value
            min_val = min(numbers)
            min_indices = [i for i, x in enumerate(numbers) if x == min_val]

            # Exclude the smallest values and find the second smallest
            remaining = [x for x in numbers if x != min_val]
            if not remaining:
                return min_indices, []  # No second smallest

            second_min_val = min(remaining)
            second_min_indices = [i for i, x in enumerate(numbers) if x == second_min_val]

            return min_indices, second_min_indices

        # then, add textbf or underline to the correct rows
        for col_idx, col in enumerate(df_new.columns):
            if r'\uparrow' in col or r'\downarrow' in col:
                # factor = 1 if r'\downarrow' in col else -1
                # numbers = [factor * extract_first_float(s) for s in df_new[col]]
                ranks = df_new[col].map(extract_first_float).rank(method="min", ascending=r'\downarrow' in col)
                for rank, color in [(1, 'gold'), (2, 'silver'), (3, 'bronze')]:
                    df_new.loc[ranks == rank, col] = df_new.loc[ranks == rank, col].apply(
                        lambda x: f"\\textcolor{{{color}}}{{\\textbf{{{x}}}}}"
                    )

                # min_indices, second_min_indices = find_smallest_and_second_smallest_indices(numbers)
                # for idx in min_indices:
                #     df_new.iloc[idx, col_idx] = r'\textbf{' + df_new.iloc[idx, col_idx] + r'}'
                # for idx in second_min_indices:
                #     df_new.iloc[idx, col_idx] = r'\underline{' + df_new.iloc[idx, col_idx] + r'}'


        # ----- create latex table -----

        rows = []
        rows.append(r'\begin{tabular}{' + 'llcccccrr' + r'}')
        rows.append(r'\toprule')
        # rows.append(' & '.join(df_new.columns) + r' \\')

        col_names_split = [col.split('\n') for col in df_new.columns]
        n_rows_header = max([len(rows) for rows in col_names_split])
        for row_idx in range(n_rows_header):
            rows.append(' & '.join([r'\textbf{' + lst[row_idx] + r'}' if row_idx < len(lst) else ''
                                    for lst in col_names_split]) + r' \\')
        rows.append(r'\midrule')

        for row_index, row in df_new.iterrows():
            rows.append(' & '.join([row[col_name] for col_name in df_new.columns]) + r' \\')

        rows.append(r'\bottomrule')
        rows.append(r'\end{tabular}')

        table = '\n'.join(rows)

        with open(Path(save_dir) / 'leaderboard.tex', 'w') as f:
            f.write(table)

    # FIXME: Avoid hardcoding
    def plot_tuning_impact(
        self,
        df: pd.DataFrame,
        framework_types: list,
        save_prefix: str,
        baselines: list[str] = None,
        baseline_colors: list[str] = None,
        show: bool = True,
        use_gmean=False,
        use_score: bool = True,
        df_elo: pd.DataFrame = None,
        name_suffix: str | None = None,
        imputed_names: list[str] | None = None,
        use_y: bool = False,
        metric: str = "normalized-error",
        plot_tune_types: list[str] | None = None,
    ):
        same_width = use_y
        use_lim = True
        use_elo = df_elo is not None
        lower_is_better = True
        lim = None
        xlim = None
        ylim = None

        if imputed_names is None:
            imputed_names = []
        # imputed_names = imputed_names or ['TabPFNv2', 'TabICL']

        df = df.copy(deep=True)

        framework_col = "framework_type"
        # framework_col = "framework_name"

        groupby_columns_extra = ["dataset"]

        if use_elo:
            metric = "elo"
            use_lim = True
            lim = [self.elo_ymin, None]
            lower_is_better = False
            df = df_elo.copy(deep=True)
            df = df[[self.method_col, "elo", "elo+", "elo-"]]
            groupby_columns_extra = []
        elif use_score:
            lower_is_better = False
            df["normalized-score"] = 1 - df[metric]
            # df_plot_w_mean_per_dataset["normalized-score"] = 1 - df_plot_w_mean_per_dataset["normalized-error"]
            metric = "normalized-score"
        else:
            metric = metric

        f_map, f_map_type, f_map_inverse, f_map_type_name = self.get_framework_type_method_names(
            framework_types=framework_types,
        )

        df = df.copy()
        df.loc[:, "framework_type"] = df[self.method_col].map(f_map_type).fillna(df[self.method_col])
        df.loc[:, "tune_method"] = df[self.method_col].map(f_map_inverse).fillna("default")

        if baselines is None:
            baselines = []
        if baseline_colors is not None:
            assert len(baselines) == len(
                baseline_colors), f"A color must be specified for each baseline via the `baseline_colors` argument."

        has_non_baselines = len(framework_types) != 0
        framework_types = baselines + framework_types

        df["framework_type"] = df["framework_type"].map(f_map_type_name).fillna(df["framework_type"])
        framework_types = [f_map_type_name[ft] if ft in f_map_type_name else ft for ft in framework_types]

        if plot_tune_types:
            df = df[df["tune_method"].isin(plot_tune_types) | df[self.method_col].isin(baselines)]

        df_plot = df[df["framework_type"].isin(framework_types)]
        # df_plot = df_plot[~df_plot["framework_type"].isin(imputed_names)]

        # pd.set_option('display.max_columns', None)  # todo
        # print(f'{df_plot.head()=}')

        # df_plot_w_mean_2 = df_plot.groupby(["framework_type", "tune_method"])[metric].mean().reset_index()

        df_plot_w_mean_per_dataset = df_plot.groupby(["framework_type", "tune_method", *groupby_columns_extra])[
            metric].mean().reset_index()

        if use_gmean:
            # FIXME: Doesn't plot correctly, need to figure out error bars for geometric mean
            df_plot_eps = df_plot.copy(deep=True)
            df_plot_eps[metric] += 0.01
            from scipy.stats import gmean
            df_plot_w_gmean_per_dataset = df_plot.groupby(["framework_type", "tune_method", *groupby_columns_extra])[
                metric].apply(gmean).reset_index()
            df_plot_w_mean_per_dataset = df_plot_w_gmean_per_dataset

        df_plot_w_mean_2 = df_plot_w_mean_per_dataset.groupby(["framework_type", "tune_method"])[
            metric].mean().reset_index()

        df_plot_w_mean_2 = df_plot_w_mean_2.sort_values(by=metric, ascending=lower_is_better)
        baseline_means = {}
        for baseline in baselines:
            baseline_means[baseline] = df_plot_w_mean_2[df_plot_w_mean_2["framework_type"] == baseline][metric].iloc[0]

        df_plot_w_mean_2 = df_plot_w_mean_2[~df_plot_w_mean_2["framework_type"].isin(baselines)]

        df_plot_mean_dedupe = df_plot_w_mean_2.drop_duplicates(subset=["framework_type"], keep="first")

        framework_type_order = list(df_plot_mean_dedupe["framework_type"].to_list())
        framework_type_order.reverse()

        # change to names
        # df_plot_w_mean_per_dataset["framework_type"] = df_plot_w_mean_per_dataset["framework_type"].map(f_map_type_name)

        # sns.set_color_codes("pastel")
        # with sns.plotting_context("notebook", font_scale=0.8, rc={
        #     "pgf.texsystem": "pdflatex",
        #     'font.family': 'serif',
        #     'font.size': 10.95,
        #     'text.usetex': True,
        #     'pgf.rcfonts': False,
        #     # 'legend.framealpha': 0.5,
        #     'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{xcolor}'
        # }):

        with sns.axes_style("whitegrid"):
            # with plt.rc_context({'font.family': 'serif', "text.usetex": True, 'font.size': 12, 'axes.labelsize': 12, 'xtick.labelsize': 12}):
            with plt.rc_context(self.rc_context_params
                                # | figsizes.neurips2024(height_to_width_ratio=0.8)
                                ):
            # with plt.rc_context(fontsizes.neurips2024() | fonts.neurips2024()):
                # with plt.rc_context(figsizes.neurips2024(height_to_width_ratio=0.8)):
                colors = sns.color_palette("pastel").as_hex()
                errcolors = sns.color_palette("deep").as_hex()

                if use_lim and not lim:
                    lim = [0, None]
                if use_y:
                    pos = metric
                    y = framework_col
                    figsize = (4, 5)
                    xlim = lim

                    framework_type_order.reverse()

                else:
                    pos = framework_col
                    y = metric
                    ylim = lim
                    figsize = (0.5*len(framework_types), 2.7)
                    # figsize = None

                fig, ax = plt.subplots(1, 1, figsize=figsize)
                # fig, ax = plt.subplots(1, 1)

                if use_y:
                    baseline_func = ax.axvline
                else:
                    baseline_func = ax.axhline

                linewidth = 0.0 if use_y else 0.3
                err_linewidth = 1.6
                err_linewidths = {
                    'tuned_ensembled': err_linewidth,
                    'tuned': err_linewidth * 0.8,
                    'default': err_linewidth * 0.6,
                    'holdout_tuned_ensembled': err_linewidth * 0.6,
                }
                err_alpha = 0.6

                to_plot = [
                    dict(
                        x=pos, y=y,
                        # hue="tune_method",  # palette=["m", "g", "r],
                        label="Tuned + Ensembled",
                        data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned_ensembled"],
                        ax=ax,
                        order=framework_type_order, color=colors[2],
                        width=0.6, linewidth=linewidth,
                        err_kws={"color": errcolors[2], "linewidth": err_linewidths['tuned_ensembled'], 'alpha': err_alpha},
                    ),
                    # dict(
                    #     x=x, y=y,
                    #     # hue="tune_method",  # palette=["m", "g", "r],
                    #     label="Default (Holdout)",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "holdout"], ax=ax,
                    #     order=framework_type_order,
                    #     color=colors[4],
                    #     width=0.7, linewidth=linewidth,
                    #     err_kws={"color": errcolors[4]},
                    # ),
                    # dict(
                    #     x=x, y=y,
                    #     # hue="tune_method",  # palette=["m", "g", "r],
                    #     label="Tuned (Holdout)",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "holdout_tuned"], ax=ax,
                    #     order=framework_type_order,
                    #     color=colors[5],
                    #     width=0.65, linewidth=linewidth,
                    #     err_kws={"color": errcolors[5]},
                    # ),
                    dict(
                        x=pos, y=y,
                        # hue="tune_method",  # palette=["m", "g", "r],
                        label="Tuned",
                        data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned"], ax=ax,
                        order=framework_type_order,
                        color=colors[1],
                        width=0.5, linewidth=linewidth,
                        err_kws={"color": errcolors[1], "linewidth": err_linewidths['tuned'], 'alpha': err_alpha},
                    ),
                    dict(
                        x=pos, y=y,
                        # hue="tune_method",  # palette=["m", "g", "r],
                        label="Default",
                        data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "default"], ax=ax,
                        order=framework_type_order, color=colors[0],
                        width=0.4, linewidth=linewidth,
                        err_kws={"color": errcolors[0], "linewidth": err_linewidths['default'], 'alpha': err_alpha},
                        alpha=1.0,
                    ),
                    dict(
                        x=pos, y=y,
                        # hue="tune_method",  # palette=["m", "g", "r],
                        label="Tuned + Ensembled (Holdout)",
                        data=df_plot_w_mean_per_dataset[
                            df_plot_w_mean_per_dataset["tune_method"] == "holdout_tuned_ensembled"], ax=ax,
                        order=framework_type_order,
                        color=colors[3],
                        width=0.3, linewidth=linewidth,
                        err_kws={"color": errcolors[3], "linewidth": err_linewidths['holdout_tuned_ensembled'],
                                 'alpha': err_alpha},
                    ),
                    # dict(
                    #     x=x, y=y,
                    #     # hue="tune_method",  # palette=["m", "g", "r],
                    #     label="Best",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "best"], ax=ax,
                    #     order=framework_type_order, color=colors[3],
                    #     width=0.55, linewidth=linewidth,
                    #     err_kws={"color": errcolors[3]},
                    #     alpha=1.0,
                    # ),
                    # dict(
                    #     x=x, y=y,
                    #     # hue="tune_method",  # palette=["m", "g", "r],
                    #     label="Tuned (4h)",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned_4h"], ax=ax,
                    #     order=framework_type_order,
                    #     color=colors[4],
                    #     width=0.5, linewidth=linewidth,
                    #     err_kws={"color": errcolors[4]},
                    # ),
                    # dict(
                    #     x=x, y=y,
                    #     # hue="tune_method",  # palette=["m", "g", "r],
                    #     label="Tuned + Ensembled (4h)",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned_ensembled_4h"], ax=ax,
                    #     order=framework_type_order, color=colors[5],
                    #     width=0.4,
                    #     err_kws={"color": errcolors[5]},
                    # ),

                ]

                if use_score:
                    widths = [plot_line["width"] for plot_line in to_plot]
                    colors = [plot_line["color"] for plot_line in to_plot]
                    err_kws_lst = [plot_line["err_kws"] for plot_line in to_plot]

                    # to_plot.reverse()
                    for plot_line, width, color, err_kws in zip(to_plot, widths, colors, err_kws_lst):
                        if same_width:
                            plot_line["width"] = 0.6 * 1.3
                        else:
                            plot_line["width"] = width * 1.3
                        # plot_line["color"] = color
                        # plot_line["err_kws"] = err_kws

                for plot_line in to_plot:
                    boxplot = sns.barplot(**plot_line)

                if use_y:
                    boxplot.set(xlabel='Elo' if metric=='elo' else 'Normalized score', ylabel=None)
                else:
                    boxplot.set(xlabel=None, ylabel='Elo' if metric=='elo' else 'Normalized score')  # remove method in the x-axis
                # boxplot.set_title("Effect of tuning and ensembling")

                # # FIXME: (Nick) HACK, otherwise it isn't in the plot, don't know why
                # if use_elo:
                #     if baseline_means and "Portfolio-N200 (ensemble) (4h)" in baselines:
                #         max_baseline_mean = max([v for k, v in baseline_means.items()])
                #         if ylim is not None:
                #             ylim[1] = max_baseline_mean + 50
                #         if xlim is not None:
                #             xlim[1] = max_baseline_mean + 50

                # do this before setting x/y limits
                for baseline_idx, (baseline, color) in enumerate(zip(baselines, baseline_colors)):
                    baseline_mean = baseline_means[baseline]
                    # baseline_func(baseline_mean, label=baseline, color=color, linewidth=2.0, ls="--")
                    baseline_func(baseline_mean, color=color, linewidth=2.0, ls="--", zorder=-10)

                    if baseline == 'Portfolio-N200 (ensemble) (4h)':
                        baseline = 'TabArena ensemble (4h)'

                    if use_y:
                        ax.text(y=(1 - 0.035 * (1 + 2*(len(baselines) - 1 - baseline_idx))) * ax.get_ylim()[0],
                                x=baseline_mean * 0.985, s=baseline, ha='right', color=color)
                    else:
                        ax.text(x=0.5, y=baseline_mean * 0.97, s=baseline, va='top', color=color)


                if ylim is not None:
                    ax.set_ylim(ylim)
                if xlim is not None:
                    ax.set_xlim(xlim)

                ticks = boxplot.get_yticks() if use_y else boxplot.get_xticks()
                ticklabels = [tick.get_text() for tick in
                               (boxplot.get_yticklabels() if use_y else boxplot.get_xticklabels())]

                if use_elo:
                    # ----- add elo error bars -----
                    # Get the bar positions


                    # Add asymmetric error bars manually
                    for pos, framework_type in zip(ticks, ticklabels):
                        for tune_method, errcolor in zip(["default", "tuned", "tuned_ensembled", "holdout_tuned_ensembled"], errcolors):
                            row = df_plot.loc[(df_plot["framework_type"] == framework_type) & (df_plot["tune_method"] == tune_method)]
                            if len(row) == 1:
                                # not all methods have tuned or tuned_ensembled
                                y = row['elo'].values
                                yerr_low = row['elo-'].values
                                yerr_high = row['elo+'].values
                                if use_y:
                                    plotline, caps, barlinecols = plt.errorbar(y, pos, xerr=[yerr_low, yerr_high],
                                                                               fmt='none', color=errcolor,
                                                                               alpha=err_alpha,
                                                                               linewidth=err_linewidths[tune_method])
                                else:
                                    plotline, caps, barlinecols = plt.errorbar(pos, y, yerr=[yerr_low, yerr_high],
                                                                               fmt='none', color=errcolor,
                                                                               alpha=err_alpha,
                                                                               linewidth=err_linewidths[tune_method])
                                # don't round because it will make the lines longer
                                # plt.setp(barlinecols[0], capstyle="round")


                # ----- highlight bars that contain imputed results -----

                # Map x-tick positions to category labels
                label_lookup = dict(zip(ticks, ticklabels))

                has_imputed = False

                for i, bar in enumerate(boxplot.patches):
                    # Get position and convert to category label
                    pos = bar.get_y() + bar.get_height() / 2 if use_y else bar.get_x() + bar.get_width() / 2
                    category_index = round(pos)  # x-ticks are usually 0, 1, 2, ...
                    category = label_lookup.get(category_index)

                    if category in imputed_names:
                        has_imputed = True
                        bar.set_hatch('xx')

                if not use_y:
                    # ----- alternate rows of x tick labels -----
                    # Get current x tick labels
                    labels = [label.get_text() for label in boxplot.get_xticklabels()]

                    # Add newline to every second label
                    new_labels = [label if i % 2 == 0 else r'$\uparrow$' + '\n' + label for i, label in enumerate(labels)]

                    if has_non_baselines:
                        # Apply modified labels
                        boxplot.set_xticks(labels)
                        boxplot.set_xticklabels(new_labels)

                # remove unnecessary extra space on the sides
                if use_y:
                    plt.ylim(len(boxplot.get_yticklabels()) - 0.35, -0.65)
                else:
                    plt.xlim(-0.5, len(boxplot.get_xticklabels()) - 0.5)


                # ax.legend(loc="upper center", ncol=5)
                # these are not the final legend parameters, see below
                ax.legend(loc="upper center", bbox_to_anchor=[0.5, 1.02])

                # reordering the labels
                handles, labels = ax.get_legend_handles_labels()

                # this doesn't work, it also removes the hatch from the actual bars in the plot
                # for handle in handles:
                #     patches = []
                #     if isinstance(handle, Patch):
                #         patches = [handle]
                #     elif isinstance(handle, BarContainer):
                #         patches = handle.patches
                #     for patch in patches:
                #         # remove hatch from existing handles
                #         # It can be present if one of the imputed methods is the best method, e.g., for multiclass
                #         patch.set(hatch=None)

                if has_imputed:
                    # Create a custom legend patch for "imputed"
                    imputed_patch = Patch(facecolor='gray', edgecolor='white', hatch='xx', label='Partially imputed')

                    # Add to existing legend
                    handles.append(imputed_patch)
                    labels.append("Partially imputed")

                # quick fix
                is_holdout_plot = "Tuned + Ensembled (Holdout)" in labels
                if is_holdout_plot:
                    valid_idxs = [i for i, label in enumerate(labels) if label != "Default"]
                    labels = [labels[i] for i in valid_idxs]
                    handles = [handles[i] for i in valid_idxs]

                # specify order
                # len_baselines = len(baselines)
                # len_baselines = 0  # we don't put them in the legend anymore
                # num_other = len(labels) - len_baselines
                # order = [n + len_baselines for n in range(num_other)] + [n for n in range(len_baselines)]
                # order = [3, 4, 5, 0, 1, 2]
                order = list(range(len(labels)))
                order = list(reversed(order))
                # if len(order) == 3:
                #     order = [2, 1, 0]

                # pass handle & labels lists along with order as below
                ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
                          ncol=(len(labels)+1)//2 if has_imputed and use_y else len(labels),
                          bbox_to_anchor=[0.35 if use_y else 0.5, 1.05])

                # if use_y:
                #     boxplot.margins(y=0.05)
                # else:
                #     boxplot.margins(x=0.05)

                # ax.legend(bbox_to_anchor=[0.1, 0.5], loc='center left', ncol=5)
                plt.tight_layout()

                if save_prefix:
                    if name_suffix is None:
                        name_suffix = ""
                    fig_path = Path(save_prefix)
                    fig_path.mkdir(parents=True, exist_ok=True)
                    if use_gmean:
                        fig_name = f"tuning-impact-gmean{name_suffix}.{self.figure_file_type}"
                    else:
                        fig_name = f"tuning-impact{name_suffix}.{self.figure_file_type}"
                    fig_save_path = fig_path / fig_name
                    plt.savefig(fig_save_path, dpi=300)
                if show:
                    plt.show()
                plt.close()

    def plot_tabarena_times(self, df: pd.DataFrame, output_dir: Path | str, show: bool = True):
        df = df.copy()

        datasets_impute_freq = df.groupby("dataset")["imputed"].mean()
        datasets_no_impute = list(datasets_impute_freq[datasets_impute_freq <= 0].index)

        df = df[df["dataset"].isin(datasets_no_impute)]

        framework_types = self._get_config_types(df_results=df)

        f_map, f_map_type, f_map_inverse, f_map_type_name = self.get_framework_type_method_names(
            framework_types=framework_types,
        )

        df["framework_type"] = df[self.method_col].map(f_map_type).fillna(df[self.method_col])
        df["tune_method"] = df[self.method_col].map(f_map_inverse).fillna("default")
        df = df[df["tune_method"].isin(["default", "tuned_ensembled"])]
        df = df[df['framework_type'].isin(framework_types)]
        df.loc[:, "framework_type"] = df["framework_type"].map(f_map_type_name).fillna(df["framework_type"])

        gpu_methods = ['TabICL', 'TabDPT', 'TabPFNv2', "ModernNCA", "TabM"]

        # add device name
        framework_types = df["framework_type"].unique()
        device_map = {
            ft: f'{ft} ' + r'(GPU)' if ft in gpu_methods else f'{ft} (CPU)' if not ft.endswith("(CPU)") else ft for ft in framework_types
        }
        device_map = {}
        for ft in framework_types:
            if ft in gpu_methods:
                ft_new = f'{ft} (GPU)'
            elif ft.endswith("(CPU)"):
                ft_new = ft
            elif ft.endswith("(GPU)"):
                ft_new = ft
            else:
                ft_new = f'{ft} (CPU)'
            device_map[ft] = ft_new

        df["framework_type"] = df["framework_type"].map(device_map).fillna(df["framework_type"])

        # take mean times
        df = df.groupby(['dataset', 'framework_type', 'tune_method'])[['time_train_s_per_1K', 'time_infer_s_per_1K']].mean().reset_index()
        df = df.groupby(['framework_type', 'tune_method'])[['time_train_s_per_1K', 'time_infer_s_per_1K']].median().reset_index()

        # ----- ChatGPT plotting code -----

        # Unique values for mapping
        # Sort frameworks by max train time
        sorted_frameworks = (
            df.groupby('framework_type')['time_train_s_per_1K']
            .min()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        frameworks = sorted_frameworks
        y_positions = np.arange(len(frameworks))

        # Maps for tuning method to color and marker
        tune_methods = df['tune_method'].unique()
        # color_map = {tm: c for tm, c in zip(tune_methods, plt.cm.tab10.colors)}
        sns_colors = sns.color_palette("muted").as_hex()
        # sns_colors = sns.color_palette("pastel").as_hex()
        color_map = {'default': sns_colors[0], 'tuned': sns_colors[1], 'tuned_ensembled': sns_colors[2]}
        marker_list = ['o', 's', '^', 'D', 'P', '*', 'X', 'v']
        marker_map = {tm: m for tm, m in zip(tune_methods, marker_list)}

        # Create side-by-side subplots with shared y-axis
        fig, (ax_train, ax_infer) = plt.subplots(
            1, 2, sharey=True, figsize=(5, 4)
        )

        # Alternate row background on both axes
        for i in range(0, len(frameworks), 2):
            for ax in [ax_train, ax_infer]:
                ax.axhspan(i - 0.5, i + 0.5, facecolor='lightgray', alpha=0.3, zorder=0)

        # Plot training and inference times
        for i, fw in enumerate(frameworks):
            df_fw = df[df['framework_type'] == fw]
            for _, row in df_fw.iterrows():
                color = color_map[row['tune_method']]
                marker = marker_map[row['tune_method']]
                ax_train.plot(row['time_train_s_per_1K'], i, marker=marker, color=color, linestyle='None')
                ax_infer.plot(row['time_infer_s_per_1K'], i, marker=marker, color=color, linestyle='None')

        # Train time axis
        ax_train.set_xscale('log')
        ax_train.set_xlabel("Median time per 1K samples [s]")
        ax_train.set_title(r"\textbf{Train+val time}", fontweight='bold')
        ax_train.set_yticks(y_positions)
        ax_train.set_yticklabels(frameworks, fontsize=10)
        ax_train.grid(True, axis='x', alpha=0.5)

        # Inference time axis
        ax_infer.set_xscale('log')
        ax_infer.set_xlabel("Median time per 1K samples [s]")
        ax_infer.set_title(r"\textbf{Inference time}", fontweight='bold')
        ax_infer.set_yticks(y_positions)
        ax_infer.tick_params(labelleft=False)  # Explicitly hide y-tick labels
        ax_infer.grid(True, axis='x', alpha=0.5)

        for ax in [ax_train, ax_infer]:
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
            # ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
            # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            # ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        tune_method_display_names = {
            'default': 'Default',
            'tuned': 'Tuned',
            'tuned_ensembled': 'Tuned + Ensembled'
        }

        # Add legend above both plots
        legend_elements = [
            plt.Line2D([0], [0], marker=marker_map[tm], color=color_map[tm],
                       linestyle='None', label=tune_method_display_names[tm], markersize=8)
            for tm in tune_methods
        ]
        fig.legend(handles=legend_elements,  # title='Tuning Method',
                   loc='upper center', bbox_to_anchor=(0.65, 1.01), ncol=3, fontsize=10, title_fontsize=11)

        # Layout adjustment (no clipping)
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_dir / f'time_plot.{self.figure_file_type}')
        if show:
            plt.show()
        plt.close(fig)

    def get_ensemble_weights(
        self,
        df_results: pd.DataFrame,
        method: str,
        excluded_families: list[str] = None,
        aggregate_folds: bool = False,
    ) -> pd.DataFrame:
        if self.datasets is not None:
            df_results = df_results[df_results["dataset"].isin(self.datasets)]
        if excluded_families is None:
            excluded_families = []

        df_results_method = df_results[df_results[self.method_col] == method]

        df_ensemble_weights = df_results_method[["dataset", "fold", "ensemble_weight"]]

        full_dict = []
        # available_configs = set()
        # for ensemble_weights in df_ensemble_weights["ensemble_weight"].values:
        #     for k in ensemble_weights.keys():
        #         if k not in available_configs:
        #             available_configs.add(k)
        #     ens_weights_w_dataset_fold = ensemble_weights.copy(deep=True)
        #     full_dict.append(ensemble_weights)
        #
        # df_ensemble_weights_2 = pd.DataFrame()

        for d, f, ensemble_weights in zip(df_ensemble_weights["dataset"], df_ensemble_weights["fold"], df_ensemble_weights["ensemble_weight"]):
            if isinstance(ensemble_weights, str):
                ensemble_weights = json.loads(ensemble_weights)
            assert isinstance(ensemble_weights, dict)
            ens_weights_w_dataset_fold = dict()
            ens_weights_w_dataset_fold["dataset"] = d
            ens_weights_w_dataset_fold["fold"] = f
            ens_weights_w_dataset_fold.update(ensemble_weights)
            full_dict.append(ens_weights_w_dataset_fold)
            pass

        model_to_families = self.config_types

        model_families = set()
        for m, f in model_to_families.items():
            if f not in model_families:
                model_families.add(f)

        weight_per_family_dict = []
        for cur_dict in full_dict:
            new_dict = {}
            for k, v in cur_dict.items():
                if k == "dataset":
                    new_dict["dataset"] = v
                elif k == "fold":
                    new_dict["fold"] = v
                else:
                    model_family = model_to_families[k]
                    if v is not None:
                        if model_family not in new_dict:
                            new_dict[model_family] = 0
                        new_dict[model_family] += v
            weight_per_family_dict.append(new_dict)

        import pandas as pd
        df = pd.DataFrame(weight_per_family_dict)
        df = df.set_index(["dataset", "fold"])
        df = df.fillna(0)

        df_cols = df.columns
        f_to_add = []
        for f in model_families:
            if f not in df_cols:
                f_to_add.append(f)
        df[f_to_add] = 0

        if excluded_families:
            df = df.drop(columns=excluded_families)

        df = self._get_ensemble_weights(
            df_ensemble_weights=df,
            aggregate_folds=aggregate_folds,
            sort_by_mean=True,
        )

        return df

    def _process_baselines(
        self,
        df_results: pd.DataFrame, baselines: list[str] | None | str,
        baseline_colors: list[str] | None,
    ) -> tuple[list[str], list[str]]:
        methods = df_results[self.method_col].unique()

        if baselines is None:
            baselines = []
        elif baselines == "auto":
            baselines = list(
                df_results[
                    df_results["method_type"].isin(["baseline", "portfolio"]) |
                    ((df_results["method_type"] == "config") & df_results["method_subtype"].isna())
                ][self.method_col].unique()
            )
        else:
            missing_baselines = [b for b in baselines if b not in methods]
            if missing_baselines:
                print(f"Missing specified baselines: {missing_baselines}")
        if baseline_colors is None:
            default_baseline_colors = [
                "black",
                "purple",
                "blue",
                "red",
                "darkgray",
            ]
            # Assign colors dynamically, cycling if baselines > baseline_colors
            baseline_colors = list(itertools.islice(itertools.cycle(default_baseline_colors), len(baselines)))
        assert len(baselines) == len(baseline_colors)
        assert len(baselines) == len(list(set(baselines))), f"Duplicates keys found in baselines: {baselines}"
        # Filter both baselines and baseline_colors using the same mask
        filtered = [(b, c) for b, c in zip(baselines, baseline_colors) if b in methods]

        baselines = [b for b, _ in filtered]
        baseline_colors = [c for _, c in filtered]
        return baselines, baseline_colors

    @classmethod
    def _get_ensemble_weights(
        cls,
        df_ensemble_weights: pd.DataFrame,
        aggregate_folds: bool = True,
        sort_by_mean: bool = True,
    ) -> pd.DataFrame:
        df_ensemble_weights = copy.deepcopy(df_ensemble_weights)
        if aggregate_folds:
            df_ensemble_weights = df_ensemble_weights.groupby(level='dataset').mean()
        else:
            index_new = list(df_ensemble_weights.index.to_flat_index())
            index_new = [str(t[0]) + "_" + str(t[1]) for t in index_new]
            df_ensemble_weights.index = index_new

        if sort_by_mean:
            s = df_ensemble_weights.sum()
            df_ensemble_weights = df_ensemble_weights[s.sort_values(ascending=False).index]
        return df_ensemble_weights

    def get_framework_type_method_names(self, framework_types):
        return get_framework_type_method_names(
            framework_types=framework_types,
            max_runtimes=[
                (3600 * 4, "_4h"),
                (None, None),
            ],
            f_map_type_name=self.get_method_rename_map(),
        )

    # TODO: aggregate_config_family: bool
    # TODO: sort rows by size? color by problem type?
    def _plot_ensemble_weights_heatmap(
        self,
        df_ensemble_weights: pd.DataFrame,
        aggregate_folds: bool = True,
        sort_by_mean: bool = True,
        include_mean: bool = True,
        **kwargs,
    ):
        """

        Parameters
        ----------
        df_ensemble_weights : pd.DataFrame
            The 2nd output object of `repo.evaluate_ensembles(...)
        aggregate_folds : bool, default True
            If True, averages folds of datasets together into single rows representing a dataset.
            If False, each fold of each dataset will be its own row.
        sort_by_mean : bool, default True
            If True, will sort columns by the mean value of the column.
            If False, columns will remain in the original order.
        include_mean : bool, default True
            If True, will add a row at the bottom with label "mean" representing the mean of the config weights across all tasks.
            NaN values are considered 0 for the purposes of calculating the mean.
        **kwargs
            Passed to the `create_heatmap` function

        Returns
        -------
        plt

        """
        # df_ensemble_weights = self.get_ensemble_weights(
        #     df_ensemble_weights=df_ensemble_weights,
        #     aggregate_folds=aggregate_folds,
        #     sort_by_mean=sort_by_mean,
        # )

        p = create_heatmap(df=df_ensemble_weights, include_mean=include_mean, **kwargs)
        return p

    def plot_ensemble_weights_heatmap(self, df_ensemble_weights: pd.DataFrame, **kwargs):
        # FIXME: if family never present, then this won't work
        p = self._plot_ensemble_weights_heatmap(df_ensemble_weights=df_ensemble_weights, **kwargs)
        fig_path = Path(f"{self.output_dir}/figures")
        fig_path.mkdir(parents=True, exist_ok=True)
        p.savefig(fig_path / f"ens-weights-per-dataset.{self.figure_file_type}")

    # FIXME: clean this up
    def generate_runtime_plot(self, df_results: pd.DataFrame):
        df_results_configs = df_results[df_results["method_type"] == "config"]
        df_results_configs = df_results_configs.copy(deep=True)

        framework_types = self._get_config_types(df_results=df_results_configs)
        df_results_configs = df_results_configs[df_results_configs["config_type"].isin(framework_types)]

        method_rename_map = self.get_method_rename_map()
        df_results_configs["config_type"] = df_results_configs["config_type"].map(method_rename_map).fillna(df_results_configs["config_type"])

        plot_train_time_deep_dive(
            df=df_results_configs,
            expname_outdir=self.output_dir,
            method_col=self.method_col,
            family_col="config_type",
            show=False
        )


class TabArenaEvaluator_2025_06_12(TabArenaEvaluator):
    def get_method_rename_map(self) -> dict[str, str]:
        method_rename_map = super().get_method_rename_map()
        method_rename_map["REALMLP"] = "RealMLP"
        method_rename_map["REALMLP_GPU"] = "RealMLP (GPU)"
        return method_rename_map
