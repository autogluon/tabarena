from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabarena.evaluation._fillna import fillna_metrics
from tabarena.repository.repo_utils import convert_time_infer_s_from_sample_to_batch

if TYPE_CHECKING:
    from tabarena.repository import EvaluationRepository, EvaluationRepositoryCollection


# TODO: This class is WIP.
# TODO: Add unit tests
class RepoMetrics:
    """Assembles per-``(dataset, fold, framework)`` metric tables and zeroshot portfolios from a repository."""

    def __init__(
        self,
        repo: EvaluationRepository | EvaluationRepositoryCollection,
    ):
        self.repo = repo

    def assemble_metrics(
        self,
        results_df: pd.DataFrame = None,
        datasets: list[str] | None = None,
        folds: list[int] | None = None,
        configs: list[str] | None = None,
        baselines: list[str] | None = None,
        convert_from_sample_to_batch: bool = False,
        keep_extra_columns: bool = False,
        include_metric_error_val: bool = False,
        fillna: bool | str = "auto",
    ) -> pd.DataFrame:
        """Assemble one per-``(dataset, fold, framework)`` metrics table from the repository.

        Concatenates the repo's per-config metrics with its baseline metrics (when present),
        optionally prepended with an externally-produced ``results_df``, then restricts the rows
        to the requested ``datasets`` / ``folds`` / ``configs`` / ``baselines``. The kept columns
        are ``metric_error``, ``time_train_s``, ``time_infer_s``, ``metric`` and ``problem_type``
        (plus ``metric_error_val`` when ``include_metric_error_val``, and any ``aux_metric*``
        columns the configs carry). This does not rank or compare methods; it only gathers their
        per-task metrics into a single indexed frame for a downstream evaluator/reporter.

        Parameters
        ----------
        results_df : pd.DataFrame, optional
            Externally-produced results to include alongside the repo's configs/baselines (e.g. a
            freshly simulated method). Re-indexed to ``(dataset, fold, framework)``.
        datasets, folds, configs, baselines : list, optional
            Restrict the table to these datasets / folds / config names / baseline names. ``None``
            keeps all (all datasets of the repo by default).
        convert_from_sample_to_batch : bool, default False
            Convert ``time_infer_s`` from per-sample to per-batch (requires ``repo.task_metadata``).
        keep_extra_columns : bool, default False
            Keep columns of ``results_df`` beyond the standard metric columns instead of dropping
            them.
        include_metric_error_val : bool, default False
            Also include the validation-set ``metric_error_val`` column.
        fillna : bool or "auto", default "auto"
            Impute missing ``(dataset, fold, framework)`` rows from the repo's fallback config
            (``repo._config_fallback``), adding a boolean ``imputed`` column and an
            ``impute_method`` column. ``"auto"`` imputes iff a fallback config is set.

        Returns:
        -------
        pd.DataFrame
            Metrics indexed by ``(dataset, fold, framework)`` and sorted by index.
        """
        if datasets is None:
            datasets = self.repo.datasets()
        columns = ["metric_error", "time_train_s", "time_infer_s", "metric", "problem_type"]
        aux_columns = ["aux_metric", "aux_metric_error", "aux_metric_error_val"]

        if results_df is not None:
            df_exp = results_df
            if results_df.index.names != [None]:
                df_exp = df_exp.reset_index()
            df_exp = df_exp.set_index(["dataset", "fold", "framework"])
            if not keep_extra_columns:
                df_exp = df_exp[columns]
            else:
                extra_columns = [c for c in df_exp.columns if c not in columns]
                df_exp = df_exp[columns + extra_columns]
        else:
            df_exp = None

        config_columns = columns
        if include_metric_error_val:
            config_columns = [*config_columns, "metric_error_val"]
        df_configs = self.repo._zeroshot_context.df_configs
        config_aux_columns = [c for c in aux_columns if c in df_configs.columns]
        # Dropping task column in df_tr
        df_tr = df_configs.set_index(["dataset", "fold", "framework"])[config_columns + config_aux_columns]

        mask = df_tr.index.get_level_values("dataset").isin(datasets)
        if folds is not None:
            mask = mask & df_tr.index.get_level_values("fold").isin(folds)
        if configs is not None:
            mask = mask & df_tr.index.get_level_values("framework").isin(configs)
        df_tr = df_tr[mask]

        if self.repo.task_metadata is not None and convert_from_sample_to_batch:
            df_tr = convert_time_infer_s_from_sample_to_batch(df_tr, repo=self.repo)

        if self.repo._zeroshot_context.df_baselines is not None:
            df_baselines = self.repo._zeroshot_context.df_baselines.set_index(["dataset", "fold", "framework"])
            baseline_columns = columns
            if include_metric_error_val:
                baseline_columns = [*baseline_columns, "metric_error_val"]
                if "metric_error_val" not in df_baselines:
                    df_baselines["metric_error_val"] = np.nan
            df_baselines = df_baselines[baseline_columns]

            mask = df_baselines.index.get_level_values("dataset").isin(datasets)
            if folds is not None:
                mask = mask & df_baselines.index.get_level_values("fold").isin(folds)
            if baselines is not None:
                mask = mask & df_baselines.index.get_level_values("framework").isin(baselines)
            df_baselines = df_baselines[mask]

            if self.repo.task_metadata is not None and convert_from_sample_to_batch:
                df_baselines = convert_time_infer_s_from_sample_to_batch(df_baselines, repo=self.repo)
        else:
            if baselines:
                raise AssertionError(f"Baselines specified but no baseline methods exist! (baselines={baselines})")
            df_baselines = None

        dfs_to_concat = [df_exp, df_tr, df_baselines]
        dfs_to_concat = [df for df in dfs_to_concat if df is not None and len(df) > 0]

        df = pd.concat(dfs_to_concat, axis=0)
        df = df.sort_index()

        if isinstance(fillna, str):
            assert fillna == "auto"
            fillna = self.repo._config_fallback is not None

        if fillna:
            assert self.repo._config_fallback is not None
            df_fillna = self.assemble_metrics(
                configs=[self.repo._config_fallback],
                baselines=[],
                datasets=datasets,
                folds=folds,
                fillna=False,
                keep_extra_columns=keep_extra_columns,
                include_metric_error_val=include_metric_error_val,
            )
            df_fillna = df_fillna.droplevel("framework")
            df = self._fillna_metrics(df_metrics=df, df_fillna=df_fillna)

            df["impute_method"] = np.nan
            df["impute_method"] = df["impute_method"].astype(object)
            df.loc[df[df["imputed"]].index, "impute_method"] = self.repo._config_fallback

        return df

    @classmethod
    def _fillna_metrics(cls, df_metrics: pd.DataFrame, df_fillna: pd.DataFrame) -> pd.DataFrame:
        """Fills missing (dataset, fold, framework) rows in df_metrics with the (dataset, fold) row in df_fillna.

        Thin adapter over :func:`tabarena.evaluation._fillna.fillna_metrics` (keyed on ``framework``)
        that accepts and returns frames indexed by ``(dataset, fold, framework)`` /
        ``(dataset, fold)``; the shared helper works on flat frames.

        Parameters
        ----------
        df_metrics
            Per-``(dataset, fold, framework)`` metrics, indexed by that triple.
        df_fillna
            Fallback rows indexed by ``(dataset, fold)`` (the framework level already dropped).

        Returns:
        -------
        pd.DataFrame
            ``df_metrics`` completed and indexed by ``(dataset, fold, framework)``, with an
            ``imputed`` column.

        """
        df_filled = fillna_metrics(
            df_to_fill=df_metrics.reset_index(),
            df_fillna=df_fillna.reset_index(),
            key_col="framework",
        )
        return df_filled.set_index(["dataset", "fold", "framework"])

    # TODO: Prototype, find a better way to do this
    # TODO: Docstring
    def compute_avg_config_prediction_delta(
        self,
        configs: list[str],
        datasets: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Parameters
        ----------
        configs
        datasets
        folds

        Returns:
        -------
        delta_avg_abs_mean : pd.DataFrame
            The per-task normalized mean of the absolute delta in cells between two configs' test predictions.
            Maximum value is 1 (maximum disagreement), minimum value is 0 (identical configs).
        delta_avg_std : pd.DataFrame
            The per-task normalized mean of the standard deviation of the delta in cells between two configs' test predictions.
            Maximum value is 1 (maximum disagreement), minimum value is 0 (identical configs).
        """
        if datasets is None:
            datasets = self.repo.datasets()
        import numpy as np

        delta_comparison = {}
        delta_std_comparison = {}
        for dataset in datasets:
            folds = self.repo.dataset_to_folds(dataset=dataset)
            for fold in folds:
                delta_comparison[(dataset, fold)] = {}
                delta_std_comparison[(dataset, fold)] = {}
                print(f"{dataset} | {fold}")
                # TODO: Only load y_pred_test 1 time for each task+config
                # TODO: don't need to do (x, y) and (y, x)
                for compare_conf1 in configs:
                    for compare_conf2 in configs:
                        if compare_conf1 == compare_conf2:
                            continue
                        print(f"{compare_conf1} vs {compare_conf2}")
                        y_pred_test1 = self.repo.predict_test(dataset=dataset, fold=fold, config=compare_conf1)
                        y_pred_test2 = self.repo.predict_test(dataset=dataset, fold=fold, config=compare_conf2)
                        delta = y_pred_test2 - y_pred_test1
                        # print(delta)
                        np.mean(delta)
                        abs_mean = np.mean(np.abs(delta))
                        stddev = np.std(delta)
                        print(f"\t{abs_mean:.3f}\t{stddev:.3f}")
                        delta_comparison[(dataset, fold)][(compare_conf1, compare_conf2)] = abs_mean
                        delta_std_comparison[(dataset, fold)][(compare_conf1, compare_conf2)] = stddev
                # normalize
                max_abs_mean = 0
                max_std = 0
                for k, v in delta_comparison[(dataset, fold)].items():
                    max_abs_mean = max(max_abs_mean, v)
                for k, v in delta_std_comparison[(dataset, fold)].items():
                    max_std = max(max_std, v)
                for k in delta_comparison[(dataset, fold)]:
                    if max_abs_mean != 0:
                        delta_comparison[(dataset, fold)][k] /= max_abs_mean
                    if max_std != 0:
                        delta_std_comparison[(dataset, fold)][k] /= max_std
                # FIXME: FINISH

        delta_avg_abs_mean = pd.DataFrame(delta_comparison).mean(axis=1).unstack().fillna(0)
        delta_avg_std = pd.DataFrame(delta_std_comparison).mean(axis=1).unstack().fillna(0)

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)

        pca.fit(delta_avg_abs_mean)
        delta_avg_abs_mean_projection = pca.transform(delta_avg_abs_mean)

        # FIXME: Make plotting optional
        from matplotlib import pyplot as plt

        _fig, ax = plt.subplots(figsize=(10, 10))
        # config_names = list(delta_avg_abs_mean.index)
        delta_avg_abs_mean_projection = pd.DataFrame(delta_avg_abs_mean_projection, index=delta_avg_abs_mean.index)
        for config in delta_avg_abs_mean_projection.index:
            ax.scatter(
                delta_avg_abs_mean_projection.loc[config, 0], delta_avg_abs_mean_projection.loc[config, 1], label=config
            )
        ax.legend()
        ax.grid(True)
        plt.savefig("pca_projection_test")

        return delta_avg_abs_mean, delta_avg_std
