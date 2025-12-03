from __future__ import annotations

from typing import Type

import numpy as np
import pandas as pd

from tabarena.simulation.ensemble_selection_config_scorer import EnsembleScorer, EnsembleScorerMaxModels
from tabarena.portfolio.greedy_portfolio_generator import zeroshot_results
from ..repository import EvaluationRepository, EvaluationRepositoryCollection
from ..repository.repo_utils import convert_time_infer_s_from_sample_to_batch


# TODO: This class is WIP.
# TODO: Add unit tests
class Evaluator:
    """
    Computes metrics and statistics to compare methods.
    """
    def __init__(
        self,
        repo: EvaluationRepository | EvaluationRepositoryCollection,
    ):
        self.repo = repo

    def compare_metrics(
        self,
        results_df: pd.DataFrame = None,
        datasets: list[str] = None,
        folds: list[int] = None,
        configs: list[str] = None,
        baselines: list[str] = None,
        convert_from_sample_to_batch: bool = False,
        keep_extra_columns: bool = False,
        include_metric_error_val: bool = False,
        fillna: bool | str = "auto",
    ) -> pd.DataFrame:
        if datasets is None:
            datasets = self.repo.datasets()
        columns = ["metric_error", "time_train_s", "time_infer_s", "metric", "problem_type"]

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
            config_columns = config_columns + ["metric_error_val"]
        # Dropping task column in df_tr
        df_tr = self.repo._zeroshot_context.df_configs.set_index(["dataset", "fold", "framework"])[config_columns]

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
                baseline_columns = baseline_columns + ["metric_error_val"]
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
            df_fillna = self.compare_metrics(
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
        """
        Fills missing (dataset, fold, framework) rows in df_metrics with the (dataset, fold) row in df_fillna.

        Parameters
        ----------
        df_metrics
        df_fillna

        Returns
        -------

        """
        # FIXME: Include frameworks that have 0 presence?
        #  baselines + configs
        unique_frameworks = list(df_metrics.index.unique(level="framework"))

        df_filled = df_fillna.index.to_frame().merge(
            pd.Series(data=unique_frameworks, name="framework"),
            how="cross",
        )
        df_filled = df_filled.set_index(keys=list(df_filled.columns))

        # missing results
        nan_vals = df_filled.index.difference(df_metrics.index)

        # fill valid values
        fill_cols = list(df_metrics.columns)
        df_filled[fill_cols] = np.nan
        df_filled[fill_cols] = df_filled[fill_cols].astype(df_metrics.dtypes)
        df_filled.loc[df_metrics.index] = df_metrics

        a = df_fillna.loc[nan_vals.droplevel(level="framework")]
        a.index = nan_vals
        df_filled.loc[nan_vals] = a

        df_filled["imputed"] = False
        df_filled.loc[nan_vals, "imputed"] = True

        df_metrics = df_filled

        return df_metrics

    # TODO: Prototype, find a better way to do this
    # TODO: Docstring
    def compute_avg_config_prediction_delta(
        self,
        configs: list[str],
        datasets: list[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """


        Parameters
        ----------
        configs
        datasets
        folds

        Returns
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
                        mean = np.mean(delta)
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
                for k in delta_comparison[(dataset, fold)].keys():
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
        fig, ax = plt.subplots(figsize=(10, 10))
        # config_names = list(delta_avg_abs_mean.index)
        delta_avg_abs_mean_projection = pd.DataFrame(delta_avg_abs_mean_projection, index=delta_avg_abs_mean.index)
        for config in delta_avg_abs_mean_projection.index:
            ax.scatter(delta_avg_abs_mean_projection.loc[config, 0], delta_avg_abs_mean_projection.loc[config, 1], label=config)
        ax.legend()
        ax.grid(True)
        plt.savefig("pca_projection_test")

        return delta_avg_abs_mean, delta_avg_std

    # TODO: WIP
    # TODO: Add a non-loo version
    # TODO: Rename
    # FIXME: Make it work with framework_types + max_models_per_type
    def zeroshot_portfolio(
        self,
        configs: list[str] | None = None,
        n_portfolios: int = 200,  # FIXME
        n_ensemble: int | None = None,
        time_limit: float | None = 14400,
        engine: str = "ray",
        rename_columns: bool = True,  # TODO: Align them automatically so this isn't needed
        n_ensemble_in_name: bool = True,
        n_max_models_per_type: int | str | None = None,
        n_eval_folds: int | None = None,
        ensemble_cls: Type[EnsembleScorer] = EnsembleScorerMaxModels,
        ensemble_kwargs: dict = None,
    ) -> pd.DataFrame:
        repo = self.repo

        if configs is None:
            configs = repo.configs()

        if n_eval_folds is None:
            n_eval_folds = repo.n_folds()

        a = zeroshot_results(
            repo=repo,
            dataset_names=repo.datasets(),
            n_portfolios=[n_portfolios],
            n_ensembles=[n_ensemble],
            max_runtimes=[time_limit],
            n_ensemble_in_name=n_ensemble_in_name,
            n_max_models_per_type=[n_max_models_per_type],
            n_eval_folds=n_eval_folds,
            configs=configs,
            engine=engine,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
        )

        df_zeroshot_portfolio = pd.DataFrame(a)

        if rename_columns:
            df_zeroshot_portfolio = df_zeroshot_portfolio.rename(columns={
                "metadata": "method_metadata",
            })
            datasets_info = repo.datasets_info()

            df_zeroshot_portfolio["problem_type"] = df_zeroshot_portfolio["dataset"].map(datasets_info["problem_type"])
            df_zeroshot_portfolio["metric"] = df_zeroshot_portfolio["dataset"].map(datasets_info["metric"])

        return df_zeroshot_portfolio
