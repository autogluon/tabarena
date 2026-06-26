"""``RepoSimulator`` — post-hoc TabArena method simulation over a processed repository.

Given a processed :class:`~tabarena.repository.EvaluationRepository` (cached out-of-fold and
test predictions), this computes the *derived* leaderboard methods on top of it without
re-fitting anything: per-config-family HPO (``tuned`` / ``tuned + ensemble``), zeroshot
portfolios, single-config and config-family results, and baselines. The output rows carry the
``method_type`` / ``method_subtype`` / ``config_type`` columns the leaderboard consumes.

This is the repo-level engine. :class:`~tabarena.models._method_simulator.MethodSimulator` is the
per-:class:`~tabarena.models._method_metadata.MethodMetadata` facade that drives one of these over
a single method's processed results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from tabarena.evaluation.repo_metrics import RepoMetrics
from tabarena.portfolio.greedy_portfolio_generator import zeroshot_results
from tabarena.simulation.ensemble_selection_config_scorer import EnsembleScorer, EnsembleScorerMaxModels

if TYPE_CHECKING:
    from tabarena.repository import EvaluationRepository


class RepoSimulator:
    def __init__(
        self, repo: EvaluationRepository, output_dir: str | None = None, backend: Literal["ray", "native"] = "ray"
    ):
        self.repo = repo
        self.repo_metrics = RepoMetrics(repo=self.repo)
        self.output_dir = output_dir
        self.backend = backend
        assert self.backend in ["ray", "native"]
        if self.backend == "ray":
            self.engine = "ray"
        else:
            self.engine = "sequential"

    def get_config_type_groups(self) -> dict:
        config_type_groups = {}
        configs_type = self.repo.configs_type()
        all_configs = self.repo.configs()
        for c in all_configs:
            if configs_type[c] not in config_type_groups:
                config_type_groups[configs_type[c]] = []
            config_type_groups[configs_type[c]].append(c)

        return config_type_groups

    def run_hpo(self, model_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_results_family_hpo_ens = self.run_ensemble_config_type(
            config_type=model_type,
            fit_order="original",
            seed=0,
            n_iterations=40,
            time_limit=None,
        )
        df_results_family_hpo_ens["framework"] = f"{model_type} (tuned + ensemble)"

        df_results_family_hpo = self.run_ensemble_config_type(
            config_type=model_type,
            fit_order="original",
            seed=0,
            n_iterations=1,
            time_limit=None,
        )
        df_results_family_hpo["framework"] = f"{model_type} (tuned)"
        return df_results_family_hpo, df_results_family_hpo_ens

    def run_hpo_by_family(self, model_types: list[str] | None = None) -> pd.DataFrame:
        config_type_groups = self.get_config_type_groups()

        hpo_results_lst = []

        if model_types is None:
            model_types = list(config_type_groups.keys())
        for family in model_types:
            assert family in config_type_groups, (
                f"Model family {family} missing from available families: {list(config_type_groups.keys())}"
            )

        for family in model_types:
            df_results_family_hpo, df_results_family_hpo_ens = self.run_hpo(model_type=family)
            hpo_results_lst += [df_results_family_hpo, df_results_family_hpo_ens]

        return pd.concat(hpo_results_lst, ignore_index=True)

    def run_ensemble_config_type(
        self,
        config_type: str | list[str],
        n_iterations: int,
        n_configs: int | None = None,
        fixed_configs: list[str] | None = None,
        time_limit: float | None = None,
        fit_order: Literal["original", "random"] = "original",
        seed: int = 0,
        **kwargs,
    ) -> pd.DataFrame:
        # FIXME: Don't recompute this each call, implement `self.repo.configs(config_types=[config_type])`
        config_type_groups = self.get_config_type_groups()
        if isinstance(config_type, list):
            configs = []
            for ct in config_type:
                configs += config_type_groups[ct]
        else:
            configs = config_type_groups[config_type]

        if fixed_configs is not None:
            for c in fixed_configs:
                assert c in configs, f"config {c!r} missing from configs (config_type={config_type!r}!"
            configs = [c for c in configs if c not in fixed_configs]

        if fit_order == "random":
            # randomly shuffle the configs
            rng = np.random.default_rng(seed=seed)
            configs = [str(x) for x in rng.permuted(configs)]

        if fixed_configs is not None:
            configs = fixed_configs + configs
        if n_configs is not None:
            configs = configs[:n_configs]
        df_results_family_hpo, _ = self.repo.evaluate_ensembles(
            configs=configs,
            ensemble_size=n_iterations,
            fit_order="original",
            seed=0,
            time_limit=time_limit,
            backend=self.backend,
            **kwargs,
        )
        df_results_family_hpo = df_results_family_hpo.reset_index()
        df_results_family_hpo["method_type"] = "hpo"

        method_subtype = "tuned" if n_iterations == 1 else "tuned_ensemble"
        df_results_family_hpo["method_subtype"] = method_subtype
        df_results_family_hpo["config_type"] = str(config_type)

        method_metadata = dict(
            n_iterations=n_iterations,
            n_configs=n_configs,
            time_limit=time_limit,
            config_type=config_type,
            fit_order=fit_order,
        )

        df_results_family_hpo["method_metadata"] = [method_metadata] * len(df_results_family_hpo)

        return df_results_family_hpo

    def evaluate_ensembles(
        self,
        configs: list[str] | None = None,
        time_limit: float | None = None,
        n_iterations: int = 40,
        fit_order: Literal["original", "random"] = "original",
        seed: int = 0,
        backend_group_folds: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        if configs is None:
            configs = self.repo.configs()
        df_results, _ = self.repo.evaluate_ensembles(
            configs=configs,
            fit_order=fit_order,
            ensemble_size=n_iterations,
            seed=seed,
            time_limit=time_limit,
            backend=self.backend,
            backend_group_folds=backend_group_folds,
            **kwargs,
        )
        df_results = df_results.reset_index()
        df_results["method_type"] = "portfolio"

        method_subtype = "tuned" if n_iterations == 1 else "tuned_ensemble"
        df_results["method_subtype"] = method_subtype

        method_metadata = dict(
            n_iterations=n_iterations,
            time_limit=time_limit,
            fit_order=fit_order,
        )

        df_results["method_metadata"] = [method_metadata] * len(df_results)

        return df_results

    def evaluate_ensembles_per(
        self,
        df_info: pd.DataFrame,
        time_limit: float | None = None,
        n_iterations: int = 40,
        fit_order: Literal["original", "random"] = "original",
        seed: int = 0,
        **kwargs,
    ) -> pd.DataFrame:
        df_results, _ = self.repo.evaluate_ensembles_per(
            df_info=df_info,
            fit_order=fit_order,
            ensemble_size=n_iterations,
            seed=seed,
            time_limit=time_limit,
            backend=self.backend,
            **kwargs,
        )
        df_results = df_results.reset_index()
        df_results["method_type"] = "portfolio"

        method_subtype = "tuned" if n_iterations == 1 else "tuned_ensemble"
        df_results["method_subtype"] = method_subtype

        method_metadata = dict(
            n_iterations=n_iterations,
            time_limit=time_limit,
            fit_order=fit_order,
        )

        df_results["method_metadata"] = [method_metadata] * len(df_results)

        return df_results

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
        ensemble_cls: type[EnsembleScorer] = EnsembleScorerMaxModels,
        ensemble_kwargs: dict | None = None,
        patience_callback: list | None = None,
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
            patience_callback=patience_callback,
        )

        df_zeroshot_portfolio = pd.DataFrame(a)

        if rename_columns:
            df_zeroshot_portfolio = df_zeroshot_portfolio.rename(
                columns={
                    "metadata": "method_metadata",
                }
            )
            datasets_info = repo.datasets_info()

            df_zeroshot_portfolio["problem_type"] = df_zeroshot_portfolio["dataset"].map(datasets_info["problem_type"])
            df_zeroshot_portfolio["metric"] = df_zeroshot_portfolio["dataset"].map(datasets_info["metric"])

        return df_zeroshot_portfolio

    def run_zs(
        self,
        n_portfolios: int = 200,
        n_ensemble: int | None = None,
        n_ensemble_in_name: bool = True,
        n_max_models_per_type: int | str | None = None,
        time_limit: float | None = 14400,
        **kwargs,
    ) -> pd.DataFrame:
        df_zeroshot_portfolio = self.zeroshot_portfolio(
            n_portfolios=n_portfolios,
            n_ensemble=n_ensemble,
            n_ensemble_in_name=n_ensemble_in_name,
            n_max_models_per_type=n_max_models_per_type,
            time_limit=time_limit,
            engine=self.engine,
            **kwargs,
        )
        df_zeroshot_portfolio["method_type"] = "portfolio"
        # df_zeroshot_portfolio = self.repo_metrics.assemble_metrics(results_df=df_zeroshot_portfolio, configs=[], baselines=[])
        return df_zeroshot_portfolio

    def run_zs_from_types(self, config_types: list[str], **kwargs):
        configs = self.repo.configs(config_types=config_types)
        return self.run_zs(configs=configs, **kwargs)

    def run_baselines(self) -> pd.DataFrame | None:
        if not self.repo.baselines():
            return None
        df_results_baselines = self.repo_metrics.assemble_metrics(
            configs=[], include_metric_error_val=True
        ).reset_index()
        df_results_baselines["method_type"] = "baseline"
        return df_results_baselines

    def run_config_family(self, config_type: str) -> pd.DataFrame:
        configs = self.repo.configs(config_types=[config_type])
        df_results_configs = self.repo_metrics.assemble_metrics(
            configs=configs, baselines=[], include_metric_error_val=True
        ).reset_index()
        df_results_configs["method_type"] = "config"
        configs_types = self.repo.configs_type()
        df_results_configs["config_type"] = df_results_configs["framework"].map(configs_types)
        return df_results_configs

    # FIXME: Temp
    def run_portfolio_search(
        self,
        result_baselines: pd.DataFrame,
        model_types: list[str] | None = None,
        selected_types: list[str] | None = None,
        n_portfolio: int = 25,
        n_ensemble: int = 40,
        time_limit: float | None = 14400,
        average_seeds: bool = False,
    ) -> pd.DataFrame:
        from bencheval.evaluator import BenchmarkEvaluator

        calibration_framework = "RF (default)"
        elo_bootstrap_rounds = 100
        if model_types is None:
            model_types = self.repo.config_types()

        n_types = len(model_types)

        if selected_types is None:
            selected_types = []
        for _i in range(n_types):
            model_types_avail = [model_type for model_type in model_types if model_type not in selected_types]
            results_dict_cur_round = {}
            for model_type in model_types_avail:
                candidate_selected_types = [*selected_types, model_type]
                print(candidate_selected_types)
                candidate_configs = self.repo.configs(config_types=candidate_selected_types)
                cur_result = self.run_zs(
                    configs=candidate_configs,
                    n_portfolios=n_portfolio,
                    n_ensemble=n_ensemble,
                    n_ensemble_in_name=False,
                    time_limit=time_limit,
                )
                cur_result["method"] = model_type
                results_dict_cur_round[model_type] = cur_result

            combined_data_cur_round = pd.concat(list(results_dict_cur_round.values()), ignore_index=True)
            combined_data = pd.concat([result_baselines, combined_data_cur_round], ignore_index=True)

            arena = BenchmarkEvaluator(
                task_col="dataset",
                groupby_columns=["problem_type", "metric"],
                seed_column="fold",
            )
            leaderboard = arena.leaderboard(
                data=combined_data,
                average_seeds=average_seeds,
                include_elo=True,
                elo_kwargs=dict(
                    calibration_framework=calibration_framework,
                    calibration_elo=1000,
                    BOOTSTRAP_ROUNDS=elo_bootstrap_rounds,
                ),
            ).reset_index(drop=False)
            leaderboard_cur_round = leaderboard[leaderboard["method"].isin(results_dict_cur_round.keys())]
            print(leaderboard[["method", "elo", "improvability"]].to_markdown(index=False))
            best_method_info_cur = leaderboard_cur_round.sort_values(by="elo", ascending=False).iloc[0]
            best_method_cur = best_method_info_cur["method"]
            best_method_cur_elo = best_method_info_cur["elo"]
            print(f"Best: {best_method_cur}\tElo: {best_method_cur_elo:.2f}")
            selected_types.append(best_method_cur)
            print(f"Selected Types: {selected_types}")

    # FIXME: This is a hack
    def _config_default(self, config_type: str, use_first_if_missing=False, return_none_if_missing=False) -> str | None:
        configs = self.repo.configs(config_types=[config_type])
        configs_default = [c for c in configs if "_c1_" in c or c.endswith("_c1")]
        if len(configs_default) == 1:
            return configs_default[0]
        if len(configs_default) == 0:
            configs_default = [c for c in configs if "_r1_" in c or c.endswith("_r1")]
            if len(configs_default) == 0:
                if (len(configs) > 0) and use_first_if_missing:
                    return configs[0]
                if return_none_if_missing:
                    return None
                raise ValueError(
                    f"Could not find any default config for config_type='{config_type}'\n\tconfigs={configs}",
                )
            return configs_default[0]
        # >1
        remaining = [c for c in configs_default if c.endswith("_c1")]
        if len(remaining) == 1:
            return remaining[0]
        if len(remaining) > 1:
            configs_default = remaining
        else:
            len_suffix = [len(c.rsplit("_c1_", maxsplit=1)[-1]) for c in configs_default]
            min_suffix = min(len_suffix)
            configs_default = [c for i, c in enumerate(configs_default) if len_suffix[i] == min_suffix]
        configs_default = sorted(configs_default)
        if len(configs_default) > 1:
            print(
                f"Found {len(configs_default)} potential default configs for config_type='{config_type}', but only one should exist."
                f"\n\tpotential defaults: {configs_default}"
                f"\n\tconfigs={configs}"
                f"\nSelecting {configs_default[0]} as default via string sort.",
            )
        return configs_default[0]

    def run_config(self, config: str) -> pd.DataFrame:
        configs = [config]
        return self.repo_metrics.assemble_metrics(
            configs=configs,
            baselines=[],
            include_metric_error_val=True,
        ).reset_index()

    def run_config_default(self, model_type: str) -> pd.DataFrame:
        config_default = self._config_default(config_type=model_type, use_first_if_missing=True)
        df_results_config = self.run_config(config=config_default)
        configs_types = self.repo.configs_type()
        df_results_config["method_type"] = "config"
        df_results_config["method_subtype"] = "default"
        df_results_config["config_type"] = df_results_config["framework"].map(configs_types)
        df_results_config["framework"] = f"{model_type} (default)"
        return df_results_config

    def run_minimal_single(self, model_type: str, tune: bool = True) -> pd.DataFrame:
        """Run logic that isn't impacted by other methods or other datasets.

        Returns:
        -------

        """
        config_default = self._config_default(config_type=model_type, use_first_if_missing=True)
        if config_default is not None:
            df_results_config_default = self.run_config_default(model_type=model_type)
            df_results_config_default["imputed"] = False
        else:
            df_results_config_default = None

        df_results_hpo = self.run_hpo_by_family(model_types=[model_type]) if tune else None

        to_concat_lst = [
            df_results_config_default,
            df_results_hpo,
        ]
        to_concat_lst = [df for df in to_concat_lst if df is not None]

        return pd.concat(to_concat_lst, ignore_index=True)
