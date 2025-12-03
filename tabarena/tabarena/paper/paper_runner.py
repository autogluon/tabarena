from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from tabarena.evaluation.evaluator import Evaluator
from tabarena.repository import EvaluationRepository


class PaperRun:
    def __init__(self, repo: EvaluationRepository, output_dir: str = None, backend: Literal["ray", "native"] = "ray"):
        self.repo = repo
        self.evaluator = Evaluator(repo=self.repo)
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

    def run_hpo_by_family(self, include_uncapped: bool = False, include_4h: bool = True, model_types: list[str] | None = None) -> pd.DataFrame:
        config_type_groups = self.get_config_type_groups()

        hpo_results_lst = []

        # for model_key, model_name in [("REALMLP", "RealMLP"), ("XGB", "XGBoost"), ("CAT", "CatBoost"), ("GBM", "LightGBM"), ("RF", "RandomForest"), ("XT", "ExtraTrees")]:
        #     realmlp_og = [c for c in config_type_groups[model_key] if c == f"{model_name}_c1_BAG_L1" or "_alt_" not in c]
        #     realmlp_alt = [c for c in config_type_groups[model_key] if c == f"{model_name}_c1_BAG_L1" or "_alt_" in c]
        #     config_type_groups[f"{model_key}_OG"] = realmlp_og
        #     config_type_groups[f"{model_key}_ALT"] = realmlp_alt

        if model_types is None:
            model_types = list(config_type_groups.keys())
        for family in model_types:
            assert family in config_type_groups, f"Model family {family} missing from available families: {list(config_type_groups.keys())}"

        if include_4h:
            time_limit = 3600 * 4
            # FIXME: do multiple random seeds and average
            for family in model_types:
                df_results_family_hpo_ens = self.run_ensemble_config_type(
                    config_type=family, fit_order="random", seed=0, n_iterations=40, time_limit=time_limit,
                )
                df_results_family_hpo_ens["framework"] = f"{family} (tuned + ensemble) (4h)"
                df_results_family_hpo = self.run_ensemble_config_type(
                    config_type=family, fit_order="random", seed=0, n_iterations=1, time_limit=time_limit,
                )
                df_results_family_hpo["framework"] = f"{family} (tuned) (4h)"
                hpo_results_lst.append(df_results_family_hpo)
                hpo_results_lst.append(df_results_family_hpo_ens)

        if include_uncapped:
            for family in model_types:
                df_results_family_hpo_ens = self.run_ensemble_config_type(
                    config_type=family, fit_order="original", seed=0, n_iterations=40, time_limit=None,
                )
                df_results_family_hpo_ens["framework"] = f"{family} (tuned + ensemble)"

                df_results_family_hpo = self.run_ensemble_config_type(
                    config_type=family, fit_order="original", seed=0, n_iterations=1, time_limit=None,
                )
                df_results_family_hpo["framework"] = f"{family} (tuned)"
                hpo_results_lst.append(df_results_family_hpo)
                hpo_results_lst.append(df_results_family_hpo_ens)

        df_results_hpo_all = pd.concat(hpo_results_lst, ignore_index=True)
        return df_results_hpo_all

    def run_ensemble_config_type(
        self,
        config_type: str,
        n_iterations: int,
        n_configs: int = None,
        fixed_configs: list[str] | None = None,
        time_limit: float | None = None,
        fit_order: Literal["original", "random"] = "original",
        seed: int = 0,
        **kwargs,
    ) -> pd.DataFrame:
        # FIXME: Don't recompute this each call, implement `self.repo.configs(config_types=[config_type])`
        config_type_groups = self.get_config_type_groups()
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

        if n_iterations == 1:
            method_subtype = "tuned"
        else:
            method_subtype = "tuned_ensemble"
        df_results_family_hpo["method_subtype"] = method_subtype
        df_results_family_hpo["config_type"] = config_type

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
        )
        df_results = df_results.reset_index()
        df_results["method_type"] = "portfolio"

        if n_iterations == 1:
            method_subtype = "tuned"
        else:
            method_subtype = "tuned_ensemble"
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

        if n_iterations == 1:
            method_subtype = "tuned"
        else:
            method_subtype = "tuned_ensemble"
        df_results["method_subtype"] = method_subtype

        method_metadata = dict(
            n_iterations=n_iterations,
            time_limit=time_limit,
            fit_order=fit_order,
        )

        df_results["method_metadata"] = [method_metadata] * len(df_results)

        return df_results

    def run_zs(
        self,
        n_portfolios: int = 200,
        n_ensemble: int = None,
        n_ensemble_in_name: bool = True,
        n_max_models_per_type: int | str | None = None,
        time_limit: float | None = 14400,
        **kwargs,
    ) -> pd.DataFrame:
        df_zeroshot_portfolio = self.evaluator.zeroshot_portfolio(
            n_portfolios=n_portfolios,
            n_ensemble=n_ensemble,
            n_ensemble_in_name=n_ensemble_in_name,
            n_max_models_per_type=n_max_models_per_type,
            time_limit=time_limit,
            engine=self.engine,
            **kwargs,
        )
        df_zeroshot_portfolio["method_type"] = "portfolio"
        # df_zeroshot_portfolio = self.evaluator.compare_metrics(results_df=df_zeroshot_portfolio, configs=[], baselines=[])
        return df_zeroshot_portfolio

    def run_zs_from_types(self, config_types: list[str], **kwargs):
        configs = self.evaluator.repo.configs(config_types=config_types)
        return self.run_zs(configs=configs, **kwargs)

    def run_baselines(self) -> pd.DataFrame | None:
        if not self.repo.baselines():
            return None
        df_results_baselines = self.evaluator.compare_metrics(configs=[], include_metric_error_val=True).reset_index()
        df_results_baselines["method_type"] = "baseline"
        return df_results_baselines

    def run_config_family(self, config_type: str) -> pd.DataFrame:
        configs = self.repo.configs(config_types=[config_type])
        df_results_configs = self.evaluator.compare_metrics(configs=configs, baselines=[], include_metric_error_val=True).reset_index()
        df_results_configs["method_type"] = "config"
        configs_types = self.repo.configs_type()
        df_results_configs["config_type"] = df_results_configs["framework"].map(configs_types)
        return df_results_configs
