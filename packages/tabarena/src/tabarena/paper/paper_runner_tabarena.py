from __future__ import annotations

import pandas as pd

from bencheval.evaluator import BenchmarkEvaluator

from .paper_runner import PaperRun


class PaperRunTabArena(PaperRun):
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
        return self.evaluator.compare_metrics(
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
