from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from tabarena.benchmark.result import AGBagResult, BaselineResult, ConfigResult
from tabarena.repository import EvaluationRepository


# TODO: Inspect artifact folder to load all results without needing to specify them explicitly
#  generate_repo_from_dir(expname)
class ExperimentResults:
    def __init__(
        self,
        task_metadata: pd.DataFrame,
    ):
        self.task_metadata = task_metadata

    def repo_from_results(
        self,
        results_lst: list[dict[str, Any] | BaselineResult],
        calibrate: bool = False,
        include_holdout: bool = False,
    ) -> EvaluationRepository:
        results_lst: list[BaselineResult] = [BaselineResult.from_dict(result) for result in results_lst]

        results_configs: list[ConfigResult] = []
        results_baselines: list[BaselineResult] = []
        for result in results_lst:
            if isinstance(result, ConfigResult):
                results_configs.append(result)
            else:
                results_baselines.append(result)

        n_configs = len(results_configs)
        if calibrate:
            results_configs_calibrated = []
            for i, result in enumerate(results_configs):
                if i % 100 == 0:
                    print(f"Calibrating: {i+1}/{n_configs}\t{result.framework}")
                results_configs_calibrated.append(self._calibrate(result=result))
            results_configs += results_configs_calibrated

        n_configs = len(results_configs)
        if include_holdout:
            for r_i, result in enumerate(results_configs):
                if isinstance(result, AGBagResult):
                    if r_i % 100 == 0:
                        print(f"Generating Holdout Results: {r_i + 1}/{n_configs}\t{result.framework}")
                    results_new: list[BaselineResult] = result.bag_artifacts()
                    results_baselines += results_new

        results_lst_df = [result.compute_df_result() for result in results_configs]
        results_lst_df_baselines = [result.compute_df_result() for result in results_baselines]
        df_configs = pd.concat(results_lst_df, ignore_index=True) if results_lst_df else None
        df_baselines = pd.concat(results_lst_df_baselines, ignore_index=True) if results_lst_df_baselines else None

        configs_hyperparameters = self._get_configs_hyperparameters(results_configs=results_configs)
        results_lst_simulation_artifacts = [result.generate_old_sim_artifact() for result in results_configs]

        # TODO: per-fold pred_proba_test and pred_proba_val (indices?)
        repo: EvaluationRepository = EvaluationRepository.from_raw(
            df_configs=df_configs,
            df_baselines=df_baselines,
            results_lst_simulation_artifacts=results_lst_simulation_artifacts,
            task_metadata=self.task_metadata,
            configs_hyperparameters=configs_hyperparameters,
        )

        return repo

    def _calibrate(self, result: ConfigResult) -> ConfigResult:
        problem_type = result.result["problem_type"]
        if problem_type == "multiclass":
            # FIXME: What about binary?
            result_calibrated = result.generate_calibrated(method="v2", name_suffix="_CAL")
        else:
            result_calibrated = copy.deepcopy(result)
            result_calibrated.result["framework"] = result_calibrated.result["framework"] + "_CAL"
        return result_calibrated

    def _get_configs_hyperparameters(self, results_configs: list[ConfigResult]) -> dict | None:
        configs_hyperparameters = {}
        for result in results_configs:
            if result.framework in configs_hyperparameters:
                continue
            configs_hyperparameters[result.framework] = result.hyperparameters
        if not configs_hyperparameters:
            configs_hyperparameters = None
        return configs_hyperparameters
