from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from tabarena.nips2025_utils.artifacts import tabarena_method_metadata_collection
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena import EvaluationRepository
from tabarena.simulation.ensemble_selection_config_scorer import EnsembleScorer


# TODO: `evaluate_task` is where the main logic lives. This is copy pasted from `EnsembleScorer`, feel free to edit to add calibration
# TODO: If using backend="ray" this might crash due to non-serializable. For testing with ray, may need to move this to a separate file.
# TODO: Note for speed we collapse multiclass pred proba into binary pred proba for calculating log loss.
#  You can disable this if you want with `ensemble_kwargs={"use_fast_metrics": False}`
class EnsembleScorerCalibrated(EnsembleScorer):
    def evaluate_task(self, dataset: str, fold: int, models: list[str]) -> dict[str, object]:
        n_models = len(models)
        task_metadata = self.task_metrics_metadata[dataset]
        metric_name = task_metadata["metric"]
        problem_type = task_metadata["problem_type"]

        y_val_og = self.repo.labels_val(dataset=dataset, fold=fold)
        y_test = self.repo.labels_test(dataset=dataset, fold=fold)

        # If filtering models, need to keep track of original model order to return ensemble weights list
        models_filtered = self.filter_models(dataset=dataset, fold=fold, models=models)
        models, models_filtered_idx = self._get_models_filtered_idx(models=models, models_filtered=models_filtered)

        pred_val_og, pred_test = self.get_preds_from_models(dataset=dataset, fold=fold, models=models)

        if self.optimize_on == "val":
            # Use the original validation data for a fair comparison that mirrors what happens in practice
            y_val = y_val_og
            pred_val = pred_val_og
        elif self.optimize_on == "test":
            # Optimize directly on test (unrealistic, but can be used to measure the gap in generalization)
            # TODO: Another variant that could be implemented, do 50% of test as val and the rest as test
            #  to simulate impact of using holdout validation
            y_val = copy.deepcopy(y_test)
            pred_val = copy.deepcopy(pred_test)
        else:
            raise ValueError(f"Invalid value for `optimize_on`: {self.optimize_on}")

        if problem_type == 'binary':
            # Force binary prediction probabilities to 1 dimensional prediction probabilites of the positive class
            # if it is in multiclass format
            if len(pred_val.shape) == 3:
                pred_val = pred_val[:, :, 1]
            if len(pred_test.shape) == 3:
                pred_test = pred_test[:, :, 1]

        fit_metric_name = self.proxy_fit_metric_map.get(metric_name, metric_name)

        eval_metric = self._get_metric_from_name(metric_name=metric_name, problem_type=problem_type)
        fit_eval_metric = self._get_metric_from_name(metric_name=fit_metric_name, problem_type=problem_type)

        if hasattr(fit_eval_metric, 'preprocess_bulk'):
            y_val, pred_val = fit_eval_metric.preprocess_bulk(y_val, pred_val)

        if hasattr(fit_eval_metric, 'post_problem_type'):
            fit_problem_type = fit_eval_metric.post_problem_type
        else:
            fit_problem_type = problem_type

        weighted_ensemble = self.ensemble_method(
            problem_type=fit_problem_type,
            metric=fit_eval_metric,
            **self.ensemble_method_kwargs,
        )

        weighted_ensemble.fit(predictions=pred_val, labels=y_val)

        if hasattr(eval_metric, 'preprocess_bulk'):
            y_test, pred_test = eval_metric.preprocess_bulk(y_test, pred_test)

        if hasattr(eval_metric, 'post_problem_type'):
            predict_problem_type = eval_metric.post_problem_type
        else:
            predict_problem_type = problem_type
        weighted_ensemble.problem_type = predict_problem_type

        if eval_metric.needs_pred:
            y_test_pred = weighted_ensemble.predict(pred_test)
        else:
            y_test_pred = weighted_ensemble.predict_proba(pred_test)
        err = eval_metric.error(y_test, y_test_pred)

        metric_error_val = None
        if self.return_metric_error_val:
            if hasattr(eval_metric, 'preprocess_bulk'):
                y_val_og, pred_val_og = eval_metric.preprocess_bulk(y_val_og, pred_val_og)
            if eval_metric.needs_pred:
                y_val_pred = weighted_ensemble.predict(pred_val_og)
            else:
                y_val_pred = weighted_ensemble.predict_proba(pred_val_og)
            metric_error_val = eval_metric.error(y_val_og, y_val_pred)

        ensemble_weights: np.array = weighted_ensemble.weights_

        # ensemble_weights has to be updated, need to be in the original models order
        ensemble_weights_fixed = np.zeros(n_models, dtype=np.float64)
        ensemble_weights_fixed[models_filtered_idx] = ensemble_weights
        ensemble_weights = ensemble_weights_fixed

        results = dict(
            metric_error=err,
            ensemble_weights=ensemble_weights,
        )
        if self.return_metric_error_val:
            results["metric_error_val"] = metric_error_val

        return results


if __name__ == "__main__":
    method_type = "LightGBM"
    # print(tabarena_method_metadata_collection.info())  # available types
    method_metadata = tabarena_method_metadata_collection.get_method_metadata(method_type)

    if not method_metadata.path_processed_exists:
        # download the processed data if needed. Will take some time (~15 GB for methods with 201 configs)
        print(f"Downloading processed data for {method_metadata.method}...")
        method_metadata.method_downloader(verbose=True).download_processed()

    run_toy = True  # False to run a full run
    repo_cache_path = "repo_tmp_cal.pkl"

    if run_toy:
        if not (Path(repo_cache_path).exists() and Path(repo_cache_path).is_file()):
            repo = method_metadata.load_processed()
            repo = repo.subset(folds=[0], configs=repo.configs()[:5], problem_types=["multiclass"])
            repo.save(path=repo_cache_path)
        # much faster for debugging
        repo = EvaluationRepository.load(path=repo_cache_path)
        shared_kwargs = dict(
            repo=repo,
            n_iterations=2,
            fit_order="original",
            backend="native",
        )
    else:
        repo = method_metadata.load_processed()  # the full data
        shared_kwargs = dict(
            repo=repo,
            fit_order="original",
        )


    df_results_hpo_og = method_metadata.generate_hpo_result(
        ensemble_cls=EnsembleScorer,  # Normal logic
        **shared_kwargs,
    )
    df_results_hpo_og["method"] = df_results_hpo_og["ta_name"] + "-ORIGINAL"
    df_results_hpo_og["method_type"] = "baseline"  # Do this so it appears on the leaderboard
    method_name_og = df_results_hpo_og.iloc[0]["method"]

    df_results_hpo_cal = method_metadata.generate_hpo_result(
        ensemble_cls=EnsembleScorerCalibrated,  # Your logic
        # ensemble_kwargs={},  # Feel free to add any init kwargs for the ensemble cls here
        **shared_kwargs,
    )
    df_results_hpo_cal["method"] = df_results_hpo_cal["ta_name"] + "-CALIBRATED"
    df_results_hpo_cal["method_type"] = "baseline"  # Do this so it appears on the leaderboard
    method_name_cal = df_results_hpo_cal.iloc[0]["method"]

    new_results = pd.concat([df_results_hpo_og, df_results_hpo_cal], ignore_index=True)

    ta_context = TabArenaContext()
    leaderboard = ta_context.compare(
        output_dir="output_test_calibration",
        new_results=new_results,
        only_valid_tasks=True,
    )

    leaderboard_new_results = leaderboard[leaderboard["method"].isin([method_name_og, method_name_cal])]

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)
        print(leaderboard_new_results)
