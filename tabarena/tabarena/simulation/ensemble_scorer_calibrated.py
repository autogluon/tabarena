from __future__ import annotations

import copy

import numpy as np

from tabarena.simulation.ensemble_selection_config_scorer import EnsembleScorerMaxModels


# FIXME: WIP
# FIXME: Requires probmetrics and pytorch-minimize
class EnsembleScorerCalibrated(EnsembleScorerMaxModels):
    def __init__(
        self,
        calibrator_type: str = "logistic",
        calibrate_per_model: bool = False,
        calibrate_after_ens: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.calibrator_type = calibrator_type
        self.calibrate_per_model = calibrate_per_model
        self.calibrate_after_ens = calibrate_after_ens

    def get_calibrator(self):
        from probmetrics.calibrators import get_calibrator
        # also: pip install probmetrics pytorch-minimize
        calibrator = get_calibrator(self.calibrator_type)
        return calibrator

    def evaluate_task(self, dataset: str, fold: int, models: list[str]) -> dict[str, object]:
        n_models = len(models)
        task_metadata = self.task_metrics_metadata[dataset]
        metric_name = task_metadata["metric"]
        problem_type = task_metadata["problem_type"]

        if problem_type == "multiclass" and self.calibrator_type is not None:
            use_fast_metrics = False
            calibrator = self.get_calibrator()
            calibrate_after_ens = self.calibrate_after_ens
            calibrate_per_model = self.calibrate_per_model
        else:
            use_fast_metrics = self.use_fast_metrics
            calibrator = None
            calibrate_after_ens = False
            calibrate_per_model = False

        fit_metric_name = self.proxy_fit_metric_map.get(metric_name, metric_name)

        eval_metric = self._get_metric_from_name(metric_name=metric_name, problem_type=problem_type, use_fast_metrics=use_fast_metrics)
        fit_eval_metric = self._get_metric_from_name(metric_name=fit_metric_name, problem_type=problem_type, use_fast_metrics=use_fast_metrics)

        y_val_og = self.repo.labels_val(dataset=dataset, fold=fold)
        y_test = self.repo.labels_test(dataset=dataset, fold=fold)

        # If filtering models, need to keep track of original model order to return ensemble weights list
        models_filtered = self.filter_models(dataset=dataset, fold=fold, models=models)
        models, models_filtered_idx = self._get_models_filtered_idx(models=models, models_filtered=models_filtered)

        pred_val_og, pred_test = self.get_preds_from_models(dataset=dataset, fold=fold, models=models)

        if calibrate_per_model:
            for i, m in enumerate(models):
                y_val_pred_model = pred_val_og[i, :, :]
                y_test_pred_model = pred_test[i, :, :]

                calibrator_model = self.get_calibrator()

                if self.optimize_on == "val":
                    calibrator_model.fit(y_val_pred_model, y_val_og)
                elif self.optimize_on == "test":
                    calibrator_model.fit(y_test_pred_model, y_test)
                else:
                    raise ValueError(f"Invalid value for `optimize_on`: {self.optimize_on}")

                pred_val_og[i, :, :] = calibrator_model.predict_proba(y_val_pred_model)
                pred_test[i, :, :] = calibrator_model.predict_proba(y_test_pred_model)

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

        metric_error_val = None
        if self.return_metric_error_val or calibrate_after_ens:
            if hasattr(eval_metric, 'preprocess_bulk'):
                y_val_og, pred_val_og = eval_metric.preprocess_bulk(y_val_og, pred_val_og)
            if eval_metric.needs_pred:
                y_val_pred = weighted_ensemble.predict(pred_val_og)
            else:
                y_val_pred = weighted_ensemble.predict_proba(pred_val_og)
            if calibrate_after_ens:
                if self.optimize_on == "val":
                    calibrator.fit(y_val_pred, y_val_og)
                elif self.optimize_on == "test":
                    calibrator.fit(y_test_pred, y_test)
                else:
                    raise ValueError(f"Invalid value for `optimize_on`: {self.optimize_on}")
                
                y_val_pred = calibrator.predict_proba(y_val_pred)
                y_test_pred = calibrator.predict_proba(y_test_pred)
            metric_error_val = eval_metric.error(y_val_og, y_val_pred)

        err = eval_metric.error(y_test, y_test_pred)

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
