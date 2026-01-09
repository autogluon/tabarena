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

        if problem_type in ["binary", "multiclass"] and self.calibrator_type is not None:
            if problem_type == "binary":
                use_fast_metrics = self.use_fast_metrics
            else:
                use_fast_metrics = False
            calibrator = self.get_calibrator()
            calibrate_after_ens = self.calibrate_after_ens
            calibrate_per_model = self.calibrate_per_model
        else:
            use_fast_metrics = self.use_fast_metrics
            calibrator = None
            calibrate_after_ens = False
            calibrate_per_model = False

        eval_metric, fit_eval_metric, predict_problem_type, fit_problem_type = self._get_metrics(
            metric_name=metric_name,
            problem_type=problem_type,
            use_fast_metrics=use_fast_metrics,
        )

        y_val_og = self.repo.labels_val(dataset=dataset, fold=fold)
        y_test = self.repo.labels_test(dataset=dataset, fold=fold)

        # If filtering models, need to keep track of original model order to return ensemble weights list
        models_filtered = self.filter_models(dataset=dataset, fold=fold, models=models)
        models, models_filtered_idx = self._get_models_filtered_idx(models=models, models_filtered=models_filtered)

        pred_val_og, pred_test = self.get_preds_from_models(dataset=dataset, fold=fold, models=models)

        if calibrate_per_model:
            for i, m in enumerate(models):
                if problem_type == "multiclass":
                    y_val_pred_model = pred_val_og[i, :, :]
                    y_test_pred_model = pred_test[i, :, :]
                else:
                    y_val_pred_model = pred_val_og[i, :]
                    y_test_pred_model = pred_test[i, :]

                calibrator_model = self.get_calibrator()

                if self.optimize_on == "val":
                    calibrator_model.fit(y_val_pred_model, y_val_og)
                elif self.optimize_on == "test":
                    calibrator_model.fit(y_test_pred_model, y_test)
                else:
                    raise ValueError(f"Invalid value for `optimize_on`: {self.optimize_on}")

                pred_val_cal = calibrator_model.predict_proba(y_val_pred_model)
                pred_test_cal = calibrator_model.predict_proba(y_test_pred_model)

                if problem_type == "multiclass":
                    pred_val_og[i, :, :] = pred_val_cal
                    pred_test[i, :, :] = pred_test_cal
                else:
                    if problem_type == "binary":
                        pred_val_cal = pred_val_cal[:, 1]
                        pred_test_cal = pred_test_cal[:, 1]
                    pred_val_og[i, :] = pred_val_cal
                    pred_test[i, :] = pred_test_cal

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

        weighted_ensemble = self.fit_ensemble(
            pred=pred_val,
            y=y_val,
            fit_eval_metric=fit_eval_metric,
            fit_problem_type=fit_problem_type,
            predict_problem_type=predict_problem_type,
        )

        if hasattr(eval_metric, 'preprocess_bulk'):
            y_test, pred_test = eval_metric.preprocess_bulk(y_test, pred_test)

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

                if problem_type == "binary":
                    y_val_pred = y_val_pred[:, 1]
                    y_test_pred = y_test_pred[:, 1]

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


class EnsembleScorerCalibratedCV(EnsembleScorerMaxModels):
    def __init__(
        self,
        calibrator_type: str = "logistic",
        calibrate_per_model: bool = False,
        calibrate_after_ens: bool = True,
        calibrator_n_splits: int = 10,
        calibrator_shuffle: bool = True,
        calibrator_random_state: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.calibrator_type = calibrator_type
        self.calibrate_per_model = calibrate_per_model
        self.calibrate_after_ens = calibrate_after_ens

        # Cross-validation settings for calibration OOF preds
        self.calibrator_n_splits = calibrator_n_splits
        self.calibrator_shuffle = calibrator_shuffle
        self.calibrator_random_state = calibrator_random_state

    def get_calibrator(self):
        from probmetrics.calibrators import get_calibrator

        calibrator = get_calibrator(self.calibrator_type)
        return calibrator

    def _get_cv_splitter(
        self,
        n_splits: int,
        problem_type: str,
        random_state: int,
    ):
        stratify = problem_type in ("binary", "multiclass")
        from autogluon.common.utils.cv_splitter import CVSplitter
        return CVSplitter(
            n_splits=n_splits,
            n_repeats=1,
            stratify=stratify,
            random_state=random_state,
        )

    def _calibrate_with_cv_for_val_and_full_for_test(
        self,
        *,
        calibrator_factory,
        proba_val: np.ndarray,
        y_val: np.ndarray,
        proba_test: np.ndarray,
        problem_type: str,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          - calibrated_val_oof: out-of-fold calibrated probabilities for the "validation-side" data
          - calibrated_test: calibrated probabilities for test, using calibrator fit on full validation-side data
        """
        n_splits = int(self.calibrator_n_splits) if self.calibrator_n_splits is not None else 0

        # Default: if CV not possible, fall back to in-sample calibration (previous behavior)
        def _fit_full_and_predict(val_proba, val_y, test_proba):
            cal = calibrator_factory()
            cal.fit(val_proba, val_y)
            return cal.predict_proba(val_proba), cal.predict_proba(test_proba)

        # Need at least 2 splits, and enough samples per class for stratification in classification
        n_samples = int(proba_val.shape[0])
        if n_splits < 2 or n_samples < 2:
            return _fit_full_and_predict(proba_val, y_val, proba_test)

        # if problem_type in ("binary", "multiclass"):
        #     # If some class has < n_splits samples, StratifiedKFold will error; fall back.
        #     _, counts = np.unique(y_val, return_counts=True)
        #     if counts.min() < n_splits:
        #         return _fit_full_and_predict(proba_val, y_val, proba_test)

        if problem_type in ("binary", "multiclass"):
            # If some class has < 2 samples, StratifiedKFold will error; fall back.
            _, counts = np.unique(y_val, return_counts=True)
            if counts.min() < 2:
                return _fit_full_and_predict(proba_val, y_val, proba_test)

        splitter = self._get_cv_splitter(
            n_splits=n_splits,
            problem_type=problem_type,
            random_state=random_state,
        )

        # Choose splitter
        # if problem_type in ("binary", "multiclass"):
        #     from sklearn.model_selection import StratifiedKFold
        #
        #     # # If some class has < n_splits samples, StratifiedKFold will error; fall back.
        #     # _, counts = np.unique(y_val, return_counts=True)
        #     # if counts.min() < n_splits:
        #     #     return _fit_full_and_predict(proba_val, y_val, proba_test)
        #
        #     # splitter = StratifiedKFold(
        #     #     n_splits=n_splits,
        #     #     shuffle=self.calibrator_shuffle,
        #     #     random_state=self.calibrator_random_state,
        #     # )
        #     # split_iter = splitter.split(np.zeros(n_samples, dtype=int), y_val)
        #
        # else:
        #     # Calibration currently only used for multiclass in your codepath,
        #     # but keep this generic.
        #     from sklearn.model_selection import KFold
        #
        #     # splitter = KFold(
        #     #     n_splits=n_splits,
        #     #     shuffle=self.calibrator_shuffle,
        #     #     random_state=self.calibrator_random_state,
        #     # )
        #     # split_iter = splitter.split(np.zeros(n_samples, dtype=int))
        split_iter = splitter.split(None, y_val)

        # OOF calibrated predictions on "val-side"
        calibrated_val_oof = np.empty_like(proba_val)
        for train_idx, holdout_idx in split_iter:
            cal = calibrator_factory()
            cal.fit(proba_val[train_idx], y_val[train_idx])
            calibrated_val_oof_split = cal.predict_proba(proba_val[holdout_idx])
            if problem_type == "binary":
                calibrated_val_oof_split = calibrated_val_oof_split[:, 1]
            calibrated_val_oof[holdout_idx] = calibrated_val_oof_split

        # Test-side calibration remains: fit on full "val-side", predict test
        cal_full = calibrator_factory()
        cal_full.fit(proba_val, y_val)
        calibrated_test = cal_full.predict_proba(proba_test)
        if problem_type == "binary":
            calibrated_test = calibrated_test[:, 1]

        return calibrated_val_oof, calibrated_test

    def evaluate_task(self, dataset: str, fold: int, models: list[str]) -> dict[str, object]:
        n_models = len(models)
        task_metadata = self.task_metrics_metadata[dataset]
        metric_name = task_metadata["metric"]
        problem_type = task_metadata["problem_type"]

        if problem_type in ["binary", "multiclass"] and self.calibrator_type is not None:
            if problem_type == "binary":
                use_fast_metrics = self.use_fast_metrics
            else:
                use_fast_metrics = False
            calibrator = self.get_calibrator()
            calibrate_after_ens = self.calibrate_after_ens
            calibrate_per_model = self.calibrate_per_model
        else:
            use_fast_metrics = self.use_fast_metrics
            calibrator = None
            calibrate_after_ens = False
            calibrate_per_model = False

        eval_metric, fit_eval_metric, predict_problem_type, fit_problem_type = self._get_metrics(
            metric_name=metric_name,
            problem_type=problem_type,
            use_fast_metrics=use_fast_metrics,
        )

        y_val_og = self.repo.labels_val(dataset=dataset, fold=fold)
        y_test = self.repo.labels_test(dataset=dataset, fold=fold)

        models_filtered = self.filter_models(dataset=dataset, fold=fold, models=models)
        models, models_filtered_idx = self._get_models_filtered_idx(models=models, models_filtered=models_filtered)

        pred_val_og, pred_test = self.get_preds_from_models(dataset=dataset, fold=fold, models=models)

        # When optimize_on="test" and calibration uses CV, we need a separate
        # "val-side" prediction array that is OOF on test, while keeping pred_test
        # as the (possibly in-sample) test predictions (unchanged logic).
        pred_test_oof_for_opt = None

        if calibrate_per_model:
            for i, _m in enumerate(models):
                if problem_type == "multiclass":
                    y_val_pred_model = pred_val_og[i, :, :]
                    y_test_pred_model = pred_test[i, :, :]
                else:
                    y_val_pred_model = pred_val_og[i, :]
                    y_test_pred_model = pred_test[i, :]

                if self.optimize_on == "val":
                    # OOF-calibrate val predictions; fit-full-on-val for test preds
                    val_oof, test_cal = self._calibrate_with_cv_for_val_and_full_for_test(
                        calibrator_factory=self.get_calibrator,
                        proba_val=y_val_pred_model,
                        y_val=y_val_og,
                        proba_test=y_test_pred_model,
                        problem_type=problem_type,
                        random_state=self.calibrator_random_state,
                    )

                    if problem_type == "multiclass":
                        pred_val_og[i, :, :] = val_oof
                        pred_test[i, :, :] = test_cal
                    else:
                        pred_val_og[i, :] = val_oof
                        pred_test[i, :] = test_cal

                elif self.optimize_on == "test":
                    # "Validation-side" data is the test set here; make OOF version for optimization,
                    # but keep pred_test calibrated by fitting on full test and predicting test (same as before).
                    test_oof, test_cal = self._calibrate_with_cv_for_val_and_full_for_test(
                        calibrator_factory=self.get_calibrator,
                        proba_val=y_test_pred_model,
                        y_val=y_test,
                        proba_test=y_test_pred_model,  # full-fit predict on same set -> same as before
                        problem_type=problem_type,
                        random_state=self.calibrator_random_state,
                    )

                    if pred_test_oof_for_opt is None:
                        pred_test_oof_for_opt = copy.deepcopy(pred_test)

                    if problem_type == "multiclass":
                        pred_test[i, :, :] = test_cal
                        pred_test_oof_for_opt[i, :, :] = test_oof
                    else:
                        pred_test[i, :] = test_cal
                        pred_test_oof_for_opt[i, :] = test_oof
                else:
                    raise ValueError(f"Invalid value for `optimize_on`: {self.optimize_on}")

        if self.optimize_on == "val":
            y_val = y_val_og
            pred_val = pred_val_og
        elif self.optimize_on == "test":
            y_val = copy.deepcopy(y_test)
            # Use OOF-calibrated version for optimization if available; else fall back to pred_test
            if pred_test_oof_for_opt is not None:
                pred_val = pred_test_oof_for_opt
            else:
                pred_val = copy.deepcopy(pred_test)
        else:
            raise ValueError(f"Invalid value for `optimize_on`: {self.optimize_on}")

        if problem_type == "binary":
            if len(pred_val.shape) == 3:
                pred_val = pred_val[:, :, 1]
            if len(pred_test.shape) == 3:
                pred_test = pred_test[:, :, 1]

        weighted_ensemble = self.fit_ensemble(
            pred=pred_val,
            y=y_val,
            fit_eval_metric=fit_eval_metric,
            fit_problem_type=fit_problem_type,
            predict_problem_type=predict_problem_type,
        )

        if hasattr(eval_metric, "preprocess_bulk"):
            y_test, pred_test = eval_metric.preprocess_bulk(y_test, pred_test)

        if eval_metric.needs_pred:
            y_test_pred = weighted_ensemble.predict(pred_test)
        else:
            y_test_pred = weighted_ensemble.predict_proba(pred_test)

        metric_error_val = None
        if self.return_metric_error_val or calibrate_after_ens:
            if hasattr(eval_metric, "preprocess_bulk"):
                y_val_og_proc, pred_val_og_proc = eval_metric.preprocess_bulk(y_val_og, pred_val_og)
            else:
                y_val_og_proc, pred_val_og_proc = y_val_og, pred_val_og

            if eval_metric.needs_pred:
                y_val_pred = weighted_ensemble.predict(pred_val_og_proc)
            else:
                y_val_pred = weighted_ensemble.predict_proba(pred_val_og_proc)

            if calibrate_after_ens:
                if self.optimize_on == "val":
                    # OOF-calibrate validation-side ensemble preds; fit-full-on-val for test preds
                    y_val_pred_oof, y_test_pred_cal = self._calibrate_with_cv_for_val_and_full_for_test(
                        calibrator_factory=lambda: calibrator,
                        proba_val=y_val_pred,
                        y_val=y_val_og_proc,
                        proba_test=y_test_pred,
                        problem_type=problem_type,
                        random_state=self.calibrator_random_state+1,
                    )
                    y_val_pred = y_val_pred_oof
                    y_test_pred = y_test_pred_cal

                elif self.optimize_on == "test":
                    # Validation-side is test set: OOF for y_val_pred, but keep test prediction logic the same
                    y_val_pred_oof, y_test_pred_cal = self._calibrate_with_cv_for_val_and_full_for_test(
                        calibrator_factory=lambda: calibrator,
                        proba_val=y_test_pred,
                        y_val=y_test,
                        proba_test=y_test_pred,  # fit on full test, predict test (same behavior)
                        problem_type=problem_type,
                        random_state=self.calibrator_random_state + 1,
                    )
                    y_val_pred = y_val_pred_oof
                    y_test_pred = y_test_pred_cal
                else:
                    raise ValueError(f"Invalid value for `optimize_on`: {self.optimize_on}")

            metric_error_val = eval_metric.error(y_val_og_proc, y_val_pred)

        err = eval_metric.error(y_test, y_test_pred)

        ensemble_weights: np.array = weighted_ensemble.weights_

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
