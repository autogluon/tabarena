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
        return get_calibrator(self.calibrator_type)

    @staticmethod
    def _to_binary_1d(pred_proba: np.ndarray) -> np.ndarray:
        """
        Convert binary proba output to 1d positive-class probabilities if needed.
        """
        # If already 1d, assume it's positive-class proba
        if pred_proba.ndim == 1:
            return pred_proba
        # If 2d, assume [:, 1] is positive class
        if pred_proba.ndim == 2 and pred_proba.shape[1] >= 2:
            return pred_proba[:, 1]
        return pred_proba

    def _calibrate_single_prediction(
        self,
        *,
        y_train: np.ndarray,
        pred_train: np.ndarray,
        pred_val: np.ndarray,
        pred_test: np.ndarray,
        problem_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit a calibrator on (pred_fit, y_fit), then transform both pred_val and pred_test.
        Handles binary->1d conversion.
        """
        calibrator = self.get_calibrator()
        calibrator.fit(pred_train, y_train)

        pred_val_cal = calibrator.predict_proba(pred_val)
        pred_test_cal = calibrator.predict_proba(pred_test)

        if problem_type == "binary":
            pred_val_cal = self._to_binary_1d(pred_val_cal)
            pred_test_cal = self._to_binary_1d(pred_test_cal)

        return pred_val_cal, pred_test_cal

    def _calibrate_post_hoc(
        self,
        *,
        y_train: np.ndarray,
        pred_train: np.ndarray,
        pred_val: np.ndarray,
        pred_test: np.ndarray,
        problem_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._calibrate_single_prediction(
            y_train=y_train,
            pred_train=pred_train,
            pred_val=pred_val,
            pred_test=pred_test,
            problem_type=problem_type,
        )

    def _calibrate_per_model(
        self,
        y_train: np.ndarray,
        pred_train: np.ndarray,
        pred_val: np.ndarray,
        pred_test: np.ndarray,
        problem_type: str,
        models: list[str],
    ):
        pred_val = copy.deepcopy(pred_val)
        pred_test = copy.deepcopy(pred_test)

        for i in range(len(models)):
            # Fit + transform both splits for this model
            pred_val_i_cal, pred_test_i_cal = self._calibrate_single_prediction(
                y_train=y_train,
                pred_train=pred_train[i],
                pred_val=pred_val[i],
                pred_test=pred_test[i],
                problem_type=problem_type,
            )

            pred_val[i] = pred_val_i_cal
            pred_test[i] = pred_test_i_cal
        return pred_val, pred_test

    def evaluate_task(self, dataset: str, fold: int, models: list[str]) -> dict[str, object]:
        models_og = models

        task_metadata = self.task_metrics_metadata[dataset]
        metric_name = task_metadata["metric"]
        problem_type = task_metadata["problem_type"]

        # Calibration only makes sense for probabilistic classification problems
        do_calibration = (problem_type in {"binary", "multiclass"}) and (self.calibrator_type is not None)

        if do_calibration:
            # For multiclass, avoid "fast" metric wrappers if they assume binary specifics
            use_fast_metrics = self.use_fast_metrics if problem_type == "binary" else False
            calibrate_after_ens = self.calibrate_after_ens
            calibrate_per_model = self.calibrate_per_model
        else:
            use_fast_metrics = self.use_fast_metrics
            calibrate_after_ens = False
            calibrate_per_model = False

        eval_metric, fit_eval_metric = self._get_metrics(
            metric_name=metric_name,
            problem_type=problem_type,
            use_fast_metrics=use_fast_metrics,
        )

        # Labels
        y_val = self.repo.labels_val(dataset=dataset, fold=fold)
        y_test = self.repo.labels_test(dataset=dataset, fold=fold)

        # Filter models + mapping to original order
        models_filtered = self.filter_models(dataset=dataset, fold=fold, models=models)
        models, models_filtered_idx = self._get_models_filtered_idx(models=models, models_filtered=models_filtered)

        pred_val, pred_test = self.get_preds_from_models(dataset=dataset, fold=fold, models=models)

        if calibrate_per_model:
            y_train, pred_train = self._get_train(y_val=y_val, pred_val=pred_val, y_test=y_test, pred_test=pred_test)

            pred_val, pred_test = self._calibrate_per_model(
                y_train=y_train,
                pred_train=pred_train,
                pred_val=pred_val,
                pred_test=pred_test,
                problem_type=problem_type,
                models=models,
            )

        y_train, pred_train = self._get_train(y_val=y_val, pred_val=pred_val, y_test=y_test, pred_test=pred_test)

        # Choose ensemble method/kwargs via hooks
        ensemble_method = self.get_ensemble_method_for_task(dataset=dataset, fold=fold, models=models)
        ensemble_kwargs = self.get_ensemble_method_kwargs_for_task(dataset=dataset, fold=fold, models=models)

        evaluator = self.evaluator_cls(
            ensemble_method=ensemble_method,
            ensemble_kwargs=ensemble_kwargs,
            eval_metric=eval_metric,
            fit_eval_metric=fit_eval_metric,
            problem_type=problem_type,
        )

        ensemble = evaluator.fit(pred_train=pred_train, y_train=y_train)

        y_test_pred, y_test_proc = evaluator.predict(ensemble=ensemble, pred=pred_test, y=y_test)

        need_val_pred = self.return_metric_error_val or calibrate_after_ens
        y_val_pred = None
        y_val_proc = None
        if need_val_pred:
            y_val_pred, y_val_proc = evaluator.predict(ensemble=ensemble, pred=pred_val, y=y_val)

        # -------------------------
        # Post-ensemble calibration (calibrate ensemble outputs)
        # -------------------------
        if calibrate_after_ens and do_calibration:
            y_train_cal, pred_train_cal = self._get_train(
                y_val=y_val_proc,
                pred_val=y_val_pred,
                y_test=y_test_proc,
                pred_test=y_test_pred,
            )

            y_val_pred, y_test_pred = self._calibrate_post_hoc(
                y_train=y_train_cal,
                pred_train=pred_train_cal,
                pred_val=y_val_pred,
                pred_test=y_test_pred,
                problem_type=problem_type,
            )

        results: dict[str, object] = {}
        results["metric_error"] = evaluator.score(y=y_test_proc, y_pred=y_test_pred)

        if self.return_metric_error_val:
            results["metric_error_val"] = evaluator.score(y=y_val_proc, y_pred=y_val_pred)

        if hasattr(ensemble, "weights_"):
            weights = ensemble.weights_
            ensemble_weights_fixed = np.zeros(len(models_og), dtype=np.float64)
            ensemble_weights_fixed[models_filtered_idx] = weights
            results["ensemble_weights"] = ensemble_weights_fixed

        return results


class EnsembleScorerCalibratedCV(EnsembleScorerCalibrated):
    def __init__(
        self,
        calibrator_n_splits: int = 10,
        calibrator_random_state: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.calibrator_n_splits = calibrator_n_splits
        self.calibrator_random_state = calibrator_random_state

        if self.optimize_on != "val":
            # This class intentionally only supports CV calibration on validation.
            raise ValueError(
                f"{self.__class__.__name__} only supports optimize_on='val', got optimize_on={self.optimize_on!r}"
            )

    def _get_cv_splitter(self, n_splits: int, problem_type: str, random_state: int):
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
          - calibrated_val_oof: out-of-fold calibrated probabilities for validation-side data
          - calibrated_test: calibrated probabilities for test, using calibrator fit on full validation-side data

        For binary, outputs are 1d positive-class probabilities.
        For multiclass, outputs are 2d probabilities.
        """
        n_splits = int(self.calibrator_n_splits) if self.calibrator_n_splits is not None else 0

        def _fit_full_and_predict(val_proba, val_y, test_proba):
            cal = calibrator_factory()
            cal.fit(val_proba, val_y)
            val_out = cal.predict_proba(val_proba)
            test_out = cal.predict_proba(test_proba)
            if problem_type == "binary":
                val_out = self._to_binary_1d(val_out)
                test_out = self._to_binary_1d(test_out)
            return val_out, test_out

        n_samples = int(proba_val.shape[0])
        if n_splits < 2 or n_samples < 2:
            return _fit_full_and_predict(proba_val, y_val, proba_test)

        if problem_type in ("binary", "multiclass"):
            _, counts = np.unique(y_val, return_counts=True)
            if counts.min() < 2:
                return _fit_full_and_predict(proba_val, y_val, proba_test)

        splitter = self._get_cv_splitter(
            n_splits=n_splits,
            problem_type=problem_type,
            random_state=random_state,
        )

        calibrated_val_oof = np.empty_like(proba_val)
        for train_idx, holdout_idx in splitter.split(None, y_val):
            cal = calibrator_factory()
            cal.fit(proba_val[train_idx], y_val[train_idx])
            oof_split = cal.predict_proba(proba_val[holdout_idx])
            if problem_type == "binary":
                oof_split = self._to_binary_1d(oof_split)
            calibrated_val_oof[holdout_idx] = oof_split

        cal_full = calibrator_factory()
        cal_full.fit(proba_val, y_val)
        calibrated_test = cal_full.predict_proba(proba_test)
        if problem_type == "binary":
            calibrated_test = self._to_binary_1d(calibrated_test)

        return calibrated_val_oof, calibrated_test

    def _calibrate_per_model(
        self,
        y_train: np.ndarray,
        pred_train: np.ndarray,
        pred_val: np.ndarray,
        pred_test: np.ndarray,
        problem_type: str,
        models: list[str],
    ):
        """
        optimize_on='val' only:
          - pred_val becomes OOF-calibrated (per model)
          - pred_test becomes calibrated via calibrator fit on full val-side (per model)
        """
        pred_val_out = copy.deepcopy(pred_val)
        pred_test_out = copy.deepcopy(pred_test)

        for i in range(len(models)):
            val_oof, test_cal = self._calibrate_with_cv_for_val_and_full_for_test(
                calibrator_factory=self.get_calibrator,
                proba_val=pred_train[i],
                y_val=y_train,
                proba_test=pred_test_out[i],
                problem_type=problem_type,
                random_state=self.calibrator_random_state,
            )
            pred_val_out[i] = val_oof
            pred_test_out[i] = test_cal

        return pred_val_out, pred_test_out

    def _calibrate_post_hoc(
        self,
        *,
        y_train: np.ndarray,
        pred_train: np.ndarray,
        pred_val: np.ndarray,
        pred_test: np.ndarray,
        problem_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        val_oof, test_cal = self._calibrate_with_cv_for_val_and_full_for_test(
            calibrator_factory=self.get_calibrator,
            proba_val=pred_train,
            y_val=y_train,
            proba_test=pred_test,
            problem_type=problem_type,
            random_state=self.calibrator_random_state + 1,
        )

        return val_oof, test_cal
