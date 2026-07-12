from __future__ import annotations

import copy
import os
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from autogluon.core.metrics import Scorer, get_metric
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection

from tabarena.metrics import _fast_log_loss
from tabarena.utils import task_to_tid_fold
from tabarena.utils.aux_metric import get_aux_metric_map
from tabarena.utils.parallel_for import parallel_for

from .configuration_list_scorer import ConfigurationListScorer

if TYPE_CHECKING:
    from tabarena.repository.evaluation_repository import EvaluationRepository
    from tabarena.utils.rank_utils import RankScorer


@lru_cache(maxsize=1)
def get_fast_roc_auc() -> Scorer:
    """Lazily import the C++ fast roc_auc metric, falling back to the sklearn implementation.

    The import is deferred (rather than performed at module load) because the C++ extension
    requires g++ to compile and can raise on import. Set the env var
    ``TABARENA_SKIP_FAST_ROC_AUC=1`` to skip the fast metric entirely and always use the
    sklearn implementation.
    """
    if os.environ.get("TABARENA_SKIP_FAST_ROC_AUC", "0") == "1":
        return get_metric(metric="roc_auc", problem_type="binary")
    try:
        # FIXME: Requires g++, can lead to an exception on import as it needs to compile C code.
        from tabarena.metrics._fast_roc_auc import fast_roc_auc_cpp
    except (OSError, ValueError) as e:
        # OSError covers g++ missing (FileNotFoundError) and read-only filesystems
        # (EROFS, e.g. Singularity containers); ValueError covers a non-zero/timed-out
        # g++ run, including being unable to write cpp_auc.so into a read-only install dir.
        print(
            f"Warning: Failed to obtain c++ roc_auc metric ({type(e).__name__}: {e}). "
            "Try installing g++ or set TABARENA_SKIP_FAST_ROC_AUC=1. "
            "Falling back to sklearn implementation...",
        )
        return get_metric(metric="roc_auc", problem_type="binary")
    return fast_roc_auc_cpp


class TaskEvaluator:
    """Minimal per-task evaluator:
      - knows how to fit an ensemble given (y_train, pred_train)
      - knows how to predict/score given (y, pred)
    It does NOT know anything about repo/dataset/fold/models or optimize_on.
    """

    def __init__(
        self,
        *,
        ensemble_method: type,
        ensemble_kwargs: dict,
        eval_metric: Scorer,
        fit_eval_metric: Scorer,
        problem_type: str,
        aux_eval_metric: Scorer | None = None,
    ):
        self._ensemble_method = ensemble_method
        self._ensemble_kwargs = ensemble_kwargs
        self._eval_metric = eval_metric
        self._fit_eval_metric = fit_eval_metric
        self._aux_eval_metric = aux_eval_metric
        self._predict_problem_type = getattr(eval_metric, "post_problem_type", problem_type)
        self._fit_problem_type = getattr(fit_eval_metric, "post_problem_type", problem_type)
        self.problem_type = problem_type

    @staticmethod
    def _maybe_preprocess_bulk(metric: Scorer, y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(metric, "preprocess_bulk"):
            y, pred = metric.preprocess_bulk(y, pred)
        return y, pred

    def init_ens(self):
        return self._ensemble_method(
            problem_type=self._fit_problem_type,
            metric=self._fit_eval_metric,
            **self._ensemble_kwargs,
        )

    def fit(self, *, pred_train: np.ndarray, y_train: np.ndarray):
        ensemble = self.init_ens()

        y_train, pred_train = self._maybe_preprocess_bulk(self._fit_eval_metric, y_train, pred_train)
        ensemble.fit(predictions=pred_train, labels=y_train)

        # Ensure prediction interface aligns with the metric we ultimately evaluate with
        ensemble.problem_type = self._predict_problem_type
        return ensemble

    def predict(
        self, *, ensemble, pred: np.ndarray, y: np.ndarray, eval_metric: Scorer | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if eval_metric is None:
            eval_metric = self._eval_metric

        y, pred = self._maybe_preprocess_bulk(eval_metric, y, pred)

        if eval_metric.needs_pred:
            reset_problem_type = False
            original_problem_type = ensemble.problem_type
            new_problem_type = self.problem_type
            if original_problem_type != new_problem_type:
                reset_problem_type = True
                ensemble.problem_type = new_problem_type
            y_pred = ensemble.predict(pred)
            if reset_problem_type:
                ensemble.problem_type = original_problem_type
        else:
            y_pred = ensemble.predict_proba(pred)
        return y_pred, y

    def error(self, *, y: np.ndarray, y_pred: np.ndarray, eval_metric: Scorer | None = None) -> float:
        if eval_metric is None:
            eval_metric = self._eval_metric
        return eval_metric.error(y, y_pred)

    def error_fit(self, *, y: np.ndarray, y_pred: np.ndarray) -> float:
        return self._fit_eval_metric.error(y, y_pred)

    def run(
        self,
        *,
        pred_train: np.ndarray,
        y_train: np.ndarray,
        pred_test: np.ndarray,
        y_test: np.ndarray,
        return_metric_error_val: bool,
        pred_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> tuple[dict[str, object], object]:
        ensemble = self.fit(pred_train=pred_train, y_train=y_train)

        y_test_pred, y_test_proc = self.predict(ensemble=ensemble, pred=pred_test, y=y_test)
        results: dict[str, object] = {"metric_error": self.error(y=y_test_proc, y_pred=y_test_pred)}

        if self._aux_eval_metric is not None:
            y_test_pred_aux, y_test_proc_aux = self.predict(
                ensemble=ensemble, pred=pred_test, y=y_test, eval_metric=self._aux_eval_metric
            )
            results["aux_metric_error"] = self.error(
                y=y_test_proc_aux, y_pred=y_test_pred_aux, eval_metric=self._aux_eval_metric
            )

        if return_metric_error_val:
            if pred_val is None or y_val is None:
                raise ValueError("pred_val and y_val must be provided when return_metric_error_val=True")
            y_val_pred, y_val_proc = self.predict(ensemble=ensemble, pred=pred_val, y=y_val)
            results["metric_error_val"] = self.error(y=y_val_proc, y_pred=y_val_pred)
            if self._aux_eval_metric is not None:
                y_val_pred_aux, y_val_proc_aux = self.predict(
                    ensemble=ensemble, pred=pred_val, y=y_val, eval_metric=self._aux_eval_metric
                )
                results["aux_metric_error_val"] = self.error(
                    y=y_val_proc_aux, y_pred=y_val_pred_aux, eval_metric=self._aux_eval_metric
                )

        if hasattr(ensemble, "weights_"):
            results["ensemble_weights"] = ensemble.weights_

        return results, ensemble


class EnsembleScorer:
    def __init__(
        self,
        repo: EvaluationRepository,
        task_metrics_metadata,
        evaluator_cls: type[TaskEvaluator] = TaskEvaluator,
        ensemble_method: type = EnsembleSelection,
        ensemble_method_kwargs: dict | None = None,
        proxy_fit_metric_map: dict | None = None,
        use_fast_metrics: bool = True,
        optimize_on: str = "val",
        return_metric_error_val: bool = True,
    ):
        """Parameters
        ----------
        repo: EvaluationRepository
        task_metrics_metadata
        ensemble_method: Type = EnsembleSelection
        ensemble_method_kwargs: dict, default = None
        proxy_fit_metric_map: dict, default = None
        use_fast_metrics: bool, default = True
        optimize_on: str, default = "val"
            If "val", optimizes on the validation data (normal process that mirrors what can be done in practice)
            If "test", optimizes on the test data (cheat mode, use this only for debugging and testing generalization gaps)
        return_metric_error_val: bool, default = True
            If True, will compute and return `metric_error_val` using the fitted ensemble in the output dict of `evaluate_task`.
        """
        if proxy_fit_metric_map is None:
            proxy_fit_metric_map = {}
        if ensemble_method_kwargs is None:
            ensemble_method_kwargs = {}

        ensemble_method_kwargs = copy.deepcopy(ensemble_method_kwargs)
        if "ensemble_size" not in ensemble_method_kwargs:
            ensemble_method_kwargs["ensemble_size"] = 100

        self.repo = repo
        self.evaluator_cls = evaluator_cls
        self.ensemble_method: type = ensemble_method
        self.ensemble_method_kwargs = ensemble_method_kwargs
        self.task_metrics_metadata = task_metrics_metadata
        self.proxy_fit_metric_map = proxy_fit_metric_map
        self.use_fast_metrics = use_fast_metrics

        assert optimize_on in ["val", "test"]
        self.optimize_on = optimize_on
        self.return_metric_error_val = return_metric_error_val

        # (dataset, fold) -> (model -> row index, pred_val, pred_test); see cache_task_preds.
        self._preds_cache: dict[tuple[str, int], tuple[dict[str, int], np.ndarray, np.ndarray]] = {}

    # -------------------------
    # Extension hooks
    # -------------------------
    def filter_models(self, dataset: str, fold: int, models: list[str]) -> list[str]:
        return models

    def get_ensemble_method_for_task(self, dataset: str, fold: int, models: list[str]) -> type:
        return self.ensemble_method

    def get_ensemble_method_kwargs_for_task(self, dataset: str, fold: int, models: list[str]) -> dict:
        return copy.deepcopy(self.ensemble_method_kwargs)

    def subsample_val_data(
        self,
        *,
        dataset: str,
        fold: int,
        problem_type: str,
        y_val: np.ndarray,
        pred_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Restrict the validation data available for post-hoc ensemble decisions. No-op by default.

        Subclasses may return a subset of ``(y_val, pred_val)`` — the *same* rows for both, selected
        along the validation-sample axis (``y_val[idx]`` and ``pred_val[:, idx]``, where ``pred_val``
        is ``(n_models, n_rows)`` or ``(n_models, n_rows, n_classes)``). The returned data is used
        both to fit the weighted ensemble (when ``optimize_on='val'``) and to compute
        ``metric_error_val``. The test set is never touched here, so ``metric_error`` always reflects
        the full test data.
        """
        return y_val, pred_val

    # -------------------------
    # Metrics helpers
    # -------------------------
    def _get_metric_from_name(
        self, metric_name: str, problem_type: str, use_fast_metrics: bool | None = None
    ) -> Scorer:
        if use_fast_metrics is None:
            use_fast_metrics = self.use_fast_metrics
        if use_fast_metrics:
            return self._get_fast_metric_if_exist(metric_name=metric_name, problem_type=problem_type)
        return get_metric(metric=metric_name, problem_type=problem_type)

    def _get_fast_metric_if_exist(self, metric_name: str, problem_type: str) -> Scorer:
        if metric_name == "log_loss":
            return _fast_log_loss.fast_log_loss
        if metric_name == "roc_auc":
            return get_fast_roc_auc()
        return get_metric(metric=metric_name, problem_type=problem_type)

    def _get_metrics(
        self,
        metric_name: str,
        problem_type: str,
        use_fast_metrics: bool | None = None,
    ) -> tuple[Scorer, Scorer]:
        fit_metric_name = self.proxy_fit_metric_map.get(metric_name, metric_name)

        eval_metric = self._get_metric_from_name(
            metric_name=metric_name, problem_type=problem_type, use_fast_metrics=use_fast_metrics
        )
        fit_eval_metric = self._get_metric_from_name(
            metric_name=fit_metric_name, problem_type=problem_type, use_fast_metrics=use_fast_metrics
        )

        return eval_metric, fit_eval_metric

    def _get_aux_metric(self, problem_type: str) -> Scorer | None:
        aux_map = get_aux_metric_map()
        if aux_map is None:
            return None
        aux_metric_name = aux_map.get(problem_type)
        if aux_metric_name is None:
            return None
        return get_metric(metric=aux_metric_name, problem_type=problem_type)

    def cache_task_preds(self, dataset: str, fold: int, models: list[str]) -> None:
        """Preload and cache this task's predictions for ``models``.

        Subsequent :meth:`get_preds_from_models` calls on the task whose models are all cached
        return row slices of the cached arrays instead of re-reading from the repo (the values
        are identical — the cache holds exactly what the repo loads, rows keyed by model). Use
        when evaluating many model subsets on the same task, so the (memmap) predictions are
        read once. Requests involving an uncached model fall through to the repo.
        """
        pred_val, pred_test = self._load_preds_from_repo(dataset=dataset, fold=fold, models=models)
        self._preds_cache[(dataset, fold)] = ({m: i for i, m in enumerate(models)}, pred_val, pred_test)

    def _load_preds_from_repo(self, dataset: str, fold: int, models: list[str]) -> tuple[np.ndarray, np.ndarray]:
        pred_val = self.repo.predict_val_multi(dataset=dataset, fold=fold, configs=models, enforce_binary_1d=True)
        pred_test = self.repo.predict_test_multi(dataset=dataset, fold=fold, configs=models, enforce_binary_1d=True)
        return pred_val, pred_test

    def get_preds_from_models(self, dataset: str, fold: int, models: list[str]) -> tuple[np.ndarray, np.ndarray]:
        cached = self._preds_cache.get((dataset, fold))
        if cached is not None:
            model_idx, pred_val, pred_test = cached
            if all(m in model_idx for m in models):
                idx = [model_idx[m] for m in models]
                return pred_val[idx], pred_test[idx]
        return self._load_preds_from_repo(dataset=dataset, fold=fold, models=models)

    def _get_train(
        self,
        y_val: np.ndarray,
        pred_val: np.ndarray,
        y_test: np.ndarray,
        pred_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.optimize_on == "val":
            return y_val, pred_val
        if self.optimize_on == "test":
            return y_test, pred_test
        raise ValueError(f"Invalid value for `optimize_on`: {self.optimize_on}")

    # -------------------------
    # Main entry
    # -------------------------
    def evaluate_task(self, dataset: str, fold: int, models: list[str]) -> dict[str, object]:
        models_og = models

        task_metadata = self.task_metrics_metadata[dataset]
        metric_name = task_metadata["metric"]
        problem_type = task_metadata["problem_type"]

        eval_metric, fit_eval_metric = self._get_metrics(
            metric_name=metric_name,
            problem_type=problem_type,
        )
        aux_eval_metric = self._get_aux_metric(problem_type=problem_type)

        y_val = self.repo.labels_val(dataset=dataset, fold=fold)
        y_test = self.repo.labels_test(dataset=dataset, fold=fold)

        models_filtered = self.filter_models(dataset=dataset, fold=fold, models=models)
        models, models_filtered_idx = self._get_models_filtered_idx(models=models, models_filtered=models_filtered)

        pred_val, pred_test = self.get_preds_from_models(dataset=dataset, fold=fold, models=models)

        # Restrict the validation data used for post-hoc ensemble decisions (no-op by default).
        y_val, pred_val = self.subsample_val_data(
            dataset=dataset,
            fold=fold,
            problem_type=problem_type,
            y_val=y_val,
            pred_val=pred_val,
        )

        y_train, pred_train = self._get_train(y_val=y_val, pred_val=pred_val, y_test=y_test, pred_test=pred_test)

        ensemble_method = self.get_ensemble_method_for_task(dataset=dataset, fold=fold, models=models)
        ensemble_kwargs = self.get_ensemble_method_kwargs_for_task(dataset=dataset, fold=fold, models=models)

        evaluator = self.evaluator_cls(
            ensemble_method=ensemble_method,
            ensemble_kwargs=ensemble_kwargs,
            eval_metric=eval_metric,
            fit_eval_metric=fit_eval_metric,
            problem_type=problem_type,
            aux_eval_metric=aux_eval_metric,
        )

        results, _fitted_ensemble = evaluator.run(
            pred_train=pred_train,
            y_train=y_train,
            pred_test=pred_test,
            y_test=y_test,
            return_metric_error_val=self.return_metric_error_val,
            pred_val=pred_val,
            y_val=y_val,
        )

        if "ensemble_weights" in results:
            ensemble_weights_fixed = np.zeros(len(models_og), dtype=np.float64)
            ensemble_weights_fixed[models_filtered_idx] = results["ensemble_weights"]
            results["ensemble_weights"] = ensemble_weights_fixed

        return results

    def _get_models_filtered_idx(self, models: list[str], models_filtered: list[str]) -> tuple[list[str], list[int]]:
        models_filtered_set = set(models_filtered)

        models_seen: set[str] = set()
        models_filtered_ordered = [
            m for m in models if (m in models_filtered_set) and not (m in models_seen or models_seen.add(m))
        ]

        if len(models_filtered_set) < len(models_filtered_ordered):
            models_idx: dict[str, list[int]] = {}
            for i, m in enumerate(models):
                models_idx.setdefault(m, []).append(i)
            models_filtered_idx = [models_idx[m].pop(0) for m in models_filtered_ordered]
        else:
            models_filtered_idx = [models.index(m) for m in models_filtered_ordered]

        return models_filtered_ordered, models_filtered_idx


class EnsembleScorerMaxModels(EnsembleScorer):
    """Identical to EnsembleScorer, with the addition of `max_models` and `max_models_per_type`.

    Parameters
    ----------
    max_models: int, default = None
        If specified, will limit ensemble candidates to the top `max_models` highest validation score models.
        This logic is applied after the filtering from `max_models_per_type`.
    max_models_per_type: int | str, default = None
        If specified, will limit ensemble candidates of a given model type to the top `max_models_per_type` highest validation score models.
        If "auto", scales dynamically with the number of rows in the dataset.
    """

    def __init__(
        self,
        repo: EvaluationRepository,
        max_models: int | None = None,
        max_models_per_type: int | str | None = None,
        **kwargs,
    ):
        super().__init__(repo=repo, **kwargs)
        assert self.repo is not None
        if max_models is not None:
            assert max_models >= 0
        if max_models_per_type is not None:
            if isinstance(max_models_per_type, str):
                assert max_models_per_type == "auto"
            else:
                assert max_models_per_type >= 0
        self.max_models = max_models
        self.max_models_per_type = max_models_per_type

    def filter_models(self, dataset: str, fold: int, models: list[str]) -> list[str]:
        """Filters models by user-defined logic. Used in class extensions."""
        if self.max_models is not None or self.max_models_per_type is not None:
            if (
                self.max_models_per_type is not None
                and isinstance(self.max_models_per_type, str)
                and self.max_models_per_type == "auto"
            ):
                max_models_per_type = self._get_max_models_per_type_auto(dataset=dataset)
            else:
                max_models_per_type = self.max_models_per_type
            models = self.repo._zeroshot_context.get_top_configs(
                dataset=dataset,
                fold=fold,
                configs=models,
                max_models=self.max_models,
                max_models_per_type=max_models_per_type,
            )
        return models

    def _get_max_models_per_type_auto(self, dataset: str) -> int:
        """Logic to mimic AutoGluon's default setting for `max_models_per_type`."""
        # TODO: Make it easier to get this info without accessing private variables in repo
        df_metadata = self.repo._zeroshot_context.df_metadata
        num_rows = int(df_metadata[df_metadata["dataset"] == dataset].iloc[0]["NumberOfInstances"] * 9 / 10)
        if num_rows < 1000:
            max_models_per_type = 1
        elif num_rows < 5000:
            max_models_per_type = 2
        elif num_rows < 10000:
            max_models_per_type = 3
        elif num_rows < 15000:
            max_models_per_type = 4
        elif num_rows < 20000:
            max_models_per_type = 5
        elif num_rows < 25000:
            max_models_per_type = 6
        elif num_rows < 30000:
            max_models_per_type = 7
        elif num_rows < 35000:
            max_models_per_type = 8
        elif num_rows < 40000:
            max_models_per_type = 9
        elif num_rows < 45000:
            max_models_per_type = 10
        elif num_rows < 50000:
            max_models_per_type = 11
        else:
            max_models_per_type = 12
        return max_models_per_type


# FIXME: Add temperature scaling!!
class EnsembleSelectionConfigScorer(ConfigurationListScorer):
    def __init__(
        self,
        tasks: list[str],
        repo: EvaluationRepository,
        ranker: RankScorer,
        tid_to_dataset_name_dict: dict[int, str],
        task_metrics_metadata: dict[int, dict[str, str]],
        ensemble_size=100,
        ensemble_selection_kwargs=None,
        backend: str = "native",
        use_fast_metrics: bool = True,
        proxy_fit_metric_map: dict | str | None = None,  # TODO: Add unit test
        ensemble_cls: type[EnsembleScorer] = EnsembleScorerMaxModels,
        ensemble_kwargs: dict | None = None,
    ):
        """A scorer object to evaluate configs via simulating ensemble selection.

        :param tasks: The list of tasks to consider for scoring.
        :param ranker: The ranking object used to compute scores on each task.
        :param task_metrics_metadata: dictionary containing metric information and problem type for all tasks
        :param ensemble_size: The maximum ensemble selection iterations when fitting the ensemble. TODO: Remove?
        :param ensemble_selection_kwargs: kwargs to pass to the init of the ensemble selection model.
        :param backend: Options include ["native", "ray"].
        :param use_fast_metrics: If True, will leverage optimized eval metrics to speed up config scoring.
        :param proxy_fit_metric_map:
            If eval_metric is among the keys in the `proxy_fit_metric_map` dictionary,
            the value eval_metric will be used during the weighted ensemble fitting process as a proxy.
            For example, the proxy metric could be faster to compute while producing a similar end result.
            If None: Do not use proxy metrics, equivalent to {}.
            If 'roc_auc_to_log_loss': set to {'roc_auc': 'log_loss'}, making 'log_loss' a proxy to 'roc_auc'
        :param: ensemble_cls: The ensemble class to use for fitting and scoring.
        :param: ensemble_kwargs: The kwargs to pass to the init call of `ensemble_cls`.
        """
        super().__init__(tasks=tasks)
        if ensemble_kwargs is None:
            ensemble_kwargs = {}
        ensemble_kwargs = ensemble_kwargs.copy()
        self.repo = repo
        self.ranker = ranker
        self.tid_to_dataset_name_dict = tid_to_dataset_name_dict
        self.ensemble_size = ensemble_size
        if ensemble_selection_kwargs is None:
            ensemble_selection_kwargs = {}
        self.ensemble_selection_kwargs = ensemble_selection_kwargs
        assert backend in ["native", "ray"]
        self.backend = backend
        self.use_fast_metrics = ensemble_kwargs.pop("use_fast_metrics", use_fast_metrics)
        if "proxy_fit_metric_map" in ensemble_kwargs:
            proxy_fit_metric_map = ensemble_kwargs.pop("proxy_fit_metric_map")
        if proxy_fit_metric_map is None:
            proxy_fit_metric_map = {}
        elif isinstance(proxy_fit_metric_map, str):
            assert proxy_fit_metric_map == "roc_auc_to_log_loss"
            proxy_fit_metric_map = {"roc_auc": "log_loss"}  # log_loss is fast to compute and a good proxy for roc_auc
        self.proxy_fit_metric_map = proxy_fit_metric_map

        ensemble_selection_kwargs = copy.deepcopy(ensemble_selection_kwargs)
        ensemble_selection_kwargs["ensemble_size"] = ensemble_size

        self.ensemble_scorer = ensemble_cls(
            repo=repo,
            task_metrics_metadata=task_metrics_metadata,
            ensemble_method_kwargs=ensemble_selection_kwargs,
            proxy_fit_metric_map=proxy_fit_metric_map,
            use_fast_metrics=self.use_fast_metrics,
            **ensemble_kwargs,
        )

    @classmethod
    def from_repo(cls, repo: EvaluationRepository, **kwargs):
        zeroshot_simulator_context = repo._zeroshot_context
        if "tasks" not in kwargs:
            kwargs["tasks"] = zeroshot_simulator_context.get_tasks()

        dataset_to_tid_dict = zeroshot_simulator_context.dataset_to_tid_dict
        task_metrics_metadata = zeroshot_simulator_context.df_metrics
        task_metrics_metadata = {
            dataset: task_metrics_metadata.loc[dataset].to_dict()
            for dataset in task_metrics_metadata.index
            if dataset in dataset_to_tid_dict
        }

        return cls(
            repo=repo,
            ranker=zeroshot_simulator_context.rank_scorer,
            tid_to_dataset_name_dict=zeroshot_simulator_context.tid_to_dataset_dict,
            task_metrics_metadata=task_metrics_metadata,
            **kwargs,
        )

    def evaluate_task(self, task: str, models: list[str]) -> dict[str, object]:
        tid, fold = task_to_tid_fold(task=task)
        dataset = self.tid_to_dataset_name_dict[tid]
        return self.ensemble_scorer.evaluate_task(dataset=dataset, fold=fold, models=models)

    def compute_errors(self, configs: list[str]) -> dict[str, dict[str, object]]:
        """Compute and return test errors and ensemble weights for all tasks on the user-specified list of configs.

        :param configs: List of model config names to ensemble and compute test errors with.
        :return: dict:
            task -> dict:
                metric_error: test evaluation metric error of the ensemble.
                metric_error_val: val evaluation metric error of the ensemble.
                ensemble_weights: model weights in the ensemble. Model weights are stored in a numpy array, with weights corresponding to the order of `configs`.
        """
        engine = self.backend
        if engine == "native":
            engine = "sequential"

        context = dict(
            self=self,
            models=configs,
        )

        progress_bar = engine != "sequential"

        inputs = [{"task": task} for task in self.tasks]
        results_rows = parallel_for(
            self.__class__.evaluate_task,
            inputs=inputs,
            context=context,
            engine=engine,
            progress_bar=progress_bar,
        )
        return dict(zip(self.tasks, results_rows, strict=False))

    def compute_ranks(self, errors: dict[str, float]) -> dict[str, float]:
        ranks = {}
        for dataset, error in errors.items():
            rank = self.ranker.rank(dataset, error)  # FIXME: Use score or error?
            ranks[dataset] = rank
        return ranks

    def compute_rank_mean(self, errors: dict[str, float]) -> float:
        ranks = self.compute_ranks(errors=errors)
        return np.mean(list(ranks.values()))

    def score(self, configs: list[str]) -> float:
        errors, _ensemble_weights = self.compute_errors(configs=configs)
        return self.compute_rank_mean(errors)

    def score_per_dataset(self, configs: list[str]) -> dict[str, float]:
        errors, _ensemble_weights = self.compute_errors(configs=configs)
        return self.compute_ranks(errors=errors)

    def subset(self, tasks):
        return self.__class__(
            tasks=tasks,
            repo=self.repo,
            ranker=self.ranker,
            ensemble_size=self.ensemble_size,
            ensemble_selection_kwargs=self.ensemble_selection_kwargs,
            tid_to_dataset_name_dict=self.tid_to_dataset_name_dict,
            task_metrics_metadata=self.ensemble_scorer.task_metrics_metadata,
        )
