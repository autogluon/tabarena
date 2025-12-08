from __future__ import annotations

import logging
import os
import time
from types import MappingProxyType

from autogluon.common.utils.try_import import try_import_catboost
from autogluon.core.constants import MULTICLASS, PROBLEM_TYPES_CLASSIFICATION, REGRESSION, QUANTILE, SOFTCLASS
from autogluon.core.utils.exceptions import TimeLimitExceeded

from autogluon.tabular.models.catboost.callbacks import EarlyStoppingCallback, MemoryCheckCallback, TimeCheckCallback
from autogluon.tabular.models.catboost.catboost_utils import CATBOOST_EVAL_METRIC_TO_LOSS_FUNCTION

logger = logging.getLogger(__name__)

from autogluon.tabular.models.catboost.catboost_model import CatBoostModel

from tabarena.benchmark.models.prep_ag.prep_mixin import ModelAgnosticPrepMixin

class PrepCatBoostModel(ModelAgnosticPrepMixin, CatBoostModel):
    """
    CatBoost model: https://catboost.ai/

    Hyperparameter options: https://catboost.ai/en/docs/references/training-parameters
    """
    ag_key = "prep_CAT"
    ag_name = "prep_CatBoost"
    ag_priority = 70
    ag_priority_by_problem_type = MappingProxyType({
        SOFTCLASS: 60
    })
    seed_name = "random_seed"
    
    # NOTE: Need to change .preprocess(X) to .preprocess(X,y,is_train=True) calls in _fit, otherwise functions are reusable

    # TODO: Use Pool in preprocess, optimize bagging to do Pool.split() to avoid re-computing pool for each fold! Requires stateful + y
    #  Pool is much more memory efficient, avoids copying data twice in memory
    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_gpus=0, num_cpus=-1, sample_weight=None, sample_weight_val=None, **kwargs):
        time_start = time.time()
        try_import_catboost()
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool

        ag_params = self._get_ag_params()
        params = self._get_model_params()

        params["thread_count"] = num_cpus
        if self.problem_type == SOFTCLASS:
            # FIXME: This is extremely slow due to unoptimized metric / objective sent to CatBoost
            from .catboost_softclass_utils import SoftclassCustomMetric, SoftclassObjective

            params.setdefault("loss_function",  SoftclassObjective.SoftLogLossObjective())
            params["eval_metric"] = SoftclassCustomMetric.SoftLogLossMetric()
        elif self.problem_type in [REGRESSION, QUANTILE]:
            # Choose appropriate loss_function that is as close as possible to the eval_metric
            params.setdefault(
                "loss_function",
                CATBOOST_EVAL_METRIC_TO_LOSS_FUNCTION.get(params["eval_metric"], params["eval_metric"])
            )

        model_type = CatBoostClassifier if self.problem_type in PROBLEM_TYPES_CLASSIFICATION else CatBoostRegressor
        num_rows_train = len(X)
        num_cols_train = len(X.columns)
        num_classes = self.num_classes if self.num_classes else 1  # self.num_classes could be None after initialization if it's a regression problem

        X = self.preprocess(X, y=y, is_train=True, prep_params=params['prep_params'])
        params.pop('prep_params', None)
        cat_features = list(X.select_dtypes(include="category").columns)
        X = Pool(data=X, label=y, cat_features=cat_features, weight=sample_weight)

        if X_val is None:
            eval_set = None
            early_stopping_rounds = None
        else:
            X_val = self.preprocess(X_val)
            X_val = Pool(data=X_val, label=y_val, cat_features=cat_features, weight=sample_weight_val)
            eval_set = X_val
            early_stopping_rounds = ag_params.get("early_stop", "adaptive")
            if isinstance(early_stopping_rounds, (str, tuple, list)):
                early_stopping_rounds = self._get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=early_stopping_rounds)

        if params.get("allow_writing_files", False):
            if "train_dir" not in params:
                try:
                    # TODO: What if path is in S3?
                    os.makedirs(os.path.dirname(self.path), exist_ok=True)
                except:
                    pass
                else:
                    params["train_dir"] = os.path.join(self.path, "catboost_info")

        # TODO: Add more control over these params (specifically early_stopping_rounds)
        verbosity = kwargs.get("verbosity", 2)
        if verbosity <= 1:
            verbose = False
        elif verbosity == 2:
            verbose = False
        elif verbosity == 3:
            verbose = 20
        else:
            verbose = True

        num_features = len(self._features)

        if num_gpus != 0:
            if "task_type" not in params:
                params["task_type"] = "GPU"
                logger.log(20, f"\tTraining {self.name} with GPU, note that this may negatively impact model quality compared to CPU training.")
                # TODO: Confirm if GPU is used in HPO (Probably not)
                # TODO: Adjust max_bins to 254?

        if params.get("task_type", None) == "GPU":
            if "colsample_bylevel" in params:
                params.pop("colsample_bylevel")
                logger.log(30, f"\t'colsample_bylevel' is not supported on GPU, using default value (Default = 1).")
            if "rsm" in params:
                params.pop("rsm")
                logger.log(30, f"\t'rsm' is not supported on GPU, using default value (Default = 1).")

        if self.problem_type == MULTICLASS and "rsm" not in params and "colsample_bylevel" not in params and num_features > 1000:
            # Subsample columns to speed up training
            if params.get("task_type", None) != "GPU":  # RSM does not work on GPU
                params["colsample_bylevel"] = max(min(1.0, 1000 / num_features), 0.05)
                logger.log(
                    30,
                    f'\tMany features detected ({num_features}), dynamically setting \'colsample_bylevel\' to {params["colsample_bylevel"]} to speed up training (Default = 1).',
                )
                logger.log(30, f"\tTo disable this functionality, explicitly specify 'colsample_bylevel' in the model hyperparameters.")
            else:
                params["colsample_bylevel"] = 1.0
                logger.log(30, f"\t'colsample_bylevel' is not supported on GPU, using default value (Default = 1).")

        logger.log(15, f"\tCatboost model hyperparameters: {params}")

        extra_fit_kwargs = dict()
        if params.get("task_type", None) != "GPU":
            callbacks = []
            if early_stopping_rounds is not None:
                callbacks.append(EarlyStoppingCallback(stopping_rounds=early_stopping_rounds, eval_metric=params["eval_metric"]))

            if num_rows_train * num_cols_train * num_classes > 5_000_000:
                # The data is large enough to potentially cause memory issues during training, so monitor memory usage via callback.
                callbacks.append(MemoryCheckCallback())
            if time_limit is not None:
                time_cur = time.time()
                time_left = time_limit - (time_cur - time_start)
                if time_left <= time_limit * 0.4:  # if 60% of time was spent preprocessing, likely not enough time to train model
                    raise TimeLimitExceeded
                callbacks.append(TimeCheckCallback(time_start=time_cur, time_limit=time_left))
            extra_fit_kwargs["callbacks"] = callbacks
        else:
            logger.log(30, f"\tWarning: CatBoost on GPU is experimental. If you encounter issues, use CPU for training CatBoost instead.")
            if time_limit is not None:
                params["iterations"] = self._estimate_iter_in_time_gpu(
                    X=X,
                    eval_set=eval_set,
                    time_limit=time_limit,
                    verbose=verbose,
                    params=params,
                    num_rows_train=num_rows_train,
                    time_start=time_start,
                    model_type=model_type,
                )
            if early_stopping_rounds is not None:
                if isinstance(early_stopping_rounds, int):
                    extra_fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
                elif isinstance(early_stopping_rounds, tuple):
                    extra_fit_kwargs["early_stopping_rounds"] = 50
        self.model = model_type(**params)

        # TODO: Custom metrics don't seem to work anymore
        # TODO: Custom metrics not supported in GPU mode
        # TODO: Callbacks not supported in GPU mode
        fit_final_kwargs = dict(
            eval_set=eval_set,
            verbose=verbose,
            **extra_fit_kwargs,
        )

        if eval_set is not None:
            fit_final_kwargs["use_best_model"] = True

        self.model.fit(X, **fit_final_kwargs)

        self.params_trained["iterations"] = self.model.tree_count_
