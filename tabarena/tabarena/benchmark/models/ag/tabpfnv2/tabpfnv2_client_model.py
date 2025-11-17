import pandas as pd

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel


class TabPFNV2ClientModel(AbstractModel):
    ag_key = "TABPFNV2_CLIENT"
    ag_name = "TabPFNv2Client"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def get_model_cls(self):
        from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
        if self.problem_type in ['binary', 'multiclass']:
            model_cls = TabPFNClassifier
        else:
            model_cls = TabPFNRegressor
        return model_cls

    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        model_cls = self.get_model_cls()

        metric_map = {
            "roc_auc": "roc",
            "accuracy": "acc",
            "balanced_accuracy": "balanced_acc",
            "log_loss": "log_loss",
            "rmse": "rmse",
            "root_mean_squared_error": "rmse",
            "r2": "r2",
        }

        eval_metric_tabpfn = metric_map[self.eval_metric.name]

        hyp = self._get_model_params()
        self.model = model_cls(
            model="latest_tabpfn_hosted",
            optimize_metric=eval_metric_tabpfn,
            # n_estimators=32,
            **hyp,
        )

        ag_params = self._get_ag_params()
        max_classes = ag_params.get("max_classes")
        if self.problem_type in [BINARY, MULTICLASS]:
            if max_classes is not None and self.num_classes > max_classes:
                # TODO: Move to earlier stage when problem_type is checked
                raise AssertionError(f"Max allowed classes for the model is {max_classes}, " f"but found {self.num_classes} classes.")

        X = self.preprocess(X)
        # if X_val is not None:
        #     X_val = self.preprocess(X_val)

        self.model = self.model.fit(
            X=X,
            y=y,
            # X_val=X_val,
            # y_val=y_val,
        )

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Converts categorical to label encoded integers
        Keeps missing values, as TabPFN automatically handles missing values internally.
        """
        X = super()._preprocess(X, **kwargs)
        # if self._feature_generator is None:
        #     self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
        #     self._feature_generator.fit(X=X)
        # if self._feature_generator.features_in:
        #     X = X.copy()
        #     X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X

    def _set_default_params(self):
        default_params = {}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_classes": 10,
            }
        )
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
           "fold_fitting_strategy": "sequential_local",  # FIXME: Comment out after debugging for large speedup
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _ag_params(self) -> set:
        return {"max_classes"}

    def _more_tags(self) -> dict:
        tags = {"can_refit_full": True}
        return tags
