from __future__ import annotations
import warnings
import logging
import time
import pandas as pd

from autogluon.tabular.models import CatBoostModel
from tabarena.benchmark.feature_selection_methods.ag.metafs.method.Add_Pandas_Metafeatures import add_pandas_metadata_selection_columns

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class MetaFS:
    """MetaFS feature selector with resource limits."""

    def __init__(self, model):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None


    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        # Time limit
        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            time_start_fit = time.time()
            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
            if kwargs["time_limit"] <= 0:
                logger.warning(
                    f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                if n_max_features is not None and len(X.columns) > n_max_features:
                    X_out = X.sample(n=n_max_features, axis=1)
                    return X_out
                else:
                    return X
        result_matrix = pd.read_parquet("../../tabarena/benchmark/feature_selection_methods/ag/metafs/method/Pandas_Matrix_Complete.parquet")
        dataset_metadata = self._extract_metadata(y)
        """if dataset_id in datasets:
            result_matrix = result_matrix[result_matrix["dataset - id"] != dataset_id]"""
        comparison_result_matrix = self.create_empty_core_matrix_for_dataset(X, model)
        comparison_result_matrix = add_pandas_metadata_selection_columns(dataset_metadata, X, comparison_result_matrix)

        X_train_new, y_train_new = self.predict_improvement(result_matrix, comparison_result_matrix, X, y)

        if X_train_new.equals(X):
            return X_train_new
        else:
            return self.fit_transform(X_train_new, y_train_new, model, n_max_features, **kwargs)


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]


    def predict_improvement(self, result_matrix, comparison_result_matrix, X_train, y_train):
        y_result = result_matrix["improvement"]
        result_matrix = result_matrix.drop("improvement", axis=1)
        comparison_result_matrix = comparison_result_matrix.drop("improvement", axis=1)
        clf = CatBoostModel()
        clf.fit(X=result_matrix, y=y_result)
        comparison_result_matrix.columns = comparison_result_matrix.columns.astype(str)
        comparison_result_matrix = comparison_result_matrix[result_matrix.columns]
        prediction = clf.predict(X=comparison_result_matrix)
        prediction_df = pd.DataFrame(prediction, columns=["predicted_improvement"])
        prediction_concat_df = pd.concat([comparison_result_matrix[["dataset - id", "feature - name", "model"]], prediction_df], axis=1)
        best_operation = prediction_concat_df.nlargest(n=1, columns="predicted_improvement", keep="first")
        if best_operation["predicted_improvement"].values[0] < 0:
            return X_train, y_train
        else:
            X_train, y_train = self.remove_feature(X_train, y_train, best_operation)
        return X_train, y_train


    @staticmethod
    def remove_feature(X_train, y_train, prediction_result):
        worst_feature_row = prediction_result.loc[prediction_result['predicted_improvement'].idxmin()]
        worst_feature_name = worst_feature_row['feature - name'].split(' - ')[-1]
        print(f"Removing worst feature: {worst_feature_name} (predicted_improvement={worst_feature_row['predicted_improvement']:.4f})")

        if worst_feature_name in X_train.columns:
            X_train = X_train.drop(columns=[worst_feature_name])
        else:
            print(f"Warning: Feature '{worst_feature_name}' not found in X.")
        return X_train, y_train


    @staticmethod
    def create_empty_core_matrix_for_dataset(X_train, model) -> pd.DataFrame:
        columns = ['dataset - id', 'feature - name', 'operator', 'model', 'improvement']
        comparison_result_matrix = pd.DataFrame(columns=columns)
        for feature1 in X_train.columns:
            featurename = "without - " + str(feature1)
            new_rows = pd.DataFrame(columns=columns)
            operator = "delete"
            new_rows.loc[len(new_rows)] = [111111111, featurename, operator, model, 0]
            comparison_result_matrix = pd.concat([comparison_result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
        return comparison_result_matrix


    @staticmethod
    def _extract_metadata(y: pd.Series) -> dict:
        n_classes = y.nunique()
        if n_classes < 200:
            task_type = "Supervised Classification"
        else:
            task_type = "Supervised Regression"
        return {
            "task_id": None,
            "task_type": task_type,
            "number_of_classes": 'N/A'
        }
