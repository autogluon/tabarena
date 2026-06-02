from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    import pandas as pd


def get_example_data_for_task_type(
    *,
    task_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return example data for a given task type."""
    if task_type == "binary":
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    elif task_type == "multiclass":
        X, y = load_iris(return_X_y=True, as_frame=True)
    elif task_type == "regression":
        X, y = load_diabetes(return_X_y=True, as_frame=True)
    else:
        raise ValueError("Invalid task type")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test


def score_for_task_type(
    y_test: pd.Series,
    y_pred: pd.Series | pd.DataFrame,
    *,
    task_type: str,
) -> float:
    """Score (higher is better) the predictions for a given task type."""
    if task_type in ["binary", "multiclass"]:
        score = accuracy_score(y_test, y_pred)
        print("Accuracy:", score)
    elif task_type == "regression":
        score = -root_mean_squared_error(y_test, y_pred)
        print("Negative RMSE:", score)
    else:
        raise ValueError("Invalid task type")

    return score
