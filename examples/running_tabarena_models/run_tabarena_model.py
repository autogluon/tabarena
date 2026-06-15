"""Use a TabArena model on your own data — three usage levels, simplest first.

Every TabArena model is an AutoGluon model, so you train and predict with the AutoGluon
API (no benchmark involved). This single file shows three increasingly powerful ways to
use one on a plain ``(X, y)`` dataset:

  [1] single  — fit one model with its default config (one train + predict).
  [2] bagged  — cross-validation bagging (the TabArena default; also gives a validation score).
  [3] tuned   — tune several random configs and post-hoc ensemble them (via TabularPredictor).

Set ``MODEL`` and ``TASK_TYPE`` below, then run the file to see all three.

Note on preprocessing: the low-level model API ([1]/[2]) needs AutoGluon's feature/label
preprocessing applied first; ``TabularPredictor`` ([3]) does its own, so it gets the raw data.
Each function below therefore handles its own preprocessing — keeping every example self-contained.
"""

from __future__ import annotations

from autogluon.core.data import LabelCleaner
from autogluon.core.models import BaggedEnsembleModel
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularPredictor
from data_utils import get_example_data_for_task_type, score_for_task_type

from tabarena.models.utils import get_configs_generator_from_name

MODEL = "RealMLP"
"""Any TabArena model name (not all support all task types). Common choices:
"RealMLP", "TabM", "LightGBM", "CatBoost", "XGBoost", "ModernNCA", "TabPFNv2", "TabICL",
"TorchMLP", "TabDPT", "EBM", "FastaiMLP", "ExtraTrees", "RandomForest", "KNN", "Linear".
For the full list see `tabarena.models.utils.get_configs_generator_from_name`. For practical
use you can also import the class directly, e.g. `from tabarena.models.realmlp.model import RealMLPModel`.
"""

TASK_TYPE = "binary"
"""One of "binary", "multiclass", "regression"."""


def _preprocess(X_train, X_test, y_train, y_test, *, task_type):
    """AutoGluon's standard feature + label preprocessing (for the low-level model API)."""
    feature_generator = AutoMLPipelineFeatureGenerator()
    label_cleaner = LabelCleaner.construct(problem_type=task_type, y=y_train)
    return (
        feature_generator.fit_transform(X_train),
        feature_generator.transform(X_test),
        label_cleaner.transform(y_train),
        label_cleaner.transform(y_test),
    )


def run_single(X_train, X_test, y_train, y_test, *, task_type, model_cls, model_config):
    """[1] One model, default config: a single fit + predict (no validation split)."""
    X_train, X_test, y_train, y_test = _preprocess(X_train, X_test, y_train, y_test, task_type=task_type)
    model = model_cls(problem_type=task_type, **model_config).fit(X=X_train, y=y_train)
    score_for_task_type(y_test, model.predict(X=X_test), task_type=task_type)


def run_bagged(X_train, X_test, y_train, y_test, *, task_type, model_cls, model_config):
    """[2] Cross-validation bagging (TabArena's default: 8 folds, out-of-fold validation)."""
    X_train, X_test, y_train, y_test = _preprocess(X_train, X_test, y_train, y_test, task_type=task_type)
    model = BaggedEnsembleModel(model_cls(problem_type=task_type, **model_config))
    model.params["fold_fitting_strategy"] = "sequential_local"
    model = model.fit(X=X_train, y=y_train, k_fold=8)
    print(f"Validation {model.eval_metric.name}:", model.score_with_oof(y=y_train))
    score_for_task_type(y_test, model.predict(X=X_test), task_type=task_type)


def run_tuned(X_train, X_test, y_train, y_test, *, task_type, model_cls, n_configs=5):
    """[3] Tune `n_configs` random configs + post-hoc ensemble, via TabularPredictor.

    TabularPredictor handles preprocessing itself, so it takes the raw data. TabArena tuned
    each model over 200 configs; we use a few here for a quick demo.
    """
    hpo_configs = get_configs_generator_from_name(MODEL).generate_all_configs_lst(num_random_configs=n_configs)
    train_data = X_train.copy()
    train_data["target"] = y_train
    predictor = TabularPredictor(
        label="target",
        problem_type=task_type,
        eval_metric="rmse" if task_type == "regression" else "accuracy",
    ).fit(
        train_data,
        hyperparameters={model_cls: hpo_configs},
        num_bag_folds=8,
        fit_weighted_ensemble=True,
        time_limit=360,  # increase for real use
    )
    predictor.leaderboard(display=True)
    score_for_task_type(y_test, predictor.predict(X_test), task_type=task_type)


if __name__ == "__main__":
    model_meta = get_configs_generator_from_name(model_name=MODEL)
    model_cls = model_meta.model_cls
    model_config = model_meta.manual_configs[0]  # the model's default config

    data = get_example_data_for_task_type(task_type=TASK_TYPE)

    print(f"\n=== [1] single {MODEL} on a {TASK_TYPE} task ===")
    run_single(*data, task_type=TASK_TYPE, model_cls=model_cls, model_config=model_config)

    print(f"\n=== [2] cross-validation bagged {MODEL} ===")
    run_bagged(*data, task_type=TASK_TYPE, model_cls=model_cls, model_config=model_config)

    print(f"\n=== [3] tuned + ensembled {MODEL} ===")
    run_tuned(*data, task_type=TASK_TYPE, model_cls=model_cls)
