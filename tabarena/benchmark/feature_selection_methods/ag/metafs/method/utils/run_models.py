import pandas as pd
import logging
import os
import ray
import tempfile

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import log_loss, root_mean_squared_error, max_error, roc_auc_score
from autogluon.tabular import TabularPredictor
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from ....metafs.method.utils.Autogluon_MultilabelPredictor import MultilabelPredictor
from ....metafs.method.utils.preprocess_data import factorize_features, factorize_target
from ....metafs.method.utils.tabrepo_2024_custom import zeroshot2024


def run_default_lgbm(X_train, y_train, X_test, y_test):
    clf = lgb.LGBMClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_test)
    print(y_pred)
    df_train = pd.concat([X_train, pd.Series(y_train)], axis=1)
    df_test = pd.concat([X_test, pd.Series(y_test)], axis=1)
    df_original = pd.concat([df_train, df_test], axis=0)
    labels = df_original.iloc[:, -1].unique()
    log_loss_test = log_loss(y_test, y_pred, labels=labels)
    print(log_loss_test)
    return log_loss_test


def run_autogluon_lgbm(X_train, y_train, X_test, y_test, zeroshot=False):
    label = "target"
    train_data = X_train
    train_data[label] = y_train
    test_data = X_test
    test_data[label] = y_test

    allowed_models = ["GBM"]  # , "RF", "KNN", "XT", "CAT", "XGB", "LR", "FASTAI", "AG_AUTOMM", "NN_TORCH"]

    zeroshot2024 = get_zeroshot_models(allowed_models, zeroshot)
    # -- Run AutoGluon
    predictor = init_and_fit_predictor(label, train_data, zeroshot2024)
    lb = predictor.leaderboard(test_data)
    return lb

def run_autogluon_lgbm_classification(X_train, y_train, X_test, y_test, zeroshot=False):
    label = "target"
    train_data = X_train
    train_data[label] = y_train
    test_data = X_test
    test_data[label] = y_test

    allowed_models = ["GBM"]  # , "RF", "KNN", "XT", "CAT", "XGB", "LR", "FASTAI", "AG_AUTOMM", "NN_TORCH"]

    zeroshot2024 = get_zeroshot_models(allowed_models, zeroshot)
    # -- Run AutoGluon
    predictor = init_and_fit_improvement_predictor_classification(label, train_data, zeroshot2024)
    lb = predictor.leaderboard(test_data)
    return lb


def run_autogluon_lgbm_regression(X_train, y_train, X_test, y_test, zeroshot=False):
    label = "target"
    train_data = X_train
    train_data[label] = y_train
    test_data = X_test
    test_data[label] = y_test

    allowed_models = ["GBM"]  # , "RF", "KNN", "XT", "CAT", "XGB", "LR", "FASTAI", "AG_AUTOMM", "NN_TORCH"]

    zeroshot2024 = get_zeroshot_models(allowed_models, zeroshot)
    # -- Run AutoGluon
    predictor = init_and_fit_improvement_predictor_regression(label, train_data, zeroshot2024)
    lb = predictor.leaderboard(test_data)
    return lb


def run_autogluon_lgbm_ray(X_train, y_train, X_test, y_test, zeroshot=False):
    label = "target"
    train_data = X_train
    train_data[label] = y_train
    test_data = X_test
    test_data[label] = y_test
    log = logging.getLogger(__name__)
    ray_mem_in_gb = 48
    log.info(f"Running on SLURM, initializing Ray with unique temp dir with {ray_mem_in_gb}GB.")
    ray_mem_in_b = int(ray_mem_in_gb * (1024.0 ** 3))
    tmp_dir_base_path = "/tmp"
    ray_dir = f"{tmp_dir_base_path}"
    print(f"Start local ray instances. Using {os.environ.get('RAY_MEM_IN_GB')} GB for Ray.")
    ray.shutdown()
    ray.init(
        address="local",
        _memory=ray_mem_in_b,
        object_store_memory=int(ray_mem_in_b * 0.3),
        _temp_dir=ray_dir,
        include_dashboard=False,
        logging_level=logging.INFO,
        log_to_driver=True,
        num_gpus=0,
        num_cpus=8,
        ignore_reinit_error=True
    )

    allowed_models = [
        "GBM",
    ]

    zeroshot2024 = get_zeroshot_models(allowed_models, zeroshot)

    # -- Run AutoGluon
    predictor = init_and_fit_predictor(label, train_data, zeroshot2024)
    lb = predictor.leaderboard(test_data, display=True)
    ray.shutdown()
    return lb


def predict_autogluon_lgbm(train_data, test_data):
    # Prepare Data
    # X_test = add_new_featurenames(X_test)
    label = 'improvement'

    # Predictor
    predictor = init_and_fit_predictor("improvement", train_data, zeroshot2024)
    # Evaluation
    # evaluation = pd.DataFrame(predictor.evaluate(X_test, ))
    # Prediction
    prediction = predictor.predict(test_data)
    prediction.rename("predicted_improvement", inplace=True)
    prediction_result = pd.concat([test_data[["dataset - id", "feature - name", "model"]], prediction], axis=1)
    return prediction_result  # evaluation,


def multi_predict_autogluon_lgbm(train_data, X_test):
    labels = ['feature - name', 'improvement']  # which columns to predict based on the others
    problem_types = ['multiclass', 'regression']  # type of each prediction problem
    save_path = 'agModels'  # specifies folder to store trained models
    time_limit = 300  # how many seconds to train the TabularPredictor for each label, set much larger in your applications!
    # Multi Predictor
    multi_predictor = MultilabelPredictor(labels=labels, problem_types=problem_types, path=save_path)
    multi_predictor.fit(train_data, labels, time_limit=time_limit)
    # Evaluation
    #  multi_evaluation = pd.DataFrame(multi_predictor.evaluate(X_test))
    # Prediction
    multi_prediction = pd.DataFrame(multi_predictor.predict(X_test))
    multi_prediction.rename(columns={"feature - name": "new - feature - name", "improvement": "predicted_improvement"},
                            inplace=True)
    multi_prediction_result = pd.concat([X_test[["dataset - id", "feature - name", "model"]], multi_prediction], axis=1)
    return multi_prediction_result  # multi_evaluation,


def get_model_score(X_train, y_train, X_test, y_test, dataset_id):
    lb = run_autogluon_lgbm(X_train, y_train, X_test, y_test)
    models = lb["model"]
    new_results = pd.DataFrame(columns=['dataset', 'model', 'score'])
    for model in models:
        score_val = lb.loc[lb['model'] == model, 'score_val'].values[0]
        new_results.loc[len(new_results)] = [dataset_id, model, score_val]
    return new_results


def get_model_score_origin_classification(X_train, y_train, X_test, y_test, dataset_id, origin):
    lb = run_autogluon_lgbm_classification(X_train, y_train, X_test, y_test)
    models = lb["model"]
    new_results = pd.DataFrame(columns=['origin', 'task_type', 'dataset', 'model', 'score_val', 'score_test'])
    for model in models:
        score_val = lb.loc[lb['model'] == model, 'score_val'].values[0]
        score_test = lb.loc[lb['model'] == model, 'score_test'].values[0]
        new_results.loc[len(new_results)] = [origin, "Classification", dataset_id, model, score_val, score_test]
    return new_results


def get_model_score_origin_regression(X_train, y_train, X_test, y_test, dataset_id, origin):
    lb = run_autogluon_lgbm_regression(X_train, y_train, X_test, y_test)
    models = lb["model"]
    new_results = pd.DataFrame(columns=['origin', 'task_type', 'dataset', 'model', 'score_val', 'score_test'])
    for model in models:
        score_val = lb.loc[lb['model'] == model, 'score_val'].values[0]
        score_test = lb.loc[lb['model'] == model, 'score_test'].values[0]
        new_results.loc[len(new_results)] = [origin, "Regression", dataset_id, model, score_val, score_test]
    return new_results


def get_model_score_regression(X_train, y_train, X_test, y_test, dataset_id):
    lb = run_autogluon_lgbm_regression(X_train, y_train, X_test, y_test)
    models = lb["model"]
    new_results = pd.DataFrame(columns=['dataset', 'model', 'score'])
    for model in models:
        score_val = lb.loc[lb['model'] == model, 'score_val'].values[0]
        new_results.loc[len(new_results)] = [dataset_id, model, score_val]
    return new_results


def init_and_fit_predictor(label, train_data, zeroshot2024):
    # try:
    #    predictor = TabularPredictor.load("/tmp/my_predictor")
    #    print("Predictor read")
    #    return predictor
    #except FileNotFoundError:
    predictor = TabularPredictor(
        label=label,
        eval_metric="log_loss",  # roc_auc (binary), log_loss (multiclass) root_mean_squared_error (regression)
        problem_type="multiclass",  # binary, multiclass, regression
        verbosity=0,
    )
    predictor.fit(
        time_limit=int(60 * 60* 10),
        memory_limit=48,
        num_cpus=8,
        num_gpus=0,
        train_data=train_data,
        presets="high_quality",
        dynamic_stacking=False,
        hyperparameters=zeroshot2024,
        num_bag_folds=8,
        num_bag_sets=1,
        num_stack_levels=0,
        fit_weighted_ensemble=False
    )
    return predictor


def init_and_fit_improvement_predictor_classification(label, train_data, zeroshot2024):
    predictor = TabularPredictor(
        label=label,
        eval_metric="log_loss",  # roc_auc (binary), log_loss (multiclass)
        problem_type="multiclass",  # binary, multiclass
        verbosity=0,
        path=tempfile.mkdtemp() + os.sep,
    )
    predictor.fit(
        time_limit=int(60 * 60 * 4),
        memory_limit=48,
        num_cpus=8,
        num_gpus=0,
        train_data=train_data,
        presets="best_quality",
        dynamic_stacking=False,
        hyperparameters=zeroshot2024,
        num_bag_folds=8,
        num_bag_sets=1,
        num_stack_levels=0,
        fit_weighted_ensemble=False
    )
    return predictor


def init_and_fit_improvement_predictor_regression(label, train_data, zeroshot2024):
    predictor = TabularPredictor(
        label=label,
        eval_metric="root_mean_squared_error",  # roc_auc (binary), log_loss (multiclass)
        problem_type="regression",  # binary, multiclass
        verbosity=0,
        path=tempfile.mkdtemp() + os.sep,
    )
    predictor.fit(
        time_limit=int(60 * 60 * 4),
        memory_limit=48,
        num_cpus=8,
        num_gpus=0,
        train_data=train_data,
        presets="best_quality",
        dynamic_stacking=False,
        hyperparameters=zeroshot2024,
        num_bag_folds=8,
        num_bag_sets=1,
        num_stack_levels=0,
        fit_weighted_ensemble=False
    )
    return predictor


def get_zeroshot_models(allowed_models, zeroshot):
    for k in list(zeroshot2024.keys()):
        if k not in allowed_models:
            del zeroshot2024[k]
        else:
            if not zeroshot:
                zeroshot2024[k] = zeroshot2024[k][:1]
            else:
                zeroshot2024[k] = zeroshot2024[k][1:]
    return zeroshot2024


def get_sklearn_model_score_classification(X_train, y_train, X_test, y_test, dataset_id, origin, model_name, seed, score_names):
    model = get_sklearn_model_classification(model_name, seed)
    X_train, X_test = factorize_features(X_train, X_test)
    y_train, y_test = factorize_target(y_train, y_test)
    model.fit(X_train, y_train)
    score_val = model.score(X_train, y_train)
    new_results = pd.DataFrame(columns=['origin', 'task_type', 'dataset', 'model', 'seed', 'score_name', 'score_val', 'score_test'])
    for score_name in score_names:
        score_test = get_sklearn_classification_scores(score_name, y_test, model, X_test)
        new_row = pd.DataFrame(columns=['origin', 'task_type', 'dataset', 'model', 'seed', 'score_name', 'score_val', 'score_test'])
        new_row.loc[len(new_results)] = [origin, "Classification", dataset_id, model_name, seed, score_name, score_val, score_test]
        new_results = pd.concat([new_results, new_row])
    return new_results


def get_sklearn_model_classification(model_name, seed):
    if model_name == "HistGradientBoosting":
        model = HistGradientBoostingClassifier(random_state=seed)
    elif model_name == "RandomForest":
        model = RandomForestClassifier(random_state=seed)
    elif model_name == "MLP":
        model = MLPClassifier(random_state=seed)
    elif model_name == "SVM":
        model = SVC(random_state=seed)
    elif model_name == "GaussianNB":
        model = GaussianNB()
    elif model_name == "KNeighbors":
        model = KNeighborsClassifier()
    else:
        model = HistGradientBoostingClassifier(random_state=seed)
    return model


def get_sklearn_classification_scores(score_name, y_test, model, X_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    if score_name == "log_loss":
        score = log_loss(y_test, y_pred_proba)
    elif score_name == "roc_auc_score":
        try:
            score = roc_auc_score(y_test, y_pred)
        except ValueError:
            score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    else:
        score = log_loss(y_test, y_pred_proba)
    return score

def get_sklearn_model_score_regression(X_train, y_train, X_test, y_test, dataset_id, origin, model_name, seed, score_names):
    model = get_sklearn_model_regression(model_name, seed)
    X_train, X_test = factorize_features(X_train, X_test)
    y_train, y_test = factorize_target(y_train, y_test)
    model.fit(X_train, y_train)
    score_val = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    new_results = pd.DataFrame(columns=['origin', 'task_type', 'dataset', 'model', 'seed', 'score_name', 'score_val', 'score_test'])
    for score_name in score_names:
        score_test = get_sklearn_regression_scores(score_name, y_test, y_pred)
        new_row = pd.DataFrame(columns=['origin', 'task_type', 'dataset', 'model', 'seed', 'score_name', 'score_val', 'score_test'])
        new_row.loc[len(new_results)] = [origin, "Regression", dataset_id, model_name, seed, score_name, score_val, score_test]
        new_results = pd.concat([new_results, new_row])
    return new_results


def get_sklearn_model_regression(model_name, seed):
    if model_name == "HistGradientBoosting":
        model = HistGradientBoostingRegressor(random_state=seed)
    elif model_name == "RandomForest":
        model = RandomForestRegressor(random_state=seed)
    elif model_name == "MLP":
        model = MLPRegressor(random_state=seed)
    elif model_name == "SVM":
        model = SVR()
    elif model_name == "GaussianNB":
        model = GaussianNB()
    elif model_name == "KNeighbors":
        model = KNeighborsRegressor()
    else:
        model = HistGradientBoostingRegressor()
    return model


def get_sklearn_regression_scores(score_name, y_test, y_pred):
    if score_name == "root_mean_squared_error":
        score = root_mean_squared_error(y_test, y_pred)
    elif score_name == "max_error":
        score = max_error(y_test, y_pred)
    else:
        score = root_mean_squared_error(y_test, y_pred)
    return score
