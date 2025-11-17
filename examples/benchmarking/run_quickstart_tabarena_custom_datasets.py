from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from tabarena.benchmark.experiment import run_experiments_new
from tabarena.benchmark.task import UserTask
from tabarena.models.utils import get_configs_generator_from_name
from tabarena.nips2025_utils.compare import compare
from tabarena.nips2025_utils.end_to_end import EndToEnd
from bencheval.website_format import format_leaderboard


def get_custom_classification_task(task_cache_dir: str) -> UserTask:
    """Example for defining a classification task/dataset."""
    # Create toy classification dataset
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y)
    # Add cat features
    cats_1 = ["a"] * 25 + ["b"] * 25 + ["c"] * 25 + ["d"] * 25
    cats_2 = ["x"] * 34 + ["y"] * 33 + ["z"] * 33
    # Add nan values
    cats_1[0] = np.nan
    cats_1[49] = np.nan
    X.iloc[0, 2] = np.nan
    X.iloc[0, 3] = np.nan
    X = X.assign(cat_1=pd.Categorical(cats_1), cat_2=pd.Categorical(cats_2))
    dataset = pd.concat([X, y.rename("target")], axis=1)

    # Create a stratified 10-repeated 3-fold split (any other split can be used as well)
    n_repeats, n_splits = 10, 3
    sklearn_splits = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=42).split(
        X=dataset.drop(columns=["target"]), y=dataset["target"]
    )
    # Transform the splits into a standard dictionary format expected by TabArena
    splits = {}
    for split_i, (train_idx, test_idx) in enumerate(sklearn_splits):
        repeat_i = split_i // n_splits
        fold_i = split_i % n_splits
        if repeat_i not in splits:
            splits[repeat_i] = {}
        splits[repeat_i][fold_i] = (train_idx.tolist(), test_idx.tolist())

    # Create the UserTask for TabArena
    user_task = UserTask(
        task_name="ToyClf",
        task_cache_path=Path(task_cache_dir),
    )
    oml_task = user_task.create_local_openml_task(
        dataset=dataset,
        target_feature="target",
        problem_type="classification",
        splits=splits,
    )
    user_task.save_local_openml_task(oml_task)
    return user_task


def get_custom_regression_task(task_cache_dir: str) -> UserTask:
    """Example for defining a custom regression task/dataset."""
    X, y = make_regression(
        n_samples=100,
        n_features=20,
        n_informative=10,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y)
    # Add cat features
    cats_1 = ["a"] * 25 + ["b"] * 25 + ["c"] * 25 + ["d"] * 25
    cats_2 = ["x"] * 34 + ["y"] * 33 + ["z"] * 33
    # Add nan values
    cats_1[0] = np.nan
    cats_1[49] = np.nan
    X.iloc[0, 2] = np.nan
    X.iloc[0, 3] = np.nan
    X = X.assign(cat_1=pd.Categorical(cats_1), cat_2=pd.Categorical(cats_2))
    dataset = pd.concat([X, y.rename("target")], axis=1)

    # Create a holdout split without repeats
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.33, random_state=42, shuffle=True)
    # Transform the splits into a standard dictionary format expected by TabArena
    splits = {0: {0: (train_idx, test_idx)}}

    # Create the UserTask for TabArena
    user_task = UserTask(
        task_name="ToyReg",
        task_cache_path=Path(task_cache_dir),
    )
    oml_task = user_task.create_local_openml_task(
        dataset=dataset,
        target_feature="target",
        problem_type="regression",
        splits=splits,
    )
    user_task.save_local_openml_task(oml_task)
    return user_task


if __name__ == "__main__":
    # locations to experiment artifacts
    tabarena_dir = str(Path(__file__).parent / "experiments" / "quickstart_custom_dataset")
    eval_dir = Path(__file__).parent / "eval" / "quickstart_custom_dataset"
    task_cache_dir = str(Path(__file__).parent / "task_cache" / "quickstart_custom_dataset")

    # Get custom dataset (see below for how to write your own function)
    tasks = [
        get_custom_classification_task(task_cache_dir=task_cache_dir),
        get_custom_regression_task(task_cache_dir=task_cache_dir),
    ]
    # Build metadata for custom tasks
    task_metadata = []
    for task in tasks:
        oml_task = task.load_local_openml_task()
        task_metadata.append(
            [
                task.task_id,
                task.tabarena_task_name,
                oml_task.task_type,
                task.tabarena_task_name,
                int(len(oml_task.get_dataset().get_data()) * 0.67),
                int(len(oml_task.get_dataset().get_data()) * 0.33),
            ]
        )
    task_metadata = pd.DataFrame(
        task_metadata,
        columns=["tid", "name", "task_type", "dataset", "n_samples_train_per_fold", "n_samples_test_per_fold"],
    )

    # This list of some methods we want fit sequentially on each task (dataset x fold)
    # Checkout the available models in tabarena.benchmark.models.utils.get_configs_generator_from_name
    model_names = [
        "LightGBM",
        "RandomForest",
        "KNN",
        "Linear",
    ]
    # Number of random search configs
    num_random_configs = 1

    model_experiments = []
    for model_name in model_names:
        config_generator = get_configs_generator_from_name(model_name)
        model_experiments.extend(
            config_generator.generate_all_bag_experiments(
                num_random_configs=num_random_configs,
                fold_fitting_strategy="sequential_local",
            )
        )

    results_lst = run_experiments_new(
        output_dir=tabarena_dir,
        model_experiments=model_experiments,
        tasks=tasks,
        repetitions_mode="TabArena-Lite",
    )

    # compute results
    end_to_end = EndToEnd.from_raw(
        results_lst=results_lst,
        task_metadata=task_metadata,
        cache=False,
        cache_raw=False,
    )
    end_to_end_results = end_to_end.to_results()
    df_results = end_to_end_results.get_results()

    leaderboard: pd.DataFrame = compare(
        df_results=df_results,
        output_dir=eval_dir,
        task_metadata=task_metadata,
        fillna="RF (default)",
        calibration_framework="RF (default)",
    )
    leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
    print(leaderboard_website.to_markdown(index=False))


