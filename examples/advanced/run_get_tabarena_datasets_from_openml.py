"""Minimal example to get the TabArena data and tasks without the TabArena framework.

To run this code, you only need to install `openml`.
    pip install openml
"""

from __future__ import annotations

import openml

# -- Parameters
tabarena_version = "tabarena-v0.1"
"""The version of the TabArena benchmark suite to use."""
tabarena_lite = False
"""If True, will use the TabArena-Lite version of the benchmark suite.
That is, only the first repeat of the first fold of each task will be used."""

# -- Get Data
benchmark_suite = openml.study.get_suite("tabarena-v0.1")
task_ids = benchmark_suite.tasks

# Iterate over all data and outer cross-validation splits from TabArena(-Lite)
print("Getting Data for TabArena tasks...")
if tabarena_lite:
    print("TabArena Lite is enabled. Getting first repeat of first fold for each task.")

for task_id in task_ids:
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    print(f"Task ID: {task.id}, Dataset ID: {dataset.id}, Dataset Name: {dataset.name}")

    # Get the number of folds and repeats used in TabArena
    if tabarena_lite:
        folds = 1
        tabarena_repeats = 1
    else:
        _, folds, _ = task.get_split_dimensions()
        n_samples = dataset.qualities["NumberOfInstances"]
        if n_samples < 2_500:
            tabarena_repeats = 10
        elif n_samples > 250_000:
            tabarena_repeats = 1
        else:
            tabarena_repeats = 3
    print(f"TabArena Repeats: {tabarena_repeats} | Folds: {folds}")

    # Load the data for each split
    for repeat in range(tabarena_repeats):
        for fold in range(folds):
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=task.target_name,
                dataset_format="dataframe",
            )
            train_indices, test_indices = task.get_train_test_split_indices(fold=fold, repeat=repeat)
            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_test = y.iloc[test_indices]

            # Train your model/system here :)
