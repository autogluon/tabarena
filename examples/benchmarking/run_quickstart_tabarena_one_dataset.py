from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from tabarena.benchmark.experiment import ExperimentBatchRunner
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.models.utils import get_configs_generator_from_name
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.fetch_metadata import load_task_metadata

if __name__ == "__main__":
    # locations to experiment artifacts
    tabarena_dir = str(Path(__file__).parent / "experiments" / "one_dataset")

    # Get tasks and metadata (could also use custom dataset / metadata here as shown in other examples).
    # `split_indices="lite"` keeps the first split (r0f0) of each dataset; subset to one OpenML task id.
    task_metadata = TaskMetadataCollection.from_preset("TabArena-v0.1").subset_tasks(
        task_ids=[363621], split_indices="lite"
    )
    metadata = load_task_metadata()

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
            ),
        )

    runner = ExperimentBatchRunner(expname=tabarena_dir, task_metadata=task_metadata)
    results_lst = runner.run_all(methods=model_experiments)

    # compute results
    end_to_end = EndToEnd.from_raw(
        results_lst=results_lst,
        task_metadata=TaskMetadataCollection.from_legacy_df(metadata),
        cache=False,
        cache_raw=False,
    )
    end_to_end_results = end_to_end.to_results()
    df_results = end_to_end_results.get_results()

    # TODO: support this with TabArena code and make it look better
    # Generate plots for oen dataset
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)

    df_results["ROC AUC"] = 1 - df_results["metric_error"]
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_results,
        x="ROC AUC",
        y="method",
    )
    plt.xlabel("ROC AUC")
    plt.ylabel("Framework")
    plt.xlim(0.5)
    plt.tight_layout()
    plt.show()

    df_results["Val ROC AUC"] = 1 - df_results["metric_error_val"]
    sns.barplot(
        data=df_results,
        x="Val ROC AUC",
        y="method",
    )
    plt.xlabel("Validation ROC AUC")
    plt.ylabel("Framework")
    plt.xlim(0.5)
    plt.tight_layout()
    plt.show()
