"""TabArena Processed-Artifact Quickstart.

This script demonstrates how to:
1) Discover available tabular ML **methods** integrated into TabArena.
2) Download **processed artifacts** (predictions/labels/metadata) for a chosen method.
3) Open an `EvaluationRepository` and inspect datasets, configs, metrics, and predictions.
4) Evaluate a simple **ensemble** across configurations.

It is designed as a readable "tour" rather than a benchmark runner.

Why "processed" artifacts?
--------------------------
Processed artifacts are ready-to-use evaluation assets (e.g., out-of-fold predictions,
test predictions, labels, and metadata). They allow you to:
- Inspect per-dataset/per-fold predictions without re-training.
- Compute metrics, compare configurations, and build ensembles instantly.
- Export AutoGluon-ready hyperparameters for a given config.

Typical Uses
------------
- Validate that an integrated method (e.g., LightGBM, Mitra_GPU, TabPFNv2, etc.) is
  properly wired into your local environment.
- Prototype ensembling strategies on top of cached predictions.
- Extract a method's **AutoGluon** hyperparameters and re-train that exact config on
  your own dataset with `TabularPredictor.fit(hyperparameters=...)`.

Data Volume Tips
----------------
- For a very small, fast download use: `method = "Mitra_GPU"`.
  This is great for connectivity checks and a quick end-to-end smoke test,
  but it has only a single config (so it won't showcase multi-config features).
- For a richer demonstration (multiple configs), try `method = "LightGBM"`.

Outputs at a Glance
-------------------
- Prints a markdown table of available methods.
- Shows dataset list, dataset info/metadata, and the first few configs.
- Displays metrics for a small slice of (datasets × configs).
- Prints head of predictions/labels for validation and test.
- Builds a simple top-N ensemble (size 100 by default) and prints its result.
- Reports the mean ensemble weight per config (top 10).

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tabarena.contexts.tabarena.methods import tabarena_method_metadata_collection

if TYPE_CHECKING:
    from tabarena import EvaluationRepository


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # 1) Surface available methods so users can pick a target method quickly.
    #    This prints a markdown table with identifying fields for each method.
    # ----------------------------------------------------------------------
    methods_info = tabarena_method_metadata_collection.info()
    print(methods_info.to_markdown())

    # ----------------------------------------------------------------------
    # 2) Choose a method to validate.
    #
    #    - "Mitra_GPU" -> tiny download, single config (fast smoke test).
    #    - "LightGBM"  -> larger download, multiple configs (better demo).
    # ----------------------------------------------------------------------
    method = "LightGBM"
    method_metadata = tabarena_method_metadata_collection.get_method_metadata(method=method)

    # ----------------------------------------------------------------------
    # 3) Ensure processed artifacts are available locally. If not, download.
    #    NOTE: Some methods may require large downloads (up to ~15 GB).
    # ----------------------------------------------------------------------
    if not method_metadata.path_processed_exists:
        print(
            f"Downloading processed data to {method_metadata.path_processed} ... "
            f"Ensure you have a fast internet connection. This download can be up to 15 GB.",
        )
        method_metadata.method_downloader().download_processed()

    # ----------------------------------------------------------------------
    # 4) Open the processed repository view for programmatic inspection.
    # ----------------------------------------------------------------------
    repo: EvaluationRepository = method_metadata.load_processed()
    repo.print_info()  # high-level repository summary

    if method_metadata.method_type != "config":
        raise AssertionError(
            f"This tutorial only supports config methods. "
            f"(method={method_metadata.method!r}, method_type={method_metadata.method_type!r})",
        )

    # ----------------------------------------------------------------------
    # 5) Explore datasets and per-dataset metadata.
    # ----------------------------------------------------------------------
    datasets = repo.datasets()
    print(f"Datasets: {datasets}")

    dataset = datasets[0]
    dataset_info = repo.dataset_info(dataset=dataset)
    print(f"Dataset Info    : {dataset_info}")

    dataset_metadata = repo.dataset_metadata(dataset=dataset)
    print(f"Dataset Metadata: {dataset_metadata}")

    # ----------------------------------------------------------------------
    # 6) Explore configs (individual model settings evaluated within a method).
    # ----------------------------------------------------------------------
    configs = repo.configs()
    print(f"Configs (first 10): {configs[:10]}")

    config_types = repo.config_types()
    assert len(config_types) == 1, (
        f"There should be exactly 1 config_type for method {method_metadata.method}: {config_types}"
    )
    repo_config_type = config_types[0]
    if repo_config_type != method_metadata.config_type:
        print(
            f"Warning: Misaligned processed config_type with method_metadata config_type!\n"
            f"\tmethod_metadata config_type: {method_metadata.config_type}\n"
            f"\t      processed config_type: {repo_config_type}\n",
        )

    config = configs[0]
    config_type = repo.config_type(config=config)
    config_hyperparameters = repo.config_hyperparameters(config=config)

    # You can pass the below autogluon_hyperparameters into AutoGluon's TabularPredictor.fit
    # to train this exact config on your own dataset:
    #
    # from autogluon.tabular import TabularPredictor
    # predictor = TabularPredictor(...).fit(..., hyperparameters=autogluon_hyperparameters)
    autogluon_hyperparameters = repo.autogluon_hyperparameters_dict(configs=[config])
    print(
        "Config Info:\n"
        f"\t                     Name: {config}\n"
        f"\t                     Type: {config_type}\n"
        f"\t          Hyperparameters: {config_hyperparameters}\n"
        f"\tAutoGluon Hyperparameters: {autogluon_hyperparameters}\n",
    )

    # ----------------------------------------------------------------------
    # 7) Inspect metrics for a small slice of (datasets × configs).
    # ----------------------------------------------------------------------
    metrics = repo.metrics(datasets=datasets[:2], configs=configs[:2])
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics}")

    # ----------------------------------------------------------------------
    # 8) Peek at predictions and ground-truth labels (validation and test).
    #    These are per-dataset, per-fold, per-config artifacts.
    # ----------------------------------------------------------------------
    predictions_test = repo.predict_test(dataset=dataset, fold=0, config=config)
    print(f"Predictions Test (config={config}, dataset={dataset}, fold=0):\n{predictions_test[:10]}")

    y_test = repo.labels_test(dataset=dataset, fold=0)
    print(f"Ground Truth Test (dataset={dataset}, fold=0):\n{y_test[:10]}")

    predictions_val = repo.predict_val(dataset=dataset, fold=0, config=config)
    print(f"Predictions Val (config={config}, dataset={dataset}, fold=0):\n{predictions_val[:10]}")

    y_val = repo.labels_val(dataset=dataset, fold=0)
    print(f"Ground Truth Val (dataset={dataset}, fold=0):\n{y_val[:10]}")

    # ----------------------------------------------------------------------
    # 9) Build a simple ensemble over many configs for the chosen dataset/fold.
    #    Returns (result_df, weights_df). Here we average weights across folds
    #    and show the highest-weighted configs.
    # ----------------------------------------------------------------------
    df_result, df_ensemble_weights = repo.evaluate_ensemble(
        dataset=dataset,
        fold=0,
        configs=configs,
        ensemble_size=100,
    )
    print(f"Ensemble result:\n{df_result}")

    df_ensemble_weights_mean_sorted = df_ensemble_weights.mean(axis=0).sort_values(ascending=False)
    print(f"Top 10 highest mean ensemble weight configs:\n{df_ensemble_weights_mean_sorted.head(10)}")
