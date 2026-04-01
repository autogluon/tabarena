import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from autogluon.core.data import LabelCleaner
from sklearn.metrics import confusion_matrix
from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import ProxyModelConfig
from tabarena.benchmark.feature_selection_methods.feature_selection_benchmark_utils import (
    get_fs_benchmark_preprocessing_pipelines,
)
from tabarena.benchmark.feature_selection_methods.feature_selection_methods_register import (
    FEATURE_SELECTION_METHODS,
    get_feature_selector_from_name,
)
from tabflow_slurm.benchmarking_setup.data_foundry_integration.data_foundry_task_creator import (
    download_data_foundry_datasets,
)


def parse_args():  # noqa: D103
    parser = argparse.ArgumentParser(description="FS Benchmark Runner")
    parser.add_argument("--method_name", type=str, default="FSBench__AccuracyFeatureSelector__5__0__lgbm__3600",
                    help="Feature Selection Method name [default: FSBench__AccuracyFeatureSelector__5__0__lgbm__3600]")
    parser.add_argument("--dataset", type=str, default="anneal/019d3f7b-494a-71fa-8eb2-25d01dfb7792",
                        help="OpenML dataset identifier [default: anneal/019d3f7b-494a-71fa-8eb2-25d01dfb7792]")
    parser.add_argument("--problem_type", type=str, default="binary",
                        help="OpenML dataset problem type [default: 'binary']")
    parser.add_argument("--noise", type=float, default=1.0, nargs="+",
                        help="Percentage of noise features relative to original feature count [default: 1.0]")
    parser.add_argument("--max-features", type=int, default=5, nargs="+",
                        help="Max feature(s) to select [default: 5]")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Number of bootstrap repeats [default: 10]")
    return parser.parse_args()

@dataclass
class ValidityResult:
    """Result object containing feature selection validity metrics from multiple repeats.

    Attributes:
    ----------
    method : str
        Name of the feature selection method evaluated.
    dataset : int
        OpenML dataset ID used for evaluation.
    problem_type : str
        ML problem type ('binary_classification', 'multiclass_classification', 'regression').
    max_features : int
        Maximum number of features requested by the selector.
    original_features : int
        Number of true (non-noise) features in the dataset.
    noise_features : int
        Number of synthetic noise features added for evaluation.
    repeats : int
        Number of bootstrap repeats performed.
    elapsed_time_fs : list[float]
        List of runtime measurements (seconds) for each repeat.
    n_samples : list[int]
        List of sample sizes used in each repeat.
    confusion_matrices : list[list[int]]
        List of 2x2 confusion matrices [TN, FP, FN, TP] for each repeat.
    validity : list[float]
        List of validity scores computed for each repeat.
    ci_lower : float
        Lower bound of the confidence interval for mean validity.
    ci_upper : float
        Upper bound of the confidence interval for mean validity.
    """
    method: str
    dataset: int
    problem_type: str
    max_features: int
    original_features: int
    noise_features: int
    repeats: int
    elapsed_time_fs: [float]
    n_samples: [int]
    confusion_matrices: [[int]]
    validity: [float]
    ci_lower: float
    ci_upper: float

def validity_fs(args) -> ValidityResult:
    """Evaluate feature selection method validity through confusion matrix analysis.

    This function assesses how well a feature selection method distinguishes original
    features from noisy synthetic features across multiple bootstrap repeats. It adds
    noise features, applies the specified feature selector, and computes validity
    metrics based on the confusion matrix between true original features and selected
    features.

    Parameters
    ----------
    args : Namespace
        Argument object containing:
        - dataset : int
            OpenML dataset ID.
        - problem_type : str
            Problem type ('binary_classification', 'multiclass_classification', etc.).
        - repeats : int
            Number of bootstrap repeats for stability assessment.
        - max_features : int
            Maximum number of features to select.
        - method_name : str
            Name of the feature selection method to evaluate.
        - noise : float
            Proportion of noise features to add relative to original features.

    Returns:
    -------
    ValidityResult
        Object containing validity metrics, confusion matrices, timing information,
        and confidence intervals for the feature selection method's performance.
    """
    dataset_id = args.dataset
    problem_type = args.problem_type
    n_repeats = args.repeats
    max_features = args.max_features
    method_name = args.method_name
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
    y = label_cleaner.transform(y)

    n_noise = int(len(X.columns) * args.noise)
    print(f"\n=== {method_name} ===")
    print("Max features: ", max_features)
    # Store binary selections across repeats
    Z = []
    times = []
    n_samples = []
    for repeat in range(n_repeats):
        # add noise features
        X_copy, orig_feature_mask = add_noise(X, n_noise)
        if repeat != 0:
            # Resample data (bootstrap) for variability
            sample_idx = np.random.choice(len(X_copy), size=len(X_copy), replace=True)
            X_repeat = X_copy.iloc[sample_idx].reset_index(drop=True)
            y_repeat = y.iloc[sample_idx].reset_index(drop=True)
        else:
            X_repeat = X_copy
            y_repeat = y
        n_samples.append(len(X_repeat))
        proxy_config = ProxyModelConfig(
            problem_type=problem_type,
            eval_metric="roc_auc",
            model_hyperparameters={"num_boost_round": 1},
        )
        start_time = time.monotonic()
        feature_selector = get_feature_selector_from_name(name=method_name)
        feature_selector = feature_selector(max_features=max_features, proxy_mode_config=proxy_config)
        selected_features = feature_selector.fit_transform(X=X_repeat, y=y_repeat)
        elapsed_time = time.monotonic() - start_time
        times.append(elapsed_time)
        # Binary row: 1 if selected, 0 otherwise
        selected_binary = np.zeros(len(X_copy.columns), dtype=bool)
        selected_indices = [X_copy.columns.get_loc(f) for f in selected_features]
        selected_binary[selected_indices] = True
        orig_binary = np.array([orig_feature_mask[col] for col in X_copy.columns])

        cm = confusion_matrix(orig_binary, selected_binary)
        tn, fp, fn, tp = cm.ravel()
        print(f"TP: {tp}  FN: {fn}  FP: {fp}  TN: {tn}")

        Z.append(cm)

    # Validity metrics
    validity = getValidity(Z, max_features)
    ci = confidenceIntervals(validity)
    validity_results = ValidityResult(
        method=method_name,
        dataset=dataset_id,
        problem_type=problem_type,
        max_features=max_features,
        original_features=len(X.columns),
        noise_features=n_noise,
        repeats=n_repeats,
        elapsed_time_fs=times,
        n_samples=n_samples,
        confusion_matrices=Z,
        validity=validity,
        ci_lower=ci[0],
        ci_upper=ci[1],
    )
    print(f"Validity: {validity_results.validity}")
    return validity_results


def getValidity(Z, max_features) -> list[float]:
    """Compute normalized recall (validity) for feature selection.

    Parameters
    ----------
    Z : list of array-like, shape (n_repeats, 4)
        List of 2x2 confusion matrices [TN, FP, FN, TP] from validity_fs where
        true positives (TP) represent original features correctly identified.
    max_features : int
        Maximum number of features requested by the selector.

    Returns:
    -------
    list[float]
        List of normalized recall scores (TP / max_features) for each repeat,
        measuring proportion of selected features that are true originals.
    """
    selection_precision = []
    for elem in Z:
        _tn, _fp, _fn, tp = elem.ravel()
        precision = tp / max_features
        selection_precision.append(precision)
    return selection_precision


def confidenceIntervals(validity) -> list[float]:
    """Compute bootstrap confidence interval for validity scores.

    Parameters
    ----------
    validity : list[float]
        List of validity scores from multiple repeats, where validity[0] is typically
        the first repeat and validity[1:] contains bootstrap repeat scores.

    Returns:
    -------
    list[float]
        95% confidence interval [lower, upper] using 2.5th and 97.5th percentiles
        of bootstrap validity scores (validity[1:]).
    """
    boot_validity = validity[1:]

    lower = np.percentile(boot_validity, 2.5)
    upper = np.percentile(boot_validity, 97.5)
    return [lower, upper]


def add_noise(X, n_noise, noise_type="gaussian") -> tuple[pd.DataFrame, dict[str, bool]]:
    """Add noisy synthetic features to a dataset and shuffle all features.

    Parameters
    ----------
    X : DataFrame
        The input feature matrix.
    n_noise : int
        The number of noisy features to add.
    noise_type : str, optional
        The type of noise to generate for numeric features. Use "gaussian" for
        normally distributed noise or any other value for uniform noise, by default "gaussian".

    Returns:
    -------
    tuple[DataFrame, dict[str, bool]]
        A tuple containing the augmented and shuffled feature matrix and a mask
        indicating which shuffled features correspond to original features.
    """
    noise_cols = {}
    n_samples, n_features = X.shape

    all_feature_names = [f"feature{i}" for i in range(n_features + n_noise)]
    orig_feature_mask = {name: i < n_features for i, name in enumerate(all_feature_names)}

    X = X.rename(columns={old: f"feature{i}" for i, old in enumerate(X.columns)})

    for i in range(n_noise):
        col_idx = np.random.randint(0, n_features)  # noqa: NPY002
        sample_col = X.iloc[:, col_idx]

        if sample_col.dtype.kind in "biufc":
            if noise_type == "gaussian":
                noise = np.random.normal(sample_col.mean(), sample_col.std(), n_samples)  # noqa: NPY002
            else:
                noise = np.random.uniform(sample_col.min(), sample_col.max(), n_samples)  # noqa: NPY002
        elif sample_col.dtype.kind == "O":
            unique_vals = sample_col.dropna().unique()
            if len(unique_vals) > 1:
                probs = sample_col.value_counts(normalize=True).values  # noqa: PD011
                noise = np.random.choice(unique_vals, n_samples, p=probs)
            else:
                noise = np.full(n_samples, unique_vals[0])
        else:
            noise = np.random.normal(0, 1, n_samples)

        noise_cols[f"feature{n_features + i}"] = noise

    noise_df = pd.DataFrame(noise_cols)
    X_final = pd.concat([X, noise_df], axis=1)

    shuffle_order = np.random.permutation(len(all_feature_names))
    X_final_shuffled = X_final.iloc[:, shuffle_order]
    # Update mask to match new positions!
    orig_feature_mask_shuffled = {all_feature_names[i]: orig_feature_mask[all_feature_names[shuffle_order[i]]]
                                  for i in range(len(all_feature_names))}
    return X_final_shuffled, orig_feature_mask_shuffled


if __name__ == "__main__":
    args = parse_args()

    DEFAULT_DATA_FOUNDRY_CACHE = Path(__file__).parent / ".data_foundry_cache"

    EXAMPLE_DATA_FOUNDRY_TASKS = [
        "anneal/019d3f7b-494a-71fa-8eb2-25d01dfb7792",
        "ancestry_study/019d3f8b-5610-71fa-9135-a7642f26294b",
    ]

    datasets = download_data_foundry_datasets(
        benchmark_suite_name="feature_selection_benchmark_validity_examples",
        data_foundry_artifacts=EXAMPLE_DATA_FOUNDRY_TASKS,
        data_foundry_cache=DEFAULT_DATA_FOUNDRY_CACHE,
    ).get_metadata_for_benchmark_suite()

    preprocessing_pipeline = get_fs_benchmark_preprocessing_pipelines(
        fs_methods=FEATURE_SELECTION_METHODS,
        proxy_model_config=["lgbm"],
        time_limit=[3600],
        total_budget=5,
        include_default=True,
    )

    for method_name in preprocessing_pipeline:
        args.method_name = method_name
        for dataset in EXAMPLE_DATA_FOUNDRY_TASKS:
            args.dataset = dataset
            validity_results = validity_fs(args)
