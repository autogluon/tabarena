import argparse
import time

import numpy as np
import openml
import pandas as pd
from autogluon.core.data import LabelCleaner
from sklearn.metrics import confusion_matrix
from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import ProxyModelConfig
from tabarena.benchmark.feature_selection_methods.feature_selection_methods_register import (
    FEATURE_SELECTION_METHODS,
    get_feature_selector_from_name,
)


def parse_args():  # noqa: D103
    parser = argparse.ArgumentParser(description="FS Benchmark Runner")
    parser.add_argument("--method_name", type=str, default="MIFeatureSelector",
                        help="Feature Selection Method name [default: 'MIFeatureSelector']")
    parser.add_argument("--dataset", type=int, default=55,
                        help="OpenML dataset identifier [default: 55]")
    parser.add_argument("--problem_type", type=str, default="binary",
                        help="OpenML dataset problem type [default: 'binary']")
    parser.add_argument("--noise", type=float, default=1.0, nargs="+",
                        help="Percentage of noise features relative to original feature count [default: 1.0]")
    parser.add_argument("--max-features", type=int, default=5, nargs="+",
                        help="Max feature(s) to select [default: 5]")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Number of bootstrap repeats [default: 10]")
    return parser.parse_args()


def validity_fs(args):  # noqa: D103
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

    # Stability metrics
    validity = getValidity(Z, max_features)
    ci = confidenceIntervals(validity)
    validity_results = {
        "method": method_name,
        "dataset": dataset_id,
        "problem_type": problem_type,
        "max_features": max_features,
        "original_features": len(X.columns),
        "noise_features": n_noise,
        "repeats": n_repeats,
        "elapsed_time_fs": times,
        "n_samples": n_samples,
        "confusion_matrices": Z,
        "validity": validity,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
    }
    print(f"Validity: {validity_results}")
    return validity_results


def getValidity(Z, max_features):
    # Normalized recall
    selection_precision = []
    for elem in Z:
        _tn, _fp, _fn, tp = elem.ravel()
        precision = tp / max_features
        selection_precision.append(precision)
    return selection_precision


def confidenceIntervals(validity):
    """Confidence intervals for stability."""
    boot_validity = validity[1:]

    lower = np.percentile(boot_validity, 2.5)
    upper = np.percentile(boot_validity, 97.5)

    # z_score = norm.ppf(1 - alpha / 2)
    # variance = np.var(validity, ddof=1) / len(validity)
    # margin = z_score * math.sqrt(variance)
    return [lower, upper]


def add_noise(X, n_noise, noise_type="gaussian"):  # noqa: D103
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
    for method_name in FEATURE_SELECTION_METHODS:
        args.method_name = method_name
        validity_results = validity_fs(args)
