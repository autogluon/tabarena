import math

import numpy as np
import openml
import pandas as pd
from autogluon.core.data import LabelCleaner
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import ProxyModelConfig
from tabarena.benchmark.feature_selection_methods.feature_selection_methods_register import (
    FEATURE_SELECTION_METHODS,
    get_feature_selector_from_name,
)


def validity_fs(method_names):  # noqa: D103
    dataset_id = 55
    problem_type = "binary"
    n_repeats = 5
    max_features = 5

    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
    y = label_cleaner.transform(y)

    validity_results = {}

    for method_name in method_names:
        print(f"\n=== {method_name} ===")
        print("Max features: ", max_features)
        # Store binary selections across repeats
        Z_repeats = []

        for _repeat in range(n_repeats):
            # add noise features
            X_copy, orig_feature_mask = add_noise(X)
            # TODO - One run without bootstrapping für den originalen Run, bootstrappin für CI
            # Resample data (bootstrap) for variability
            sample_idx = np.random.choice(len(X_copy), size=len(X_copy), replace=True)
            X_repeat = X_copy.iloc[sample_idx].reset_index(drop=True)
            y_repeat = y.iloc[sample_idx].reset_index(drop=True)

            proxy_config = ProxyModelConfig(
                problem_type=problem_type,
                eval_metric="roc_auc",
                model_hyperparameters={"num_boost_round": 1},
            )

            feature_selector = get_feature_selector_from_name(name=method_name)
            feature_selector = feature_selector(max_features=max_features, proxy_mode_config=proxy_config)
            selected_features = feature_selector.fit_transform(X=X_repeat, y=y_repeat)

            # Binary row: 1 if selected, 0 otherwise
            selected_binary = np.zeros(len(X_copy.columns), dtype=bool)
            selected_indices = [X_copy.columns.get_loc(f) for f in selected_features]
            selected_binary[selected_indices] = True
            orig_binary = np.array([orig_feature_mask[col] for col in X_copy.columns])

            cm = confusion_matrix(orig_binary, selected_binary)
            tn, fp, fn, tp = cm.ravel()
            print(f"TP: {tp}  FN: {fn}  FP: {fp}  TN: {tn}")

            Z_repeats.append(cm)
        Z = np.array(Z_repeats)

        # Stability metrics
        validity = getValidity(Z, len(X.columns), max_features)
        ci = stats.t.interval(0.95, len(validity) - 1, loc=np.mean(validity), scale=np.std(validity, ddof=1) / np.sqrt(len(validity)))
        validity_results[method_name] = {
            "validity": validity,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "Z": Z  # Save for further analysis
        }
        print(f"Validity: {validity_results[method_name]['validity']}")
        print(f"Validity CI: lower ({validity_results[method_name]['ci_lower']}) - higher ({validity_results[method_name]['ci_upper']})")  # noqa: E501
    return validity_results


def getValidity(Z, real_features, max_features):
    """Let us assume we have M>1 feature sets and d>0 features in total.
    This function computes the stability estimate as given in Definition 4 in  [1].

    INPUT: A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d).
           Each row of the binary matrix represents a feature set, where a 1 at the f^th position
           means the f^th feature has been selected and a 0 means it has not been selected.

    OUTPUT: The stability of the feature selection procedure
    """
    # TODO Parametrize max_feature including max_features = real_features
    # Normalized recall
    selection_precision = []
    for elem in Z:
        _tn, _fp, _fn, tp = elem.ravel()
        precision = tp / max_features
        selection_precision.append(precision)
    return selection_precision


def confidenceIntervals(validity, real_features, max_features, alpha=0.05, res=None):
    """Confidence intervals for stability (Corollary 9)."""
    # TODO Percentile based CI - lookup for bootstrapping
    if alpha >= 1 or alpha <= 0:
        raise ValueError("Alpha must be in (0,1)")

    z_score = norm.ppf(1 - alpha / 2)
    variance = np.var(validity, ddof=1) / len(validity)
    margin = z_score * math.sqrt(variance)
    return {"validity": res["validity"], "lower": validity - margin, "upper": validity + margin}

def add_noise(X, noise_type="gaussian"):  # noqa: D103
    noise_cols = {}
    n_samples, n_features = X.shape
    n_noise = int(n_features)  # TODO 50% / 75%, Anzahl der noise features variieren -> Parameter

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
    validity_results = validity_fs(FEATURE_SELECTION_METHODS)
