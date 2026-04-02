import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import ProxyModelConfig
from tabarena.benchmark.feature_selection_methods.feature_selection_benchmark_utils import (
    get_fs_benchmark_preprocessing_pipelines,
)
from tabarena.benchmark.feature_selection_methods.feature_selection_methods_register import (
    FEATURE_SELECTION_METHODS,
    get_feature_selector_from_name,
)
from tabarena.benchmark.task.openml import OpenMLTaskWrapper
from tabflow_slurm.benchmarking_setup.data_foundry_integration.data_foundry_task_creator import (
    download_data_foundry_datasets,
)
from tabflow_slurm.run_tabarena_experiment import _parse_task_id


def parse_args():  # noqa: D103
    parser = argparse.ArgumentParser(description="FS Benchmark Runner")
    parser.add_argument("--method_name", type=str, default="FSBench__AccuracyFeatureSelector__5__0__lgbm__3600",
                    help="Feature Selection Method name [default: FSBench__AccuracyFeatureSelector__5__0__lgbm__3600]")
    parser.add_argument("--dataset", type=str, default="anneal/019d3f7b-494a-71fa-8eb2-25d01dfb7792",
                        help="OpenML dataset identifier [default: anneal/019d3f7b-494a-71fa-8eb2-25d01dfb7792]")
    parser.add_argument("--repeat", type=int, default=0,
                        help="Repeat [default: 0]")
    parser.add_argument("--problem_type", type=str, default="regression",
                        help="Problem Type [default: 'regression']")
    parser.add_argument("--eval_metric", type=str, default="rmse",
                        help="Evaluation Metrix [default: 'rmse']")
    parser.add_argument("--noise", type=float, default=1.0, nargs="+",
                        help="Percentage of noise features relative to original feature count [default: 1.0]")
    parser.add_argument("--max-features", type=int, default=5, nargs="+",
                        help="Max feature(s) to select [default: 5]")

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
    # ci_lower : float
        # Lower bound of the confidence interval for mean validity.
    # ci_upper : float
        # Upper bound of the confidence interval for mean validity.
    """
    method: str
    dataset: int
    problem_type: str
    max_features: int
    original_features: int
    noise_features: int
    repeat: int
    elapsed_time_fs: [float]
    n_sample: [int]
    confusion_matrice: [[int]]
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
    method_name = args.method_name
    dataset = args.dataset
    repeat = args.repeat
    problem_type = args.problem_type
    eval_metric = args.eval_metric
    max_features = args.max_features
    split = args.split_index.split("r")[-1].split("f")[0]

    X = task.X
    y = task.y
    n_noise = int(len(X.columns) * args.noise)

    print(f"\n=== {method_name} ===")
    print("Dataset: ", dataset, "Problem type: ", problem_type, "Eval metric: ", eval_metric, "Max features: ", max_features)
    print("Repeat: ", repeat)

    # add noise features
    X_copy, orig_feature_mask = add_noise(X, n_noise)
    if repeat > 0:
        # Resample data (bootstrap) for variability
        sample_idx = np.random.choice(len(X_copy), size=len(X_copy), replace=True)
        X_repeat = X_copy.iloc[sample_idx].reset_index(drop=True)
        y_repeat = y.iloc[sample_idx].reset_index(drop=True)
    else:
        X_repeat = X_copy
        y_repeat = y
    n_sample = len(X_repeat)
    proxy_config = ProxyModelConfig(
        problem_type=problem_type,
        eval_metric=eval_metric,
        model_hyperparameters={"num_boost_round": 1},
    )
    start_time = time.monotonic()
    feature_selector = get_feature_selector_from_name(name=method_name)
    feature_selector = feature_selector(max_features=max_features, proxy_mode_config=proxy_config)
    selected_features = feature_selector.fit_transform(X=X_repeat, y=y_repeat)
    elapsed_time = time.monotonic() - start_time
    # Binary row: 1 if selected, 0 otherwise
    selected_binary = np.zeros(len(X_copy.columns), dtype=bool)
    selected_indices = [X_copy.columns.get_loc(f) for f in selected_features]
    selected_binary[selected_indices] = True
    orig_binary = np.array([orig_feature_mask[col] for col in X_copy.columns])

    cm = confusion_matrix(orig_binary, selected_binary)
    tn, fp, fn, tp = cm.ravel()
    print(f"TP: {tp}  FN: {fn}  FP: {fp}  TN: {tn}")

    # Validity metrics
    validity = getValidity(cm, max_features)
    #ci = confidenceIntervals(validity)
    validity_results = ValidityResult(
        method=method_name,
        dataset=dataset,
        problem_type=problem_type,
        max_features=max_features,
        original_features=len(X.columns),
        noise_features=n_noise,
        repeat=repeat,
        elapsed_time_fs=elapsed_time,
        n_sample=n_sample,
        confusion_matrice=cm,
        validity=validity,
        ci_lower=None,
        ci_upper=None,
    )
    print(f"Validity: {validity_results.validity}")
    return validity_results


def getValidity(cm, max_features) -> list[float]:
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
    _tn, _fp, _fn, tp = cm.ravel()
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

    DATA_FOUNDRY_TASKS = [
        "allstate_claims_severity/019c0a71-9029-727e-a7d9-a4c48238c737",
        "ancestry_study/019d43f5-abb6-7530-8864-c21f082363b3",
        "anneal/019d3f7b-494a-71fa-8eb2-25d01dfb7792",
        "aps_failure/019d4414-84f1-763d-ab04-d8faf9e31c47",
        "audiology_diagnosis/019c7be3-f167-7218-a684-e8cd760cf045",
        "automobile/019d4417-9185-7412-aaa7-ea599ad207d3",
        "bad_customer_detection/019d441e-1bae-7f12-aec3-f882a691ba70",
        "bank_customer_churn/019d442c-fae5-748f-b144-6bd4c791f187",
        "bank_marketing/019d442e-8349-701a-a955-66e7e722724f",
        "bioresponse/019d4431-f400-7589-84da-05e4a83d0cd2",
        "body_density_prediction/019c71b9-743a-7d8b-b130-14340882cf6d",
        "brasilian_houses/019d4437-1097-76ab-86c2-1c453af76050",
        "churn/019d4438-bdb0-78de-858d-041394b879ae",
        "clock_protein_period/019d443a-2314-752d-91a3-7922279ad7fd",
        "clock_protein_toxicity/019d443b-bfe5-7c51-8f37-adcbc9ba6f72",
        "coil_2000_insurance_policies/019d443d-cbac-718a-98ec-b3a9ae46e867",
        "colon_tumor/019d4440-181a-776d-b896-984903dc749b",
        "credit_approval/019cb05b-bb5e-736e-8b40-7f5528c6fe6f",
        "credit_card_clients_default/019d4444-27b6-7cd6-81b7-8d97dfdb7bca",
        "customer_satisfaction_in_airline/019d4444-a63c-7efa-9537-5108e14ef4c0",
        "drug_induced_autoimmunity_prediction/019c9f82-d8aa-7c13-b593-d70835a80e0a",
        "forest_fires/019c71ac-de50-7c3e-bd8d-84dec1f5213f",
        "framingham_heart_study/019d4448-90df-7e75-b057-b72b7edfd6a8",
        "give_me_some_credit/019d4448-cbe4-71ad-9236-a9fe6a291951",
        "hazelnut_spread_contaminant_detection_10GHz/019d4452-ccab-7144-b341-ab8a2375aa88",
        "heart_disease_cleveland/019c7513-909c-707d-a9da-9852f346a015",
        "heart_disease_hungary/019c751a-ac4e-7446-83f1-6d3cd7333a80",
        "heart_disease_switzerland/019c751d-06af-7c95-aa04-957cd32c24f4",
        "heart_disease_va_long_beach/019c7521-3300-74cc-b795-e1b028bbd79f",
        "heloc/019d4456-12af-780b-8642-bbd1f65b0e77",
        "hepatitis_survival_prediction/019c7530-a96b-7845-a664-4c89f661b7c5",
        "hiva_agnostic/019d4457-cc30-71c7-bbef-6c46c871d114",
        "homesite_quote_conversion/019c718e-4b43-78a3-bc0f-1de37f8a9417",
        "horse_colic_survival/019c7554-7410-718c-b0ff-1d659f13c06b",
        "hr_analytics/019d445c-d0d0-7e98-8a96-979918c31004",
        "in_vehicle_coupon_recommendation/019d445e-db47-7d1a-9c30-e40eb064ba5e",
        "ionosphere/019d47b8-ec99-7c1c-91f5-20e67c05362e",
        "japanese_credit_screening/019cb04d-fef7-70be-949e-03181003021f",
        "kdd_cup_09_appetency/019d47c4-d811-73ae-9229-83595e2522b2",
        "labor_negotiations/019d47c9-11b8-792d-bdb3-382fd970bf33",
        "leukemia_allaml/019d47cb-654d-786e-b923-45941a680429",
        "ljubljana_primary_tumor/019c80b5-f3a2-7254-890f-c728e070cfed",
        "lung_cancer/019d47ce-150a-70b9-85b8-1bb85244655d",
        "lymphography/019c75c5-725c-708c-9794-eccc46c0bf81",
        "marketing_campaign/019d47e1-1c2d-798b-ac11-f72a9d7f5f2f",
        "mechanisms_of_action/019d47e2-1aff-7ffb-9b99-61c8a74fc1a2",
        "miami_housing/019d47e2-cfb1-7f2c-ac5c-edb3c90067ac",
        "mic/019d47e3-6739-7346-9627-18b9b81cebca",
        "naticusdroid_android_permissions_dataset/019d47e5-2e69-7914-812d-ee84eeab35c9",
        "nci_ovarian_cancer/019d47ea-3ee8-71e4-850a-ace45ee7456f",
        "nci_pancreatic_cancer/019d47ed-572e-7228-84b6-b3d9a0cc35b7",
        "nci_prostate_cancer/019d47f4-eec6-7dec-882d-5e08502a1236",
        "online_shoppers_purchasing_intention_dataset/019d47f6-d29c-7895-a6c6-1f1bef149ee9",
        "otto_group_product_classification_challenge/019c10cf-1388-7c7b-a68b-5ebb4df6f33a",
        "polish_companies_bankruptcy/019d47fa-1094-7725-bf7b-e68cf54d9358",
        "porto_seguro/019c0ecf-ccbf-7572-9de8-626145c54342",
        "predict_students_dropout_and_academic_success/019d47fb-2264-7890-86c9-0c4d9a7e032a",
        "prostate_cancer/019d47fe-2699-7e75-b3cc-c11f60f8aef8",
        "pva_revenue_prediction_kddcup98/019c101e-100c-7def-aa83-b7a6b15eb4e3",
        "qsar_biodeg/019d4803-a135-7879-8007-9e3ffc50059b",
        "qsar_oral_toxicity/019d4801-956d-73e5-9f5a-3fb681555cc4",
        "qsar_tid_11/019d4801-f46b-70f0-ac4f-a064e923ac10",
        "santander_customer_transaction_prediction/019c1102-f88b-7cdd-9f59-2509e8bef0a7",
        "sdss_17/019d4806-d7f4-7238-9dab-3961ba5d3927",
        "smoking_lung_cancer/019d480b-2ab4-7865-81c4-fdbf8b74d482",
        "soybean_large/019d480e-44db-74ca-a302-744fa35cd88e",
        "srbct_prediction/019d4811-ccd1-7753-8352-908101430ed0",
        "superconductivity/019d4807-5d5b-7b50-bbaf-049008bcbad1",
        "taiwanese_bankruptcy_prediction/019d4813-8e54-7165-bf85-1255476188e1",
        "thyroid_discordant/019c7a7b-48ff-7471-8c09-3a07b8434a16",
        "wine_quality/019d4816-2a31-7d60-a95f-2f6167c8e008",
    ]

    download_data_foundry_datasets(
        benchmark_suite_name="feature_selection_benchmark_validity_examples",
        data_foundry_artifacts=DATA_FOUNDRY_TASKS,
        data_foundry_cache=DEFAULT_DATA_FOUNDRY_CACHE
    )

    preprocessing_pipeline = get_fs_benchmark_preprocessing_pipelines(
        fs_methods=FEATURE_SELECTION_METHODS,
        proxy_model_config=["lgbm"],
        time_limit=[3600],
        total_budget=5,
        include_default=True,
    )

    path_to_metadata = DEFAULT_DATA_FOUNDRY_CACHE / "feature_selection_benchmark_validity_examples_tasks_metadata.csv"
    task_metadata = pd.read_csv(path_to_metadata)

    for method in FEATURE_SELECTION_METHODS:
        validity_results_method = []
        args.method_name = method
        task_metadata = task_metadata.drop_duplicates(subset="repeat", keep="first")
        for idx, row in enumerate(task_metadata.itertuples()):
            args.dataset = task_metadata["task_id_str"].iloc[idx]
            args.repeat = task_metadata["repeat"].iloc[idx]
            args.split_index = task_metadata["split_index"].iloc[idx]
            task_id = _parse_task_id(args.dataset)
            tabarena_task_name = task_id.tabarena_task_name
            task = OpenMLTaskWrapper(
                task=task_id.load_local_openml_task(),
                use_task_eval_metric=False,
            )
            args.eval_metric = task.eval_metric
            args.problem_type = task.problem_type
            validity_results = validity_fs(args)
            validity_results_method.append(validity_results)
        # Calculate confidence intervals for all repeats but the first one
        validities = [r.validity for r in validity_results_method if r.repeat > 0]
        ci = confidenceIntervals(validities)  # [lower, upper]
        for r in validity_results_method:
            if r.repeat > 0:
                r.ci_lower = ci[0]
                r.ci_upper = ci[1]
        pd.DataFrame([asdict(r) for r in validity_results_method]).to_csv(
            f"validity_results_{method}.csv", index=False
        )
