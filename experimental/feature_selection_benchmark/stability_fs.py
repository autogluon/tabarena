import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from autogluon.core.data import LabelCleaner
from scipy.stats import norm
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
    parser.add_argument("--method_name", type=str, default="MIFeatureSelector",
                        help="Feature Selection Method name [default: 'MIFeatureSelector']")
    parser.add_argument("--dataset", type=int, default=55,
                        help="OpenML dataset identifier [default: 55]")
    parser.add_argument("--problem_type", type=str, default="binary",
                        help="OpenML dataset problem type [default: 'binary']")
    parser.add_argument("--max-features", type=int, default=5, nargs="+",
                        help="Max feature(s) to select [default: 5]")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Number of bootstrap repeats [default: 10]")
    return parser.parse_args()

@dataclass
class StabilityResult:
    """Result object containing feature selection stability metrics from multiple repeats.

    Attributes:
    ----------
    method : str
        Name of the feature selection method evaluated.
    dataset : str
        Dataset identifier used for evaluation.
    problem_type : str
        ML problem type ('binary_classification', 'multiclass_classification', 'regression').
    max_features : int
        Maximum number of features requested by the selector.
    original_features : int
        Number of features in the original dataset.
    repeats : int
        Number of bootstrap repeats performed.
    elapsed_time_fs : list[float]
        List of runtime measurements (seconds) for each repeat.
    n_samples : list[int]
        List of sample sizes used in each repeat.
    selected_features : list[int]
        List of number of features selected in each repeat.
    stability : list[float]
        List of stability scores computed for each repeat.
    ci_lower : float
        Lower bound of the confidence interval for mean stability.
    ci_upper : float
        Upper bound of the confidence interval for mean stability.
    """
    method: str
    dataset: str
    problem_type: str
    max_features: int
    original_features: int
    repeats: int
    elapsed_time_fs: [float]
    n_samples: [int]
    selected_features: [int]
    stability: [float]
    ci_lower: float
    ci_upper: float

def stability_fs(args) -> StabilityResult:
    """Evaluate feature selection method stability through bootstrap consistency analysis.

    This function assesses how consistently a feature selection method selects the same
    features across multiple bootstrap repeats of the same dataset. It performs
    repeated feature selection on bootstrap samples and computes stability metrics
    based on feature selection overlap.

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

    Returns:
    -------
    StabilityResult
        Object containing stability metrics, timing information, sample sizes,
        and confidence intervals for the feature selection method's consistency.
    """
    dataset = args.dataset
    task_id = _parse_task_id(dataset)
    tabarena_task_name = task_id.tabarena_task_name
    task = OpenMLTaskWrapper(
        task=task_id.load_local_openml_task(),
        use_task_eval_metric=False,
    )
    eval_metric = task.eval_metric
    problem_type = task.problem_type
    n_repeats = args.repeats
    max_features = args.max_features
    method_name = args.method_name.split("__")[1].split("__")[0]

    X = task.X
    y = task.y

    n_features = len(X.columns)
    stability_results = {}

    print(f"\n=== {method_name} ===")
    print("Max features: ", max_features)
    print("Dataset: ", dataset)
    print("Problem type: ", problem_type)
    print("Eval metric: ", eval_metric)
    # Store binary selections across repeats
    Z = []
    times = []
    n_samples = []
    for _repeat in range(n_repeats):
        # Resample data (bootstrap) for variability
        sample_idx = np.random.choice(len(X), size=len(X), replace=True)
        X_repeat = X.iloc[sample_idx].reset_index(drop=True)
        y_repeat = y.iloc[sample_idx].reset_index(drop=True)
        n_samples.append(len(X_repeat))
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
        times.append(elapsed_time)

        # Binary row: 1 if selected, 0 otherwise
        selected_binary = np.zeros(n_features)
        selected_indices = [X.columns.get_loc(f) for f in selected_features]
        selected_binary[selected_indices] = 1
        Z.append(selected_binary)

    # Stability metrics
    stability = getStability(Z)
    ci = confidenceIntervals(Z, alpha=0.05)

    stability_results = StabilityResult(
        method=method_name,
        dataset=dataset,
        problem_type=problem_type,
        max_features=max_features,
        original_features=len(X.columns),
        repeats=n_repeats,
        elapsed_time_fs=times,
        n_samples=n_samples,
        selected_features=Z,
        stability=stability,
        ci_lower=ci["lower"],
        ci_upper=ci["upper"]
    )

    print(f"Stability: {stability_results.stability}")
    return stability_results



# NOGUEIRAS CODE (https://github.com/nogueirs/JMLR2018/blob/master/python/stability/__init__.py):
# Nogueira, Sarah, Konstantinos Sechidis, and Gavin Brown. "On the stability of feature selection algorithms."
# Journal of Machine Learning Research 18.174 (2018): 1-54.

def getStability(Z) -> float:
    """Compute feature selection stability (Nogueira et al., 2017).

    Calculates the stability estimator for M feature sets over d features, measuring
    consistency of feature selection across bootstrap repeats. The estimator is
    defined as 1 minus the normalized average pairwise feature selection variance,
    achieving maximum value (1.0) when all feature sets are identical.

    Parameters
    ----------
    Z : array-like of shape (M, d)
        Binary selection matrix where rows are feature sets (M repeats) and
        columns are features (d total). Z[m, f] = 1 if feature f was selected
        in repeat m, 0 otherwise.

    Returns:
    -------
    float
        Stability score in range [0, 1], where 1 indicates perfect consistency
        across all repeats.

    Notes:
    -----
    [1] Nogueira, S., Brown, G., & Jorge, A. (2017). On the Stability of
    Feature Selection Algorithms. JMLR, 18(1), 6345-6378.
    """
    Z = checkInputType(Z)
    M, d = Z.shape
    hatPF = np.mean(Z, axis=0)
    kbar = np.sum(hatPF)
    denom = (kbar / d) * (1 - kbar / d)
    return 1 - (M / (M - 1)) * np.mean(np.multiply(hatPF, 1 - hatPF)) / denom


def getVarianceofStability(Z) -> dict[str, float]:
    """Compute feature selection stability variance (Nogueira et al., 2017).

    Parameters
    ----------
    Z : array-like of shape (M, d)
        Binary selection matrix where rows are feature sets (M repeats) and
        columns are features (d total). Z[m, f] = 1 if feature f was selected
        in repeat m, 0 otherwise.

    Returns:
    -------
    dict[str, float]
        Dictionary with keys:
        - 'stability': Stability score in range [0, 1] (1 = perfect consistency).
        - 'variance': Variance of the stability estimator.

    Notes
    -----
    [1] Nogueira, S., Brown, G., & Jorge, A. (2017). On the Stability of
    Feature Selection Algorithms. JMLR, 18(1), 6345-6378.
    """
    Z = checkInputType(Z)  # check the input Z is of the right type
    M, d = Z.shape  # M is the number of feature sets and d the total number of features
    hatPF = np.mean(Z, axis=0)  # hatPF is a numpy.array with the frequency of selection of each feature
    kbar = np.sum(hatPF)  # kbar is the average number of selected features over the M feature sets
    k = np.sum(Z, axis=1)  # k is a numpy.array with the number of features selected on each one of the M feature sets
    denom = (kbar / d) * (1 - kbar / d)
    stab = 1 - (M / (M - 1)) * np.mean(np.multiply(hatPF, 1 - hatPF)) / denom  # the stability estimate
    phi = np.zeros(M)
    for i in range(M):
        phi[i] = (1 / denom) * (np.mean(np.multiply(Z[i,], hatPF)) - (k[i] * kbar) / d ** 2 + (stab / 2) * (
                (2 * k[i] * kbar) / d ** 2 - k[i] / d - kbar / d + 1))
    phiAv = np.mean(phi)
    variance = (4 / M ** 2) * np.sum(np.power(phi - phiAv, 2))  # the variance of the stability estimate as given in [1]
    return {"stability": stab, "variance": variance}


def confidenceIntervals(Z, alpha=0.05, res=None):
    """Compute feature selection stability confidence intervals (Nogueira et al., 2017).

    Parameters:
    ----------
    Z : array-like of shape (M, d)
        Binary selection matrix where rows are feature sets (M repeats) and
        columns are features (d total). Z[m, f] = 1 if feature f was selected
        in repeat m, 0 otherwise.
    alpha : float, optional
        Significance level for confidence interval (default: 0.05 for 95% CI).
    res : dict, optional
        Pre-computed result from `getVarianceofStability(Z)` for faster computation.

    Returns:
    -------
    dict[str, float]
        Dictionary with keys:
        - 'stability': Stability score in range [0, 1] (1 = perfect consistency).
        - 'lower': Lower bound of (1-alpha) confidence interval.
        - 'upper': Upper bound of (1-alpha) confidence interval.

    Notes:
    -----
    [1] Nogueira, S., Brown, G., & Jorge, A. (2017). On the Stability of
    Feature Selection Algorithms. JMLR, 18(1), 6345-6378.
    """
    if res is None:
        res = {}
    Z = checkInputType(Z)  # check the input Z is of the right type
    ## we check if values of alpha between ) and 1
    if alpha >= 1 or alpha <= 0:
        raise ValueError("The level of significance alpha should be a value >0 and <1")
    if len(res) == 0:
        res = getVarianceofStability(Z)  # get a dictionnary with the stability estimate and its variance
    lower = res["stability"] - norm.ppf(1 - alpha / 2) * math.sqrt(
        res["variance"])  # lower bound of the confidence interval at a level alpha
    upper = res["stability"] + norm.ppf(1 - alpha / 2) * math.sqrt(
        res["variance"])  # upper bound of the confidence interval
    return {"stability": res["stability"], "lower": lower, "upper": upper}


# this tests whether the true stability is equal to a given value stab0
def hypothesisTestV(Z, stab0, alpha=0.05):
    """Let us assume we have M>1 feature sets and d>0 features in total.
    This function implements the null hypothesis test in [1] that test whether the population stability is greater
    than a given value stab0.

    INPUTS:- A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d, raises a ValueError exception
    otherwise).
             Each row of the binary matrix represents a feature set, where a 1 at the f^th position
             means the f^th feature has been selected and a 0 means it has not been selected.
           - stab0 is the value we want to compare the stability of the feature selection to.
           - alpha is an optional argument corresponding to the level of significance of the null hypothesis test
             (default is 0.05).

    OUTPUT: A dictionnary with:
            - a boolean value for key 'reject' equal to True if the null hypothesis is rejected and to False otherwise
            - a float for the key 'V' giving the value of the test statistic
            - a float giving for the key 'p-value' giving the p-value of the hypothesis test
    """
    Z = checkInputType(Z)  # check the input Z is of the right type
    res = getVarianceofStability(Z)
    V = (res["stability"] - stab0) / math.sqrt(res["variance"])
    zCrit = norm.ppf(1 - alpha)
    reject = zCrit <= V
    pValue = 1 - norm.cdf(V)
    return {"reject": reject, "V": V, "p-value": pValue}


# this tests the equality of the stability of two algorithms
def hypothesisTestT(Z1, Z2, alpha=0.05):
    """Let us assume we have M>1 feature sets and d>0 features in total.
    This function implements the null hypothesis test of Theorem 10 in [1] that test whether
    two population stabilities are identical.

    INPUTS:- Two BINARY matrices Z1 and Z2 (given as lists or as numpy.ndarray objects of size M*d).
             Each row of the binary matrix represents a feature set, where a 1 at the f^th position
             means the f^th feature has been selected and a 0 means it has not been selected.
           - alpha is an optional argument corresponding to the level of significance of the null
             hypothesis test (default is 0.05)

    OUTPUT: A dictionnary with:
            - a boolean value for key 'reject' equal to True if the null hypothesis is rejected and to False otherwise
            - a float for the key 'T' giving the value of the test statistic
            - a float giving for the key 'p-value' giving the p-value of the hypothesis test
    """
    Z1 = checkInputType(Z1)  # check the input Z1 is of the right type
    Z2 = checkInputType(Z2)  # check the input Z2 is of the right type
    res1 = getVarianceofStability(Z1)
    res2 = getVarianceofStability(Z2)
    stab1 = res1["stability"]
    stab2 = res2["stability"]
    var1 = res1["variance"]
    var2 = res2["variance"]
    T = (stab2 - stab1) / math.sqrt(var1 + var2)
    zCrit = norm.ppf(1 - alpha / 2)
    # the cumulative inverse of the gaussian at 1-alpha/2
    reject = abs(T) >= zCrit
    pValue = 2 * (1 - norm.cdf(abs(T)))
    return {"reject": reject, "T": T, "p-value": pValue}


def checkInputType(Z):
    """This function checks that Z is of the rigt type and dimension.
    It raises an exception if not.
    OUTPUT: The input Z as a numpy.ndarray.
    """
    ### We check that Z is a list or a numpy.array
    if isinstance(Z, list):
        Z = np.asarray(Z)
    elif not isinstance(Z, np.ndarray):
        raise ValueError("The input matrix Z should be of type list or numpy.ndarray")
    ### We check if Z is a matrix (2 dimensions)
    if Z.ndim != 2:
        raise ValueError("The input matrix Z should be of dimension 2")
    return Z


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
        # "homesite_quote_conversion/019c718e-4b43-78a3-bc0f-1de37f8a9417",
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
        # "pva_revenue_prediction_kddcup98/019c101e-100c-7def-aa83-b7a6b15eb4e3",
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
    for method_name in preprocessing_pipeline:
        args.method_name = method_name
        for dataset in task_metadata["task_id_str"]:
            args.dataset = dataset
            validity_results = stability_fs(args)
