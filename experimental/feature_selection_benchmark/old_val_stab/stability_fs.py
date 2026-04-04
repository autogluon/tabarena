import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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
    parser.add_argument("--max-features", type=int, default=5, nargs="+",
                        help="Max feature(s) to select [default: 5]")
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
    repeat: int
    elapsed_time_fs: [float]
    n_sample: [int]
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
    method_name = args.method_name
    dataset = args.dataset
    repeat = args.repeat
    problem_type = args.problem_type
    eval_metric = args.eval_metric
    max_features = args.max_features
    split = args.split_index.split("r")[-1].split("f")[0]

    X = task.X
    y = task.y

    n_features = len(X.columns)
    print(f"\n=== {method_name} ===")
    print("Max features: ", max_features)
    print("Dataset: ", dataset)
    print("Problem type: ", problem_type)
    print("Eval metric: ", eval_metric)
    # Store binary selections across repeats
    # Resample data (bootstrap) for variability
    sample_idx = np.random.choice(len(X), size=len(X), replace=True)
    X_repeat = X.iloc[sample_idx].reset_index(drop=True)
    y_repeat = y.iloc[sample_idx].reset_index(drop=True)
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
    selected_binary = np.zeros(n_features)
    selected_indices = [X.columns.get_loc(f) for f in selected_features]
    selected_binary[selected_indices] = 1

    stability_results = StabilityResult(
        method=method_name,
        dataset=dataset,
        problem_type=problem_type,
        max_features=max_features,
        original_features=len(X.columns),
        repeat=repeat,
        elapsed_time_fs=elapsed_time,
        n_sample=len(X_repeat),
        selected_features=selected_binary,
        stability=None,
        ci_lower=None,
        ci_upper=None
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
        # "allstate_claims_severity/019c0a71-9029-727e-a7d9-a4c48238c737",
        "ancestry_study/019d43f5-abb6-7530-8864-c21f082363b3",
    ]

    download_data_foundry_datasets(
        benchmark_suite_name="feature_selection_benchmark_stability_examples",
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
    path_to_metadata = DEFAULT_DATA_FOUNDRY_CACHE / "feature_selection_benchmark_stability_examples_tasks_metadata.csv"
    task_metadata = pd.read_csv(path_to_metadata)

    for method in FEATURE_SELECTION_METHODS:
        stability_results_method = []
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
            stability_results = stability_fs(args)
            stability_results_method.append(stability_results)
        # Calculate confidence intervals for all repeats but the first one
        selected_features = [r.selected_features for r in stability_results_method if r.repeat > 0]
        stability = getStability(selected_features)
        ci = confidenceIntervals(selected_features)  # [lower, upper]
        for r in stability_results_method:
            r.stability = stability
            r.ci_lower = ci["lower"]
            r.ci_upper = ci["upper"]
        pd.DataFrame([asdict(r) for r in stability_results_method]).to_csv(
            f"stability_results_{method}.csv", index=False
        )
