import math

import numpy as np
import openml
from autogluon.core.data import LabelCleaner
from scipy.stats import norm
from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import ProxyModelConfig
from tabarena.benchmark.feature_selection_methods.feature_selection_methods_register import (
    FEATURE_SELECTION_METHODS,
    get_feature_selector_from_name,
)


def stability_fs(method_names):  # noqa: D103
    dataset_id = 55
    problem_type = "binary"
    n_repeats = 30
    max_features = 5

    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
    y = label_cleaner.transform(y)

    n_features = len(X.columns)
    stability_results = {}

    for method_name in method_names:
        print(f"\n=== {method_name} ===")

        # Store binary selections across repeats
        Z_repeats = []

        for _repeat in range(n_repeats):
            # Resample data (bootstrap) for variability
            sample_idx = np.random.choice(len(X), size=len(X), replace=True)
            X_repeat = X.iloc[sample_idx].reset_index(drop=True)
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
            selected_binary = np.zeros(n_features)
            selected_indices = [X.columns.get_loc(f) for f in selected_features]
            selected_binary[selected_indices] = 1
            Z_repeats.append(selected_binary)

        # Z matrix: n_repeats x n_features
        Z = np.array(Z_repeats)

        # Stability metrics
        stability = getStability(Z)
        ci = confidenceIntervals(Z, alpha=0.05)

        stability_results[method_name] = {
            "stability": stability,
            "ci_lower": ci["lower"],
            "ci_upper": ci["upper"],
            "Z": Z  # Save for further analysis
        }

        print(f"Stability: {stability:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}]")

    return stability_results



# NOGUEIRAS CODE (https://github.com/nogueirs/JMLR2018/blob/master/python/stability/__init__.py):
# Nogueira, Sarah, Konstantinos Sechidis, and Gavin Brown. "On the stability of feature selection algorithms."
# Journal of Machine Learning Research 18.174 (2018): 1-54.

def getStability(Z):
    """Let us assume we have M>1 feature sets and d>0 features in total.
    This function computes the stability estimate as given in Definition 4 in  [1].

    INPUT: A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d).
           Each row of the binary matrix represents a feature set, where a 1 at the f^th position
           means the f^th feature has been selected and a 0 means it has not been selected.

    OUTPUT: The stability of the feature selection procedure
    """
    Z = checkInputType(Z)
    M, d = Z.shape
    hatPF = np.mean(Z, axis=0)
    kbar = np.sum(hatPF)
    denom = (kbar / d) * (1 - kbar / d)
    return 1 - (M / (M - 1)) * np.mean(np.multiply(hatPF, 1 - hatPF)) / denom


def getVarianceofStability(Z):
    """Let us assume we have M>1 feature sets and d>0 features in total.
    This function computes the stability estimate and its variance as given in [1].

    INPUT: A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d, raises a ValueError exception
    otherwise).
           Each row of the binary matrix represents a feature set, where a 1 at the f^th position
           means the f^th feature has been selected and a 0 means it has not been selected.

    OUTPUT: A dictionnary where the key 'stability' provides the corresponding stability value #
            and where the key 'variance' provides the variance of the stability estimate
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
    """Let us assume we have M>1 feature sets and d>0 features in total.
    This function provides the stability estimate and the lower and upper bounds of the (1-alpha)- approximate
    confidence
    interval as given by Corollary 9 in [1].

    INPUTS: - A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d, raises a ValueError exception
    otherwise).
              Each row of the binary matrix represents a feature set, where a 1 at the f^th position
              means the f^th feature has been selected and a 0 means it has not been selected.
            - alpha is an optional argument corresponding to the level of significance for the confidence interval
              (default is 0.05), e.g. alpha=0.05 give the lower and upper bound of for a (1-alpha)=95% confidence
              interval.
            - In case you already computed the stability estimate of Z using the function getVarianceofStability(Z),
              you can provide theresult (a dictionnary) as an optional argument to this function for faster computation.

    OUTPUT: - A dictionnary where the key 'stability' provides the corresponding stability value, where:
                  - the key 'variance' provides the variance of the stability estimate;
                  - the keys 'lower' and 'upper' respectively give the lower and upper bounds
                    of the (1-alpha)-confidence interval.
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
    stability_results = stability_fs(FEATURE_SELECTION_METHODS)
