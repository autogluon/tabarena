from typing import Tuple

import numpy as np

from autogluon.core.metrics import make_scorer


def extract_true_class_prob(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Note: Data must be pre-normalized. The data is not normalized within this function for speed purposes.
    The predictions can then be passed to _fast_log_loss.
    :param y_true: an array with shape (n_samples,) that contains all the classes, values must be in [0, n_classes]
    :param y_pred:
        an array with shape (n_samples, n_classes) whose values are the prediction probability assigned to
        the ground truth classes.
        Can also be in shape (n_samples,), where it is then interpreted as binary classification pred probabilities.
    :return: an array with shape (n_samples) that contains the predicted probability of the true class for each sample
    """
    assert len(y_true) == len(y_pred)

    if y_pred.ndim == 1:
        y_true_bool = y_true.astype(np.bool_)
        true_class_prob = y_true_bool*y_pred + ~y_true_bool*(1-y_pred)
    else:
        assert y_pred.ndim == 2
        true_class_prob = y_pred[range(len(y_pred)), y_true]
    return true_class_prob


def extract_true_class_prob_bulk(y_true: np.ndarray, y_pred_bulk: np.ndarray) -> np.ndarray:
    """
    Note: Data must be pre-normalized. The data is not normalized within this function for speed purposes.
    The individual config predictions can then be passed to _fast_log_loss.

    This function is an optimized way to process a batch of config prob predictions, and is faster than calling
    `extract_true_class_prob` in a for loop.

    :param y_true: an array with shape (n_samples,) that contains all the classes, values must be in [0, n_class[
    :param y_pred_bulk:
        an array with shape (n_configs, n_samples, n_classes) whose values are the prediction probability
        assigned to the ground truth classes for each config/model.
        Can also be an array with shape (n_configs, n_samples), where it is then interpreted as binary classification.
    :return: an array with shape (n_configs, n_samples) that contains for each config the predicted probability
    of the true class for each sample.
    """
    ndim = y_pred_bulk.ndim
    if ndim not in [2, 3]:
        raise AssertionError(f'y_pred_bulk.ndim must be either 2 or 3 for this function (ndim={ndim})')
    assert y_pred_bulk.shape[1] == len(y_true), f"y_true and y_pred_bulk have different numbers of samples! " \
                                                f"({len(y_true)}, {y_pred_bulk.shape[1]})"
    if ndim == 2:
        y_true_bool = y_true.astype(np.bool_)
        true_class_prob_bulk = y_true_bool * y_pred_bulk + ~y_true_bool * (1 - y_pred_bulk)
    else:  # ndim == 3
        true_class_prob_bulk = y_pred_bulk[:, range(y_pred_bulk.shape[1]), y_true]
    true_class_prob_bulk = _epsilon(true_class_prob_bulk).astype(np.float32)
    return true_class_prob_bulk


def _preprocess_bulk(y_true: np.ndarray, y_pred_bulk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_pred_bulk = extract_true_class_prob_bulk(y_true=y_true, y_pred_bulk=y_pred_bulk)
    return y_true, y_pred_bulk


def _epsilon(y_pred, eps=1e-15):
    return np.clip(y_pred.astype(float), eps, 1 - eps)


def _mean_log_loss(true_class_prob: np.ndarray) -> float:
    return - np.log(true_class_prob).mean()


def _fast_log_loss_end_to_end(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_class_prob = extract_true_class_prob(y_true=y_true, y_pred=y_pred)
    true_class_prob = _epsilon(true_class_prob).astype(np.float32)
    return _mean_log_loss(true_class_prob=true_class_prob)


"""
Heavily optimized log_loss implementation that is valid under a specific context and avoids all sanity checks.
This can be >100x faster than sklearn.

NOTE: 
1. You must first preprocess the input y_pred by calling `extract_true_class_prob` which converts to a 1-dimensional
  array whose values are the prediction probability assigned to the ground truth class. All other classes that are not 
  the ground truth are ignored, as they are not necessary to calculate log_loss.
2. There is no epsilon / value clipping, ensure y_pred ranges do not include `0` or `1` to avoid infinite loss.
3. There is no support for sample weights.

Parameters
----------
y_true : ignored
true_class_prob : array-like of float that contains the prediction probabilities of the ground truth class. shape = (n_samples,)

Returns
-------
loss
    The negative log-likelihood
"""
# Score function for probabilistic classification
fast_log_loss = make_scorer('log_loss',
                            lambda _, true_class_prob: _mean_log_loss(true_class_prob),
                            optimum=0,
                            greater_is_better=False,
                            needs_proba=True)

fast_log_loss.preprocess_bulk = _preprocess_bulk

# Problem type representation after preprocessing
fast_log_loss.post_problem_type = "binary"

# Score function for probabilistic classification
fast_log_loss_end_to_end = make_scorer('fast_log_loss_end_to_end',
                                       _fast_log_loss_end_to_end,
                                       optimum=0,
                                       greater_is_better=False,
                                       needs_proba=True)
