"""
Original code from https://github.com/dholzmueller/probmetrics
Credit to David Holzmüller
"""

from typing import Callable

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn

from tabarena.utils.temp_scaling.distributions import CategoricalDistribution, CategoricalProbs, CategoricalLogits


class Calibrator(BaseEstimator, ClassifierMixin):
    """
    Calibrator base class. To implement,
    - override at least one of _fit_impl and _fit_torch_impl
    - override at least one of predict_proba and predict_proba_torch
    """

    def __init__(self):
        # assert self.__class__.fit == Calibrator.fit
        assert self.__class__.fit_torch == Calibrator.fit_torch

    def fit(self, X, y) -> 'Calibrator':
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

        self.classes_ = list(range(X.shape[-1]))

        if self.__class__._fit_impl != Calibrator._fit_impl:
            self._fit_impl(X, y)
            return self

        if self.__class__._fit_torch_impl != Calibrator._fit_torch_impl:
            self._fit_torch_impl(y_pred=CategoricalProbs(torch.as_tensor(X)),
                                 y_true_labels=torch.as_tensor(y, dtype=torch.long))
            return self

        raise NotImplementedError()

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError()

    def fit_torch(self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor) -> 'Calibrator':
        assert isinstance(y_true_labels, torch.Tensor)
        assert isinstance(y_pred, CategoricalDistribution)
        # default implementation, using sklearn
        self.classes_ = list(range(y_pred.get_n_classes()))

        if self.__class__._fit_torch_impl != Calibrator._fit_torch_impl:
            self._fit_torch_impl(y_pred, y_true_labels)
            return self

        if self.__class__._fit_impl != Calibrator._fit_impl:
            self._fit_impl(y_pred.get_probs().detach().cpu().numpy(), y_true_labels.detach().cpu().numpy())
            return self

        raise NotImplementedError()

    def _fit_torch_impl(self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor):
        raise NotImplementedError()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.__class__.predict_proba_torch != Calibrator.predict_proba_torch:
            return self.predict_proba_torch(CategoricalProbs(torch.as_tensor(X))).get_probs().numpy()

        raise NotImplementedError()

    def predict_proba_torch(self, y_pred: CategoricalDistribution) -> CategoricalDistribution:
        if self.__class__.predict_proba != Calibrator.predict_proba:
            y_pred_probs = y_pred.get_probs()
            probs = self.predict_proba(y_pred_probs.detach().cpu().numpy())
            return CategoricalProbs(torch.as_tensor(probs, device=y_pred_probs.device, dtype=y_pred_probs.dtype))

        raise NotImplementedError()

    def predict(self, X):
        y_probs = self.predict_proba(X)
        class_idxs = np.argmax(y_probs, axis=-1)
        return np.asarray(self.classes_)[class_idxs]


def bisection_search(f: Callable[[float], float], a: float, b: float, n_steps: int):
    for _ in range(n_steps):
        c = a + 0.5 * (b - a)
        f_c = f(c)
        if f_c > 0:
            b = c
        else:
            a = c

    return 0.5 * (a + b)


class TemperatureScalingCalibrator(Calibrator):
    def __init__(self, opt: str = 'bisection', max_bisection_steps: int = 30, lr: float = 0.1, max_iter: int = 200,
                 use_inv_temp: bool = True, inv_temp_init: float = 1 / 1.5):
        super().__init__()
        self.lr = lr
        self.max_bisection_steps = max_bisection_steps
        self.max_iter = max_iter
        self.use_inv_temp = use_inv_temp
        self.inv_temp_init = inv_temp_init
        self.opt = opt

    def _get_loss_grad(self, invtemp: float, logits: torch.Tensor, y: torch.Tensor):
        part_1 = torch.mean(torch.sum(logits * torch.softmax(invtemp * logits, dim=-1), dim=-1))
        part_2 = torch.mean(logits[torch.arange(logits.shape[0]), y])
        return (part_1 - part_2).item()

    def _fit_torch_impl(self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor):
        logits = y_pred.get_logits()
        labels = y_true_labels

        if self.opt in ['lbfgs', 'lbfgs_line_search']:
            self._fit_lbfgs(logits, labels)
        elif self.opt == 'bisection':
            self._fit_bisection(logits, labels)
        else:
            raise ValueError(f'Unknown optimizer "{self.opt}"')

        # print(f'{self.invtemp_=}')

    def _fit_lbfgs(self, logits: torch.Tensor, labels: torch.Tensor):
        # following https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
        param = nn.Parameter(
            torch.ones(1, device=logits.device) * (self.inv_temp_init if self.use_inv_temp else 1 / self.inv_temp_init))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([param], lr=self.lr, max_iter=self.max_iter,
                                      line_search_fn='strong_wolfe' if self.opt == 'lbfgs_line_search' else None)

        def eval():
            optimizer.zero_grad()
            y_pred = logits * param[:, None] if self.use_inv_temp else logits / param[:, None]
            loss = criterion(y_pred, labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        self.invtemp_ = param.item() if self.use_inv_temp else 1 / param.item()

    def _fit_bisection(self, logits: torch.Tensor, labels: torch.Tensor):
        objective_grad = lambda u, l=logits, tar=labels: self._get_loss_grad(np.exp(u), l, tar)

        # should reach about float32 accuracy
        # need log_2(32) = 5 steps to get to length 1 and then 24 more steps to get to float32 epsilon (2^{-24})
        self.invtemp_ = np.exp(bisection_search(objective_grad, a=-16, b=16, n_steps=self.max_bisection_steps))
        # print(f'{self.invtemp_=}')

    def predict_proba_torch(self, y_pred: CategoricalDistribution) -> CategoricalDistribution:
        return CategoricalLogits(self.invtemp_ * y_pred.get_logits())


class AutoGluonTemperatureScalingCalibrator(Calibrator):
    # adapted from
    # https://github.com/autogluon/autogluon/blob/c1181326cf6b7e3b27a7420273f1a82808d939e2/core/src/autogluon/core/calibrate/temperature_scaling.py#L9
    def __init__(self, init_val: float = 1, max_iter: int = 200, lr: float = 0.1):
        super().__init__()
        self.init_val = init_val
        self.max_iter = max_iter
        self.lr = lr
        self.temperature = init_val

    def _fit_torch_impl(self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor):
        y_val_tensor = y_true_labels
        temperature_param = torch.nn.Parameter(torch.ones(1).fill_(self.init_val))
        logits = y_pred.get_logits()

        is_invalid = torch.isinf(logits).any().tolist()
        if is_invalid:
            return

        nll_criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([temperature_param], lr=self.lr, max_iter=self.max_iter)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        optimizer_trajectory = []

        def temperature_scale_step():
            optimizer.zero_grad()
            temp = temperature_param.unsqueeze(1).expand(logits.size(0), logits.size(1))
            new_logits = logits / temp
            loss = nll_criterion(new_logits, y_val_tensor)
            loss.backward()
            scheduler.step()
            optimizer_trajectory.append((loss.item(), temperature_param.item()))
            return loss

        optimizer.step(temperature_scale_step)

        try:
            best_loss_index = np.nanargmin(np.array(optimizer_trajectory)[:, 0])
        except ValueError:
            return
        temperature_scale = float(np.array(optimizer_trajectory)[best_loss_index, 1])

        if np.isnan(temperature_scale):
            return

        self.temperature = temperature_scale

    def predict_proba_torch(self, y_pred: CategoricalDistribution) -> CategoricalDistribution:
        with torch.no_grad():
            return CategoricalLogits(y_pred.get_logits() / self.temperature)


class AutoGluonTemperatureScalingCalibratorFixed(AutoGluonTemperatureScalingCalibrator):
    def fit(self, X, y, scorer=None):
        if scorer is None:
            from autogluon.core.metrics import get_metric
            scorer = get_metric("log_loss", problem_type="multiclass")
        super().fit(X=X, y=y)
        if self.temperature == 1:
            return self
        elif self.temperature <= 0:
            print(f"NEGATIVE TEMP, SETTING TO 1")
            self.temperature = 1
            return self
        y_pred_proba_post = self.predict_proba(X)

        err_og = scorer.error(y, X)
        err_post = scorer.error(y, y_pred_proba_post)
        if err_post > err_og:
            print(f"WORSE ERROR: SETTING TO 1")
            self.temperature = 1
        return self


class TemperatureScalingCalibratorFixed(TemperatureScalingCalibrator):
    def fit(self, X, y, scorer=None):
        if scorer is None:
            from autogluon.core.metrics import get_metric
            scorer = get_metric("log_loss", problem_type="multiclass")
        super().fit(X=X, y=y)
        if self.inv_temp_init == 1:
            return self
        elif self.inv_temp_init <= 0:
            print(f"NEGATIVE TEMP, SETTING TO 1")
            self.inv_temp_init = 1
            return self
        y_pred_proba_post = self.predict_proba(X)

        err_og = scorer.error(y, X)
        err_post = scorer.error(y, y_pred_proba_post)
        if err_post > err_og:
            print(f"WORSE ERROR: SETTING TO 1")
            self.inv_temp_init = 1
        return self


def logloss_np(y_true: np.ndarray, y_proba: np.ndarray):
    return -np.mean(np.take_along_axis(np.log(y_proba), y_true[:, None], axis=1))
