from __future__ import annotations

import logging
import time

import pandas as pd

from autogluon.tabular import __version__

from autogluon.tabular.models.tabm.tabm_model import TabMModel
from tabarena.tabarena.tabarena.benchmark.models.prep_ag.prep_mixin import ModelAgnosticPrepMixin

logger = logging.getLogger(__name__)

class PrepTabMModel(ModelAgnosticPrepMixin, TabMModel):
    """
    TabM is an efficient ensemble of MLPs that is trained simultaneously with mostly shared parameters.

    TabM is one of the top performing methods overall on TabArena-v0.1: https://tabarena.ai

    Paper: TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling
    Authors: Yury Gorishniy, Akim Kotelnikov, Artem Babenko
    Codebase: https://github.com/yandex-research/tabm
    License: Apache-2.0

    Partially adapted from pytabkit's TabM implementation.

    .. versionadded:: 1.4.0
    """
    ag_key = "prep_TABM"
    ag_name = "prep_TabM"
    ag_priority = 85
    seed_name = "random_state"

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        **kwargs,
    ):
        start_time = time.time()

        try:
            # imports various dependencies such as torch
            from torch.cuda import is_available

            from autogluon.tabular.models.tabm._tabm_internal import TabMImplementation
        except ImportError as err:
            logger.log(
                40,
                f"\tFailed to import tabm! To use the TabM model, "
                f"do: `pip install autogluon.tabular[tabm]=={__version__}`.",
            )
            raise err

        device = "cpu" if num_gpus == 0 else "cuda"
        if (device == "cuda") and (not is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        if X_val is None:
            from autogluon.core.utils import generate_train_test_split

            X_train, X_val, y_train, y_val = generate_train_test_split(
                X=X,
                y=y,
                problem_type=self.problem_type,
                test_size=0.2,
                random_state=0,
            )

        hyp = self._get_model_params()
        bool_to_cat = hyp.pop("bool_to_cat", True)
        prep_params = hyp.pop("prep_params", {})

        X = self.preprocess(X, y=y, is_train=True, prep_params=prep_params, bool_to_cat=bool_to_cat)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        self.model = TabMImplementation(
            n_threads=num_cpus,
            device=device,
            problem_type=self.problem_type,
            early_stopping_metric=self.stopping_metric,
            **hyp,
        )

        self.model.fit(
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
            cat_col_names=X.select_dtypes(include="category").columns.tolist(),
            time_to_fit_in_seconds=time_limit - (time.time() - start_time) if time_limit is not None else None,
        )

