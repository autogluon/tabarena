"""The two AutoGluon models the system bags: LightGBM with TabPrep, and the LightGBM default.

:class:`LGBMPrepModel` is one bag child: AutoGluon's preprocessing, then (for a TabPrep config) TabPrep fitted on
the child's training rows only and applied to its validation rows, then AutoGluon's ``LGBModel`` with the config's
LightGBM parameters and the config's own seed. The outer AutoGluon bag owns the folds and the out-of-fold /
test averaging. :class:`LGBMDefaultModel` is AutoGluon's ``LGBModel`` with its default hyperparameters, under
the system's model family name.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.core.models import AbstractModel
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.tabular.models import LGBModel

from tabarena.systems.lgbm_prep_fte._tabprep import tabprep_frames

AG_KEY = "LGBM-PREP-FTE"
AG_NAME = "LGBMPrepFTE"
MAX_THREADS = 16
"""LightGBM and the feature cap use at most this many threads, as in the reference runs."""


class LGBMPrepModel(AbstractModel):
    """One fitted inner fold of an LGBM-Prep config.

    Hyperparameters (one portfolio config): ``config_id``, ``model_seed`` (LightGBM's ``random_state``),
    ``lightgbm`` (LightGBM parameters), ``tabprep_enabled`` and ``tabprep`` (TabPrep budgets and seeds).
    """

    ag_key = AG_KEY
    ag_name = AG_NAME
    # A bag child needs its inner validation fold, which TabArena's untimed warm-up fit does not pass. The
    # LightGBM stack is still warmed up by the default config's ``LGBModel``.
    warmup_dummy_fit = False

    @classmethod
    def supported_problem_types(cls):
        return ["binary", "multiclass", "regression"]

    def _get_default_resources(self):
        return MAX_THREADS, 0

    def get_minimum_resources(self, is_gpu_available=False):
        return {"num_cpus": 1, "num_gpus": 0}

    def _more_tags(self):
        return {"can_refit_full": False}

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_cpus=MAX_THREADS, num_gpus=0, **kwargs):
        if num_gpus:
            raise ValueError("LGBM-Prep runs on CPUs only (num_gpus=0)")
        if X_val is None or y_val is None:
            raise ValueError("LGBM-Prep needs the inner validation fold of its bag")
        started = time.monotonic()
        params = self._get_model_params()
        lgbm_params = copy.deepcopy(params["lightgbm"])
        # The config's seed is part of the config: it overrides any per-fold seed the bag injects.
        lgbm_params["random_state"] = params["model_seed"]
        # Inner folds can arrive with non-contiguous indices; align targets by position.
        X = self.preprocess(X).reset_index(drop=True)
        X_val = self.preprocess(X_val).reset_index(drop=True)
        threads = min(MAX_THREADS, num_cpus)
        self.tabprep_ = None
        self.selected_columns_ = list(X.columns)
        self.prep_report_ = {"tabprep_enabled": bool(params["tabprep_enabled"]), "lightgbm_threads": threads}
        if params["tabprep_enabled"]:
            frames, report, self.tabprep_, self.selected_columns_ = tabprep_frames(
                {"train": X, "val": X_val, "test": X_val.iloc[:0]},
                np.asarray(y),
                self.problem_type,
                params["tabprep"],
                num_cpus=num_cpus,
            )
            X, X_val = frames["train"], frames["val"]
            self.prep_report_.update(report)
            self.prep_report_["stage_seconds"] = self.tabprep_.timings_
        self.prep_report_["preprocessing_seconds"] = time.monotonic() - started
        remaining = None if time_limit is None else time_limit - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            raise TimeLimitExceeded(f"TabPrep used the whole fold budget: {self.prep_report_}")
        self.model = LGBModel(
            path=str(Path(self.path) / "lightgbm"),
            name="LightGBM",
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            hyperparameters=lgbm_params,
        )
        self.model.fit(
            X=X,
            y=pd.Series(np.asarray(y), index=X.index),
            X_val=X_val,
            y_val=pd.Series(np.asarray(y_val), index=X_val.index),
            time_limit=remaining,
            num_cpus=threads,
            num_gpus=0,
            verbosity=0,
        )

    def _predict_proba(self, X, num_cpus=2, **kwargs):
        X = self.preprocess(X)
        if self.tabprep_ is not None:
            X = self.tabprep_.transform(X).loc[:, self.selected_columns_]
        return self.model.predict_proba(X, num_cpus=num_cpus)


class LGBMDefaultModel(LGBModel):
    """AutoGluon's LightGBM with its default hyperparameters, in the system's model family."""

    ag_key = AG_KEY
    ag_name = AG_NAME
