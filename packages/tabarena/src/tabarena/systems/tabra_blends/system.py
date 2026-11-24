from __future__ import annotations

from tempfile import gettempdir
from time import monotonic
from typing import TYPE_CHECKING

from tabarena.benchmark.exec_models import ExternalSystemModel

if TYPE_CHECKING:
    import pandas as pd
    from autogluon.core.metrics import Scorer

    from tabarena.benchmark.exec_models.autogluon import AGSingleWrapper
    from tabarena.benchmark.task.metadata import ValidationMetadata


class TabraBlendsSystemModel(ExternalSystemModel):
    """Average 4-estimator Kumo Large and the first LimiX-2-1016 pipeline equally.

    Built with StableAI LimiX. This recipe is intended for non-commercial research.
    Both members use the per-split seed and the full labeled training split.
    """

    warmup_torch_device = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._members: list[AGSingleWrapper] = []

    def _fit_system(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        target_name: str,
        problem_type: str,
        eval_metric: Scorer,
        validation_metadata: ValidationMetadata,
        num_cpus: int | None,
        num_gpus: int | None,
        memory_limit: float | None,
        time_limit: float | None,
        random_state: int | None,
    ) -> TabraBlendsSystemModel:
        from tabarena.benchmark.exec_models.autogluon import AGSingleWrapper
        from tabarena.models.kumo_tabular.model import KumoTabularModel
        from tabarena.models.limix_2.model import LimiX2Model

        if num_gpus is not None and num_gpus < 1:
            raise ValueError("Tabra Blends needs one GPU.")

        self.cleanup()
        started = monotonic()
        recipes = (
            (KumoTabularModel, {"num_estimators": 4, "cache_context": True}),
            (LimiX2Model, {"n_estimators": 1}),
        )
        try:
            for model_cls, hyperparameters in recipes:
                remaining = None if time_limit is None else time_limit - (monotonic() - started)
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("Tabra Blends exhausted its fit budget.")
                member = AGSingleWrapper(
                    model_cls=model_cls,
                    model_hyperparameters=hyperparameters,
                    problem_type=problem_type,
                    eval_metric=eval_metric,
                    validation_metadata=validation_metadata,
                    no_validation=True,
                    init_kwargs={"verbosity": 2, "default_base_path": gettempdir()},
                    fit_kwargs={
                        "num_cpus": num_cpus,
                        "num_gpus": 1,
                        "memory_limit": memory_limit,
                        "time_limit": remaining,
                        "random_seed": 0 if random_state is None else random_state,
                        "raise_on_model_failure": True,
                    },
                )
                self._members.append(member)
                member.fit(X, y)
                # Charge member inference preparation to the system's fit.
                member.pre_predict()
            if time_limit is not None and monotonic() - started > time_limit:
                raise TimeoutError("Tabra Blends exceeded its fit budget.")
        except Exception:
            self.cleanup()
            raise
        return self

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        kumo, limix = (member.predict_proba(X) for member in self._members)
        if set(kumo.columns) != set(limix.columns) or not kumo.index.equals(limix.index):
            raise ValueError("Kumo and LimiX returned different classes or row indices.")
        return (kumo.astype("float64") + limix.loc[:, kumo.columns].astype("float64")) * 0.5

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        if self.problem_type != "regression":
            return self._predict_proba(X).idxmax(axis=1)
        kumo, limix = (member.predict(X) for member in self._members)
        if not kumo.index.equals(limix.index):
            raise ValueError("Kumo and LimiX returned different row indices.")
        return (kumo.astype("float64") + limix.astype("float64")) * 0.5

    def cleanup(self) -> None:
        for member in self._members:
            member.cleanup()
        self._members.clear()
