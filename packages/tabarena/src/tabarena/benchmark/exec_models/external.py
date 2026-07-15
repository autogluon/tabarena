"""Benchmark a self-contained ML system (one that isn't a single AutoGluon model) on TabArena.

Subclass :class:`ExternalSystemModel` to wrap any system that does its own preprocessing, training,
and prediction — for example an AutoML tool or an LLM-driven agent. See
``examples/advanced/run_quickstart_tabarena_external_system.py`` for a runnable example.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.benchmark.exec_models.base import AbstractExecModel

if TYPE_CHECKING:
    import pandas as pd
    from autogluon.core.metrics import Scorer

    from tabarena.benchmark.task.metadata import ValidationMetadata


class ExternalSystemModel(AbstractExecModel):
    """Wrap your own ML system so TabArena can benchmark it.

    Subclass this and implement:

    * ``_fit_system(self, X, y, *, target_name, problem_type, eval_metric, validation_metadata,
      num_cpus, num_gpus, memory_limit, time_limit)`` — train your system and return ``self``.
      Everything the fit needs is passed in (so you never read it off ``self``): the raw training
      frames, the label column name (``target_name`` — its real, semantic name when known), the
      task's problem type / metric / ``validation_metadata`` (group / time columns), and the compute
      budget. ``X`` is yours to edit in place — no defensive copy needed. There is no validation
      split (carve your own from ``X``/``y`` if you want one).
    * ``_predict(self, X)`` — predictions for a regression task (a ``Series`` indexed like ``X``).
    * ``_predict_proba(self, X)`` — class probabilities for a classification task (a ``DataFrame``
      with one column per class label, indexed like ``X``). Implement whichever your task needs.

    Your system is handed the raw data and does its own preprocessing and label handling; TabArena
    fixes the data splits and scoring, so results stay comparable to the leaderboard. Add ``__init__``
    arguments for your system's settings (forward ``**kwargs`` to ``super().__init__``) and a
    ``cleanup`` method to free files / memory. Two optional untimed hooks are available as no-op
    stubs on the base class: override the ``warmup_fn`` property to warm your system's environment
    (imports, JIT/kernel compilation, runtime startup) before the timed fit — it must stay
    data-independent — and/or override ``pre_predict`` / ``post_predict`` for inference-side
    preparation around the timed predict (e.g. bringing your fitted system into serving state /
    releasing it): they may touch the fitted system but never the test data. See
    ``AbstractExecModel`` for the contracts and
    ``examples/advanced/run_quickstart_tabarena_external_system.py`` for a runnable example.
    """

    # An external system gets the raw data and does its own preprocessing, label handling, and
    # validation, so there is no out-of-fold / validation artifact to report.
    preprocess_data = False
    preprocess_label = False
    can_get_oof = False
    can_get_error_val = False

    def __init__(self, *, fit_kwargs: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        # Compute budget, passed (and auto-detected) the same way as for every other method, and
        # forwarded to ``_fit_system`` so subclasses see it as an explicit fit input.
        fit_kwargs = fit_kwargs or {}
        self.num_cpus = fit_kwargs.get("num_cpus")
        self.num_gpus = fit_kwargs.get("num_gpus")
        self.memory_limit = fit_kwargs.get("memory_limit")
        self.time_limit = fit_kwargs.get("time_limit")

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Pass the full fit context to ``_fit_system`` (dropping the unused ``X_val`` / ``y_val``).

        ``X`` is handed over as a frame the system may edit in place: the owned frame when the task
        lazy-loads its data (no extra copy, no extra RAM), otherwise a defensive copy so the caller's
        training frame is never modified.
        """
        X = X if self._can_use_data_in_place else X.copy()
        random_state = self._split_seed if isinstance(self._split_seed, int) else None
        return self._fit_system(
            X,
            y,
            target_name=self.validation_metadata.get_target_name(),
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            validation_metadata=self.validation_metadata,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
            memory_limit=self.memory_limit,
            time_limit=self.time_limit,
            random_state=random_state,
        )

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
    ):
        """Train the system on the full training data and return ``self`` — implement this.

        The one method a subclass must implement.

        There is no validation split — an external system trains on all the data and carves its own
        internal validation from ``X`` / ``y`` if it wants one. The data is handed over raw (your
        system does its own preprocessing and label handling), while TabArena fixes the shared
        evaluation protocol (data splits, feature names, test-row shuffle, scoring), so results stay
        comparable to the leaderboard.

        Args:
            X: Training features, raw and unprocessed. You may edit it in place (e.g. append the
                target column) without a defensive copy: the base hands over the owned frame when the
                task lazy-loads its data, otherwise a copy, so the caller's frame is never modified.
            y: Training labels, raw and aligned with ``X`` by index.
            target_name: The name to give the label column — the task's real target name when known,
                else a safe sentinel (``validation_metadata.get_target_name()``). Passed explicitly so
                that, when the label is materialized as a column (e.g. ``X[target_name] = y``), it
                keeps its *semantic* name; this matters for systems that read column names (LLM-driven
                agents, name-aware preprocessing, ...) rather than a meaningless placeholder.
            problem_type: One of ``"binary"``, ``"multiclass"``, or ``"regression"``.
            eval_metric: The AutoGluon ``Scorer`` the task is scored with (also useful to optimize for).
            validation_metadata: The task's split metadata; read the fields your system needs —
                ``target_name``, ``group_on``, ``time_on``, ``group_time_on``, ``stratify_on``,
                ``group_labels``, ``split_time_horizon`` / ``split_time_horizon_unit``.
            num_cpus: CPU budget for the fit (``None`` = unconstrained / the system's own default).
            num_gpus: GPU budget for the fit (``None`` = unconstrained / the system's own default).
            memory_limit: Memory budget in GB for the fit (``None`` = unconstrained).
            time_limit: Wall-clock budget in seconds for the fit (``None`` = no limit).
            random_state: Per-test-split random seed, derived from the split index so each split gets
                distinct but reproducible randomness. ``None`` when no split seed is available (e.g.
                 a direct ``fit`` outside the runner); seed your system from it and apply your own default
                when ``None``.

        Returns:
            ``self``, fitted and ready for ``_predict`` / ``_predict_proba``.
        """
        raise NotImplementedError
