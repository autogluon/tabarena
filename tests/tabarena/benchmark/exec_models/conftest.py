"""Shared fakes for the exec-model tests: a predictor/trainer stand-in for AutoGluon and hookable models.

Nothing here fits a model. The classes are importable from other test packages as
``tests.tabarena.benchmark.exec_models.conftest`` (pytest registers this conftest under that name),
so the persist-inference tests and the system tests exercise the same fakes.
"""

from __future__ import annotations


class _PreparableModel:
    """A model object declaring the optional ``prepare_for_inference`` hook; records its calls."""

    def __init__(self, name: str, log: list[str]):
        self.name = name
        self._log = log

    def prepare_for_inference(self) -> None:
        self._log.append(self.name)


class _PlainModel:
    """A persisted model without the optional hook; must be skipped without error."""


class _FakeBag(_PreparableModel):
    """A bagged-ensemble stand-in whose string children are loaded by ``persist_child_models``."""

    def __init__(self, name: str, log: list[str], children: list):
        super().__init__(name, log)
        self.models = list(children)
        self.persist_child_models_calls = 0

    def persist_child_models(self) -> None:
        self.persist_child_models_calls += 1
        self.models = [_PreparableModel(child, self._log) if isinstance(child, str) else child for child in self.models]


class _FakeFittedModel:
    """A loaded best model for the metadata tests; counts the (expensive) ``get_info`` calls."""

    fit_num_cpus = 4
    fit_num_gpus = 1
    fit_num_cpus_child = 2
    fit_num_gpus_child = 0.5
    _memory_usage_estimate = 12345

    def __init__(self, info: dict):
        self._info = info
        self.get_info_calls = 0

    def get_info(self, include_feature_metadata: bool = True) -> dict:
        assert include_feature_metadata is False
        self.get_info_calls += 1
        return self._info

    def disk_usage(self) -> int:
        return 42

    def get_fit_metadata(self) -> dict:
        return {"fit": "metadata"}


class _FakeTrainer:
    """Trainer stand-in: ``models`` holds the resident objects, ``load_model`` falls back to a fresh object."""

    def __init__(self, models: dict, *, model_best: str | None = None):
        self.models = models
        self.model_best = model_best
        self.load_model_calls = 0

    def load_model(self, name: str):
        self.load_model_calls += 1
        if name in self.models:
            return self.models[name]
        return _PlainModel()


class _FakePredictor:
    """``TabularPredictor`` stand-in for the untimed inference bracket.

    ``persist`` returns ``persist_returns`` (or raises ``persist_raises``) and ``unpersist`` empties the
    trainer's resident models, the way AutoGluon's does. Every call is appended to ``calls``.
    """

    def __init__(
        self,
        *,
        persist_returns: list[str],
        trainer_models: dict,
        path: str | None = None,
        model_best: str | None = None,
        persist_raises: Exception | None = None,
    ):
        self._persist_returns = persist_returns
        self._persist_raises = persist_raises
        self._trainer = _FakeTrainer(trainer_models, model_best=model_best)
        self.model_best = model_best
        self.path = path
        self.calls: list[tuple] = []

    def persist(self, models, max_memory=0.4):
        self.calls.append(("persist", models, max_memory))
        if self._persist_raises is not None:
            raise self._persist_raises
        return list(self._persist_returns)

    def unpersist(self):
        self.calls.append(("unpersist",))
        self._trainer.models.clear()

    def model_names(self, can_infer: bool = False) -> list[str]:
        if self.model_best is not None:
            return [self.model_best]
        return list(self._trainer.models)
