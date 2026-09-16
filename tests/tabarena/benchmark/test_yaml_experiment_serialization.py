from __future__ import annotations

import io
import os

from tabarena.benchmark.experiment.experiment_constructor import (
    AGModelBagExperiment,
    AGModelExperiment,
    AGModelOuterExperiment,
    YamlExperimentSerializer,
    YamlSingleExperimentSerializer,
)
from tabarena.benchmark.validation_protocol import (
    BEYONDARENA_VALIDATION_PROTOCOL,
    TABARENA_V0PT1_VALIDATION_PROTOCOL,
    ValidationProtocol,
)
from tabarena.models.realmlp.hpo import gen_realmlp
from tabarena.models.realmlp.model import RealMLPModel


def _as_str_path(p: str | os.PathLike) -> str:
    return os.fspath(p)


def _init_memory_fs(monkeypatch):
    # --- Tiny in-memory filesystem just for this test ---
    fs: dict[str, str] = {}

    def mem_exists(path):
        path = _as_str_path(path)
        return path in fs

    def mem_open(path, mode="r", *args, **kwargs):
        path = _as_str_path(path)
        # Text-mode only (YAML). If your serializers use 'b', handle BytesIO similarly.
        if "w" in mode:
            buf = io.StringIO()
            _orig_close = buf.close

            def _close_and_persist():
                fs[path] = buf.getvalue()
                _orig_close()

            buf.close = _close_and_persist  # type: ignore[assignment]
            return buf
        if "r" in mode:
            if path not in fs:
                raise FileNotFoundError(path)
            return io.StringIO(fs[path])
        raise ValueError(f"Unsupported mode: {mode}")

    # Patch builtins.open and os.path.exists so the serializers think the file is there.
    monkeypatch.setattr("builtins.open", mem_open, raising=True)
    monkeypatch.setattr("os.path.exists", mem_exists, raising=True)


def test_yaml_experiment_serialization(monkeypatch):
    """Verify that saving and loading experiments to/from yaml results in no changes to the object."""
    # patch so no file is created on disk
    _init_memory_fs(monkeypatch=monkeypatch)

    num_random_configs = 3
    experiments_realmlp = gen_realmlp.generate_all_bag_experiments(num_random_configs=num_random_configs)
    assert len(experiments_realmlp) == num_random_configs + 1
    experiment_default: AGModelExperiment = experiments_realmlp[0]
    assert experiment_default.method_kwargs["model_cls"] == RealMLPModel

    yaml_path = "tmp.yaml"
    experiment_default.to_yaml(path=yaml_path)

    experiment_loaded = YamlSingleExperimentSerializer.from_yaml(path=yaml_path)

    assert experiment_default.__class__ == experiment_loaded.__class__
    assert experiment_default.__dict__ == experiment_loaded.__dict__

    YamlExperimentSerializer.to_yaml(experiments=[experiment_default], path=yaml_path)
    experiments_loaded = YamlExperimentSerializer.from_yaml(path=yaml_path)

    for cur_exp, cur_exp_loaded in zip(experiments_realmlp, experiments_loaded, strict=False):
        assert cur_exp.__class__ == cur_exp_loaded.__class__
        assert cur_exp.__dict__ == cur_exp_loaded.__dict__


def test_yaml_round_trip_keeps_a_stamped_validation_protocol(monkeypatch):
    """A protocol stamped by an arena context (labels included) survives the YAML round trip."""
    _init_memory_fs(monkeypatch=monkeypatch)
    (experiment,) = gen_realmlp.generate_all_bag_experiments(num_random_configs=0)
    experiment.set_validation_protocol(BEYONDARENA_VALIDATION_PROTOCOL.with_origin(arena="BeyondArena", enforced=True))

    yaml_path = "tmp_stamped.yaml"
    experiment.to_yaml(path=yaml_path)
    loaded = YamlSingleExperimentSerializer.from_yaml(path=yaml_path)

    assert loaded.validation_protocol == BEYONDARENA_VALIDATION_PROTOCOL
    assert loaded.validation_protocol.arena == "BeyondArena"
    assert loaded.validation_protocol.enforced is True
    assert loaded.__dict__ == experiment.__dict__


# An ``experiments.yaml`` written before the protocol object existed: counts and the task-specific
# flag as separate knobs (ints on the first entry, ``auto`` plus the class adaptation on the second),
# and the flag on an outer experiment that never had inner validation.
LEGACY_YAML = """
methods:
- type: tabarena.benchmark.experiment.experiment_constructor.AGModelBagExperiment
  name: LightGBM_c1_BAG_L1
  model_cls: autogluon.tabular.models.lgb.lgb_model.LGBModel
  model_hyperparameters:
    ag_args:
      name_suffix: _c1
  num_bag_folds: 8
  num_bag_sets: 1
  time_limit: 3600
  dynamic_tabarena_validation_protocol: false
  method_kwargs:
    fit_kwargs:
      num_cpus: 8
- type: tabarena.benchmark.experiment.experiment_constructor.AGModelBagExperiment
  name: RealMLP_c1_BAG_L1
  model_cls: tabarena.models.realmlp.model.RealMLPModel
  model_hyperparameters: {}
  num_bag_folds: auto
  num_bag_sets: auto
  dynamic_tabarena_validation_protocol: true
  method_kwargs:
    fit_kwargs:
      adapt_num_bag_folds_to_n_classes: true
- type: tabarena.benchmark.experiment.experiment_constructor.AGModelOuterExperiment
  name: LightGBM_c1
  model_cls: autogluon.tabular.models.lgb.lgb_model.LGBModel
  model_hyperparameters: {}
  dynamic_tabarena_validation_protocol: false
"""


def test_legacy_yaml_loads_with_a_derived_protocol():
    """Pre-protocol YAML still loads: the knobs become one ``legacy`` protocol per experiment."""
    tabarena_like, beyondarena_like, outer = YamlExperimentSerializer.from_yaml_str(LEGACY_YAML)

    assert isinstance(tabarena_like, AGModelBagExperiment)
    assert tabarena_like.validation_protocol == TABARENA_V0PT1_VALIDATION_PROTOCOL
    assert tabarena_like.validation_protocol.name == "legacy"
    assert tabarena_like.method_kwargs["fit_kwargs"]["num_cpus"] == 8
    assert "num_bag_folds" not in tabarena_like.method_kwargs["fit_kwargs"]

    assert beyondarena_like.validation_protocol == BEYONDARENA_VALIDATION_PROTOCOL
    assert "adapt_num_bag_folds_to_n_classes" not in beyondarena_like.method_kwargs["fit_kwargs"]

    assert isinstance(outer, AGModelOuterExperiment)
    assert outer.validation_protocol is None

    # The migrated experiments serialize in the new shape and round-trip unchanged.
    reloaded = YamlExperimentSerializer.from_yaml_str(YamlExperimentSerializer.to_yaml_str([tabarena_like]))[0]
    assert reloaded.validation_protocol == ValidationProtocol()
    assert reloaded.__dict__ == tabarena_like.__dict__
