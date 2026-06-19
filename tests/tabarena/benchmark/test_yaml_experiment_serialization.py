from __future__ import annotations

import io
import os

from tabarena.benchmark.experiment.experiment_constructor import (
    AGModelExperiment,
    YamlExperimentSerializer,
    YamlSingleExperimentSerializer,
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
