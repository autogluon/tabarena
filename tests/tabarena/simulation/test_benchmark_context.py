from __future__ import annotations

import json

import pytest

from tabarena.repository import EvaluationRepository
from tabarena.simulation.benchmark_context import BenchmarkContext
from tabarena.simulation.context_artificial import load_repo_artificial


def test_from_json_drops_legacy_download_map(tmp_path):
    repo = load_repo_artificial()
    repo.to_dir(tmp_path)
    path = tmp_path / "context.json"
    with open(path) as f:
        payload = json.load(f)
    assert "s3_download_map" not in payload
    payload["s3_download_map"] = None  # what contexts written by earlier versions contain
    with open(path, "w") as f:
        json.dump(payload, f)
    context = BenchmarkContext.from_json(str(path))
    assert not hasattr(context, "s3_download_map")
    assert sorted(EvaluationRepository.from_dir(tmp_path, verbose=False).tasks()) == sorted(repo.tasks())


def test_load_lists_missing_files(tmp_path):
    repo = load_repo_artificial()
    repo.to_dir(tmp_path)
    dataset, fold = repo.tasks()[0]
    removed = tmp_path / "model_predictions" / dataset / str(fold) / "pred-val.dat"
    removed.unlink()
    with pytest.raises(FileNotFoundError, match="Missing 1 required files") as excinfo:
        EvaluationRepository.from_dir(tmp_path, verbose=False)
    assert str(removed) in str(excinfo.value)
