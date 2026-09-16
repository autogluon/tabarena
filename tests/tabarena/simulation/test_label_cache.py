from __future__ import annotations

import numpy as np
import pandas as pd

from tabarena.repository import EvaluationRepository
from tabarena.simulation.context_artificial import load_repo_artificial
from tabarena.simulation.label_cache import (
    LabelFileCache,
    get_active_label_cache,
    shared_label_files,
    zip_entry_signature,
)


def _write(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def test_label_file_cache_reuses_identical_files(tmp_path):
    labels = pd.DataFrame({"y": [1, 0, 1, 1]}, index=[3, 5, 8, 9])
    other = pd.DataFrame({"y": [1, 0, 0, 1]}, index=[3, 5, 8, 9])
    _write(tmp_path / "a" / "label-val.csv.zip", labels)
    _write(tmp_path / "b" / "label-val.csv.zip", labels)
    _write(tmp_path / "c" / "label-val.csv.zip", other)
    _write(tmp_path / "d" / "label-val.csv", labels)
    assert zip_entry_signature(tmp_path / "a" / "label-val.csv.zip") is not None
    assert zip_entry_signature(tmp_path / "d" / "label-val.csv") is None

    cache = LabelFileCache()
    first = cache.read(tmp_path / "a" / "label-val.csv.zip", dataset="ds", fold=0, split="val")
    second = cache.read(tmp_path / "b" / "label-val.csv.zip", dataset="ds", fold=0, split="val")
    assert second is first
    assert first.equals(labels)
    # same key, different content: parsed fresh, and the first entry stays
    third = cache.read(tmp_path / "c" / "label-val.csv.zip", dataset="ds", fold=0, split="val")
    assert third is not first
    assert third.equals(other)
    assert cache.read(tmp_path / "a" / "label-val.csv.zip", dataset="ds", fold=0, split="val") is first
    # other key or split: no reuse
    assert cache.read(tmp_path / "a" / "label-val.csv.zip", dataset="ds", fold=1, split="val") is not first
    assert cache.read(tmp_path / "a" / "label-val.csv.zip", dataset="ds", fold=0, split="test") is not first
    # non-zip files bypass the cache
    plain = cache.read(tmp_path / "d" / "label-val.csv", dataset="ds", fold=0, split="val")
    assert plain is not first
    assert plain.equals(labels)
    assert cache.hits == 2


def test_shared_label_files_across_repos(tmp_path):
    repo = load_repo_artificial()
    repo_1 = repo.subset(configs=["NeuralNetFastAI_r1"])
    repo_2 = repo.subset(configs=["NeuralNetFastAI_r2"])
    repo_1.to_dir(tmp_path / "m1")
    repo_2.to_dir(tmp_path / "m2")

    assert get_active_label_cache() is None
    with shared_label_files() as cache:
        assert get_active_label_cache() is cache
        loaded_1 = EvaluationRepository.from_dir(tmp_path / "m1", verbose=False)
        loaded_2 = EvaluationRepository.from_dir(tmp_path / "m2", verbose=False)
    assert get_active_label_cache() is None
    assert cache.misses == 2 * len(repo.tasks())
    assert cache.hits == 2 * len(repo.tasks())
    for dataset, fold in repo.tasks():
        assert (
            loaded_1._ground_truth._label_val_dict[dataset][fold]
            is loaded_2._ground_truth._label_val_dict[dataset][fold]
        )
        assert (
            loaded_1._ground_truth._label_test_dict[dataset][fold]
            is loaded_2._ground_truth._label_test_dict[dataset][fold]
        )
        assert np.array_equal(
            loaded_1.labels_val(dataset=dataset, fold=fold), repo.labels_val(dataset=dataset, fold=fold)
        )

    # without the block, each repo parses its own copy
    plain_1 = EvaluationRepository.from_dir(tmp_path / "m1", verbose=False)
    plain_2 = EvaluationRepository.from_dir(tmp_path / "m2", verbose=False)
    dataset, fold = repo.tasks()[0]
    assert (
        plain_1._ground_truth._label_val_dict[dataset][fold] is not plain_2._ground_truth._label_val_dict[dataset][fold]
    )
