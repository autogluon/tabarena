from __future__ import annotations

from types import SimpleNamespace

import pytest

from tabarena.benchmark.task.openml.task_utils import use_cached_pickle


def _dataset(tmp_path, *, parquet=True, cache_format="pickle", pickle_file=None):
    parquet_file = tmp_path / "dataset_1.pq"
    parquet_file.write_bytes(b"")
    return SimpleNamespace(
        cache_format=cache_format,
        data_pickle_file=pickle_file,
        parquet_file=str(parquet_file) if parquet else None,
        data_file=None,
    )


def test_use_cached_pickle_registers_existing_pickle(tmp_path):
    pickle_path = tmp_path / "dataset_1.pkl.py3"
    pickle_path.write_bytes(b"")
    dataset = _dataset(tmp_path)

    use_cached_pickle(dataset)

    assert dataset.data_pickle_file == str(pickle_path)


def test_use_cached_pickle_leaves_missing_pickle_unset(tmp_path):
    dataset = _dataset(tmp_path)

    use_cached_pickle(dataset)

    assert dataset.data_pickle_file is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cache_format": "feather"},
        {"pickle_file": "/already/set.pkl.py3"},
        {"parquet": False},
    ],
)
def test_use_cached_pickle_is_a_no_op_outside_the_parquet_pickle_case(tmp_path, kwargs):
    (tmp_path / "dataset_1.pkl.py3").write_bytes(b"")
    dataset = _dataset(tmp_path, **kwargs)
    before = dataset.data_pickle_file

    use_cached_pickle(dataset)

    assert dataset.data_pickle_file == before
