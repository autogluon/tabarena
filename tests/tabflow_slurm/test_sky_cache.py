"""Tests for the seeded dataset cache (`tabflow_slurm.setup.sky_cache`), with the bucket replaced by a directory."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("tabflow_slurm.setup", reason="tabflow_slurm is not installed")

from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabflow_slurm.setup.sky_cache import (
    CacheEntry,
    CacheSeedReport,
    collect_cache_entries,
    collect_weight_entries,
    dataset_id_from_task_xml,
    is_portable_user_task,
    openml_task_entries,
    seed_dataset_cache,
    seed_model_weights,
    upgrade_legacy_user_task,
    user_task_entries,
    weight_cache_env,
)

_TASK_XML = """<oml:task xmlns:oml="http://openml.org/openml"><oml:task_id>363612</oml:task_id>
<oml:input name="source_data"><oml:data_set><oml:data_set_id>46904</oml:data_set_id></oml:data_set></oml:input></oml:task>"""


def _openml_root(tmp_path: Path, tid: int = 363612, did: int = 46904) -> Path:
    root = tmp_path / "openml"
    task_dir = root / "org/openml/www/tasks" / str(tid)
    task_dir.mkdir(parents=True)
    (task_dir / "task.xml").write_text(_TASK_XML)
    (task_dir / "datasplits.arff").write_text("splits")
    dataset_dir = root / "org/openml/www/datasets" / str(did)
    dataset_dir.mkdir(parents=True)
    for name in ("description.xml", "features.xml", f"dataset_{did}.pq"):
        (dataset_dir / name).write_text(name)
    return root


def _collection(tid: int = 363612, dataset: str = "anneal") -> TaskMetadataCollection:
    df = pd.DataFrame(
        {
            "tid": [tid],
            "dataset": [dataset],
            "problem_type": ["binary"],
            "n_folds": [1],
            "n_repeats": [1],
            "n_features": [5],
            "n_classes": [2],
            "NumberOfInstances": [100],
            "n_samples_train_per_fold": [80.0],
            "n_samples_test_per_fold": [20.0],
        }
    )
    return TaskMetadataCollection.from_legacy_df(df)


class TestEntries:
    def test_dataset_id_from_task_xml(self, tmp_path):
        path = tmp_path / "task.xml"
        path.write_text(_TASK_XML)
        assert dataset_id_from_task_xml(path) == 46904

    def test_openml_task_entries_name_the_task_and_dataset_dirs(self, tmp_path):
        root = _openml_root(tmp_path)
        entries = openml_task_entries(363612, openml_root=root)
        assert [(e.rel, e.is_dir) for e in entries] == [
            ("openml/org/openml/www/tasks/363612", True),
            ("openml/org/openml/www/datasets/46904", True),
        ]
        assert entries[1].local == root / "org/openml/www/datasets/46904"

    def test_a_task_not_cached_locally_has_no_entries(self, tmp_path):
        assert openml_task_entries(1, openml_root=tmp_path / "empty") == []

    def test_user_task_entries_take_portable_pickles_and_their_text_cache(self, tmp_path):
        from tabarena.benchmark.preprocessing.text_cache import embedding_id
        from tabarena.benchmark.task.user_task import UserTask

        task = UserTask(task_name="airfoil/uuid-1")
        openml_root, tabarena_root = tmp_path / "openml", tmp_path / "tabarena"
        (openml_root / "tabarena_tasks").mkdir(parents=True)
        pickle.dump(
            {"format": "tabarena-user-task-v1", "dataset": None},
            (openml_root / "tabarena_tasks" / f"{task.slug}.pkl").open("wb"),
        )
        text_dir = tabarena_root / "text_cache" / embedding_id()
        text_dir.mkdir(parents=True)
        (text_dir / f"{task.slug}_cache.parquet").write_bytes(b"pq")
        entries = user_task_entries(task.task_id_str, openml_root=openml_root, tabarena_root=tabarena_root)
        assert [e.rel for e in entries] == [
            f"openml/tabarena_tasks/{task.slug}.pkl",
            f"tabarena/text_cache/{embedding_id()}/{task.slug}_cache.parquet",
        ]
        assert all(not e.is_dir for e in entries)

    def test_a_legacy_user_task_pickle_is_skipped_with_a_warning(self, tmp_path):
        from tabarena.benchmark.task.user_task import UserTask

        task = UserTask(task_name="mice/uuid-2")
        openml_root = tmp_path / "openml"
        (openml_root / "tabarena_tasks").mkdir(parents=True)
        path = openml_root / "tabarena_tasks" / f"{task.slug}.pkl"
        pickle.dump({"data_pickle_file": "/home/head/.cache/openml/local/datasets/abc/data.pkl.py3"}, path.open("wb"))
        assert is_portable_user_task(path) is False
        with pytest.warns(UserWarning, match="legacy task pickle"):
            assert user_task_entries(task.task_id_str, openml_root=openml_root, tabarena_root=tmp_path / "t") == []

    def test_collect_groups_entries_per_dataset(self, tmp_path):
        root = _openml_root(tmp_path)
        by_dataset = collect_cache_entries(_collection(), openml_root=root, tabarena_root=tmp_path / "tabarena")
        assert list(by_dataset) == ["anneal"]
        assert [e.rel for e in by_dataset["anneal"]] == [
            "openml/org/openml/www/tasks/363612",
            "openml/org/openml/www/datasets/46904",
        ]


class TestSeedDatasetCache:
    def test_uploads_missing_files_verifies_and_writes_the_manifest(self, tmp_path, local_storage):
        root = _openml_root(tmp_path)
        entries = collect_cache_entries(_collection(), openml_root=root, tabarena_root=tmp_path / "tabarena")
        manifest, report = seed_dataset_cache(entries, storage=local_storage, cache_uri="gs://b/tabarena/cache/")
        assert manifest == {
            "cache_uri": "gs://b/tabarena/cache",
            "datasets": {"anneal": ["openml/org/openml/www/tasks/363612", "openml/org/openml/www/datasets/46904"]},
        }
        assert (report.datasets, report.files, report.uploaded_files) == (1, 5, 5)
        assert report.unverified == [] and report.not_cached_locally == []
        assert local_storage.read_text("gs://b/tabarena/cache/openml/org/openml/www/tasks/363612/task.xml") == _TASK_XML
        assert "5 file(s) uploaded now" in report.summary()

        # Datasets are immutable: a second setup uploads nothing and verifies the same manifest.
        uploads_before = len(local_storage.uploads)
        manifest_again, report_again = seed_dataset_cache(
            entries, storage=local_storage, cache_uri="gs://b/tabarena/cache"
        )
        assert manifest_again == manifest
        assert report_again.uploaded_files == 0
        assert len(local_storage.uploads) == uploads_before

    def test_read_only_mode_only_verifies_and_leaves_missing_datasets_to_the_workers(self, tmp_path, local_storage):
        root = _openml_root(tmp_path)
        entries = collect_cache_entries(_collection(), openml_root=root, tabarena_root=tmp_path / "tabarena")
        manifest, report = seed_dataset_cache(
            entries, storage=local_storage, cache_uri="gs://curated/cache", upload=False
        )
        assert manifest["datasets"] == {}
        assert report.uploaded_files == 0
        assert report.unverified == ["openml/org/openml/www/tasks/363612", "openml/org/openml/www/datasets/46904"]
        assert "MISSING remotely" in report.summary()

    def test_a_dataset_without_local_entries_is_reported(self, local_storage):
        manifest, report = seed_dataset_cache({"anneal": []}, storage=local_storage, cache_uri="gs://b/c")
        assert manifest["datasets"] == {}
        assert report.not_cached_locally == ["anneal"]

    def test_single_file_entries_are_uploaded_once(self, tmp_path, local_storage):
        path = tmp_path / "slug.pkl"
        path.write_bytes(b"x")
        entries = {"ds": [CacheEntry(rel="openml/tabarena_tasks/slug.pkl", local=path, is_dir=False)]}
        seed_dataset_cache(entries, storage=local_storage, cache_uri="gs://b/c")
        _, report = seed_dataset_cache(entries, storage=local_storage, cache_uri="gs://b/c")
        assert local_storage.exists("gs://b/c/openml/tabarena_tasks/slug.pkl")
        assert report.uploaded_files == 0


class _UpgradableCollection:
    """A collection stand-in for one data-foundry dataset whose materialize writes a portable pickle
    into whatever OpenML root is active (as the real DataFoundryTaskMetadataSource does).
    """

    def __init__(self, task_id_str: str, dataset: str, *, portable: bool = True, preset: str | None = "BeyondArena"):
        import types

        self.preset = preset
        self._ttm = types.SimpleNamespace(
            task_id_str=task_id_str, tabarena_task_name=dataset, data_foundry_uri="x/uuid"
        )
        self.materialized = 0
        self.portable = portable

    def __iter__(self):
        return iter([self._ttm])

    def __len__(self):
        return 1

    def subset_tasks(self, *, dataset_names):
        assert dataset_names == [self._ttm.tabarena_task_name]
        return self

    def materialize(self):
        import openml

        self.materialized += 1
        root = Path(openml.config._root_cache_directory) / "tabarena_tasks"
        root.mkdir(parents=True, exist_ok=True)
        payload = {"format": "tabarena-user-task-v1", "dataset": None} if self.portable else {"legacy": True}
        pickle.dump(payload, (root / f"{self._ttm.tabarena_task_name}.pkl").open("wb"))
        return self


class TestLegacyUpgrade:
    @staticmethod
    def _legacy(tmp_path):
        from tabarena.benchmark.task.user_task import UserTask

        task = UserTask(task_name="mice/uuid-2")
        openml_root = tmp_path / "openml"
        (openml_root / "tabarena_tasks").mkdir(parents=True)
        pickle.dump(
            {"data_pickle_file": "/head/local/datasets/abc/data.pkl.py3"},
            (openml_root / "tabarena_tasks" / f"{task.slug}.pkl").open("wb"),
        )
        return task, openml_root

    def test_replaces_the_legacy_pickle_atomically_and_restores_the_openml_root(self, tmp_path):
        import openml

        task, openml_root = self._legacy(tmp_path)
        collection = _UpgradableCollection(task.task_id_str, task.slug)
        saved_root = openml.config._root_cache_directory
        assert upgrade_legacy_user_task(collection, task.slug, openml_root=openml_root) is True
        assert collection.materialized == 1
        assert openml.config._root_cache_directory == saved_root
        assert is_portable_user_task(openml_root / "tabarena_tasks" / f"{task.slug}.pkl")
        assert not list(openml_root.glob(".upgrade_*"))  # scratch root removed

    def test_no_preset_means_no_upgrade(self, tmp_path):
        task, openml_root = self._legacy(tmp_path)
        collection = _UpgradableCollection(task.task_id_str, task.slug, preset=None)
        assert upgrade_legacy_user_task(collection, task.slug, openml_root=openml_root) is False
        assert collection.materialized == 0
        assert not is_portable_user_task(openml_root / "tabarena_tasks" / f"{task.slug}.pkl")

    def test_a_failed_conversion_keeps_the_legacy_pickle(self, tmp_path):
        task, openml_root = self._legacy(tmp_path)
        collection = _UpgradableCollection(task.task_id_str, task.slug, portable=False)
        with pytest.warns(UserWarning, match="no portable task pickle"):
            assert upgrade_legacy_user_task(collection, task.slug, openml_root=openml_root) is False
        assert not is_portable_user_task(openml_root / "tabarena_tasks" / f"{task.slug}.pkl")

    def test_collect_upgrades_then_seeds_the_dataset(self, tmp_path):
        task, openml_root = self._legacy(tmp_path)
        collection = _UpgradableCollection(task.task_id_str, task.slug)
        by_dataset = collect_cache_entries(collection, openml_root=openml_root, tabarena_root=tmp_path / "t")
        assert [e.rel for e in by_dataset[task.slug]] == [f"openml/tabarena_tasks/{task.slug}.pkl"]
        assert collection.materialized == 1


def _fake_prefetch(models, *, python, cache_root):
    """Populate a scratch cache root like a real prefetch would: an HF repo with symlinked snapshots, a tabpfn ckpt."""
    repo = Path(cache_root) / "huggingface" / "hub" / "models--Prior-Labs--tabpfn_3"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "abc").write_bytes(b"weights")
    (repo / "snapshots" / "rev1").mkdir(parents=True)
    (repo / "snapshots" / "rev1" / "model.ckpt").symlink_to(repo / "blobs" / "abc")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("rev1")
    (repo / ".locks").mkdir()
    (repo / ".locks" / "x.lock").write_text("")
    (Path(cache_root) / "xdg" / "tabpfn").mkdir(parents=True)
    (Path(cache_root) / "xdg" / "tabpfn" / "tabpfn-v3.ckpt").write_bytes(b"ckpt")
    (Path(cache_root) / "xdg" / "tabpfn" / ".download.lock").write_text("")


class TestWeights:
    def test_weight_cache_env_points_every_cache_under_the_root(self, tmp_path):
        assert weight_cache_env(tmp_path) == {
            "HF_HOME": str(tmp_path / "huggingface"),
            "XDG_CACHE_HOME": str(tmp_path / "xdg"),
            "TORCH_HOME": str(tmp_path / "torch"),
        }

    def test_collect_weight_entries_takes_snapshots_refs_and_plain_files(self, tmp_path):
        _fake_prefetch(["TabPFN-3"], python="x", cache_root=tmp_path)
        rels = [(e.rel, e.is_dir) for e in collect_weight_entries(tmp_path)]
        assert rels == [
            ("huggingface/hub/models--Prior-Labs--tabpfn_3/snapshots", True),
            ("huggingface/hub/models--Prior-Labs--tabpfn_3/refs", True),
            ("xdg/tabpfn/tabpfn-v3.ckpt", False),
        ]

    def test_seed_model_weights_prefetches_once_and_reuses_the_remote_manifest(self, tmp_path, local_storage):
        calls: list = []

        def prefetch(models, *, python, cache_root):
            calls.append(list(models))
            _fake_prefetch(models, python=python, cache_root=cache_root)

        kwargs = {
            "python": "/venv/bin/python",
            "storage": local_storage,
            "cache_uri": "gs://b/tabarena/cache",
            "scratch_dir": tmp_path / "scratch",
            "prefetch": prefetch,
            "has_prefetcher": lambda m: m == "TabPFN-3",
        }
        manifest, report = seed_model_weights(["TabPFN-3", "Linear"], **kwargs)
        assert calls == [["TabPFN-3"]]
        assert manifest == {
            "weights": {
                "TabPFN-3": [
                    "huggingface/hub/models--Prior-Labs--tabpfn_3/snapshots",
                    "huggingface/hub/models--Prior-Labs--tabpfn_3/refs",
                    "xdg/tabpfn/tabpfn-v3.ckpt",
                ],
                "Linear": [],
            },
            "offline_weights": True,
        }
        # Symlinked snapshot files were uploaded as real content; the per-model manifest was written.
        assert (
            local_storage.read_text(
                "gs://b/tabarena/cache/huggingface/hub/models--Prior-Labs--tabpfn_3/snapshots/rev1/model.ckpt"
            )
            == "weights"
        )
        assert local_storage.exists("gs://b/tabarena/cache/weights/TabPFN-3.json")
        assert report.uploaded_files == 3 and not list((tmp_path / "scratch").glob("*"))
        # A second setup takes the remote manifest and never prefetches again.
        manifest_again, report_again = seed_model_weights(["TabPFN-3"], **kwargs)
        assert calls == [["TabPFN-3"]]
        assert manifest_again["weights"]["TabPFN-3"] == manifest["weights"]["TabPFN-3"]
        assert report_again.uploaded_files == 0

    def test_read_only_mode_leaves_unseeded_weights_to_the_workers(self, tmp_path, local_storage):
        manifest, report = seed_model_weights(
            ["TabPFN-3"],
            python="x",
            storage=local_storage,
            cache_uri="gs://curated",
            upload=False,
            scratch_dir=tmp_path,
            prefetch=lambda *a, **k: pytest.fail("must not prefetch"),
            has_prefetcher=lambda m: True,
        )
        assert manifest == {"weights": {}, "offline_weights": False}
        assert report.unverified == ["weights of TabPFN-3"]
        assert isinstance(report, CacheSeedReport)
