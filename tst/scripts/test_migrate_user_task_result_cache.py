"""Tests for the UserTask result-cache migration maintainer script.

The migration logic lives in ``scripts/migrate_user_task_result_cache.py`` (not in the
``tabarena`` package), so we load it by path.
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pytest
from tabarena.benchmark.task.user_task import UserTask
from tabarena.utils.pickle_utils import is_gzipped, load_pickle

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_user_task_result_cache.py"
_spec = importlib.util.spec_from_file_location("migrate_user_task_result_cache", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
migrate_user_task_result_cache = _mod.migrate_user_task_result_cache


def _make_result(out: Path, model: str, task_key: str, *, tid: int | None = None, name: str | None = None) -> Path:
    """Create a result pkl mimicking the stored shape (task_metadata with tid/name)."""
    d = out / "data" / model / task_key / "r0f0"
    d.mkdir(parents=True, exist_ok=True)
    pkl = d / "results.pkl"
    with pkl.open("wb") as f:
        pickle.dump({"task_metadata": {"tid": tid if tid is not None else int(task_key), "name": name}}, f)
    return pkl


def test_copy_default_writes_dest_and_leaves_source(tmp_path):
    """Default (copy) writes migrated results to dest and leaves the source untouched."""
    src, dest = tmp_path / "out", tmp_path / "dest"
    ut = UserTask(task_name="blood_transfusion/uuid-123")
    old_key, slug = str(ut.task_id), ut.slug
    src_pkl = _make_result(src, "ModelA", old_key, name=f"Task-{old_key}")

    migrate_user_task_result_cache(output_dir=src, task_id_strs=[ut.task_id_str], dest_dir=dest, num_ray_cpus=1)

    moved = dest / "data" / "ModelA" / slug / "r0f0" / "results.pkl"
    assert is_gzipped(moved.read_bytes())  # compressed by default
    tm = load_pickle(moved)["task_metadata"]
    assert tm["name"] == slug and tm["tid"] == ut.task_id
    # source untouched
    assert src_pkl.exists() and not is_gzipped(src_pkl.read_bytes())
    assert load_pickle(src_pkl)["task_metadata"]["name"] == f"Task-{old_key}"
    assert not (src / "data" / "ModelA" / slug).exists()


def test_in_place_renames_source(tmp_path):
    src = tmp_path / "out"
    ut = UserTask(task_name="blood_transfusion/uuid-123")
    old_key, slug = str(ut.task_id), ut.slug
    _make_result(src, "ModelA", old_key, name=f"Task-{old_key}")

    migrate_user_task_result_cache(output_dir=src, task_id_strs=[ut.task_id_str], in_place=True, num_ray_cpus=1)

    moved = src / "data" / "ModelA" / slug / "r0f0" / "results.pkl"
    assert is_gzipped(moved.read_bytes())
    assert load_pickle(moved)["task_metadata"]["name"] == slug
    assert not (src / "data" / "ModelA" / old_key).exists()  # original renamed away


def test_default_dest_is_sibling_of_output_dir():
    assert _mod._default_dest_dir(Path("/x/output/bench")) == Path("/x/output/bench_migrated")


def test_copy_no_compress_keeps_raw(tmp_path):
    src, dest = tmp_path / "out", tmp_path / "dest"
    ut = UserTask(task_name="ds/uuid")
    _make_result(src, "ModelA", str(ut.task_id), name=f"Task-{ut.task_id}")

    migrate_user_task_result_cache(
        output_dir=src, task_id_strs=[ut.task_id_str], dest_dir=dest, compress=False, num_ray_cpus=1
    )

    moved = dest / "data" / "ModelA" / ut.slug / "r0f0" / "results.pkl"
    assert not is_gzipped(moved.read_bytes())
    assert load_pickle(moved)["task_metadata"]["name"] == ut.slug


def test_copy_dry_run_writes_nothing(tmp_path):
    src, dest = tmp_path / "out", tmp_path / "dest"
    ut = UserTask(task_name="ds/uuid")
    src_pkl = _make_result(src, "ModelA", str(ut.task_id), name=f"Task-{ut.task_id}")

    migrated = migrate_user_task_result_cache(
        output_dir=src, task_id_strs=[ut.task_id_str], dest_dir=dest, dry_run=True, num_ray_cpus=1
    )

    assert len(migrated) == 1
    assert not dest.exists()  # nothing written
    assert load_pickle(src_pkl)["task_metadata"]["name"] == f"Task-{ut.task_id}"  # source untouched


def test_ignores_non_user_task_ids(tmp_path):
    src, dest = tmp_path / "out", tmp_path / "dest"
    _make_result(src, "ModelA", "363612")  # plain OpenML integer id dir

    migrated = migrate_user_task_result_cache(
        output_dir=src, task_id_strs=["363612"], dest_dir=dest, num_ray_cpus=1
    )

    assert migrated == []
    assert not dest.exists()
    assert (src / "data" / "ModelA" / "363612").exists()  # left as-is


def test_parallel_with_ray(tmp_path):
    """The Ray path (num_ray_cpus>1, multiple units) produces the same result as sequential."""
    pytest.importorskip("ray")
    src, dest = tmp_path / "out", tmp_path / "dest"
    uts = [UserTask(task_name=f"ds_{i}/uuid-{i}") for i in range(3)]
    for ut in uts:
        _make_result(src, "ModelA", str(ut.task_id), name=f"Task-{ut.task_id}")

    migrated = migrate_user_task_result_cache(
        output_dir=src, task_id_strs=[ut.task_id_str for ut in uts], dest_dir=dest, num_ray_cpus=2
    )

    assert len(migrated) == 3
    for ut in uts:
        moved = dest / "data" / "ModelA" / ut.slug / "r0f0" / "results.pkl"
        assert is_gzipped(moved.read_bytes())
        assert load_pickle(moved)["task_metadata"]["name"] == ut.slug
