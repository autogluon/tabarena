"""Tests for the UserTask metadata-CSV migration maintainer script.

The migration logic lives in ``scripts/migrate_user_task_metadata_csv.py`` (not in the
``tabarena`` package), so we load it by path.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest
from tabarena.benchmark.task.user_task import UserTask

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_user_task_metadata_csv.py"
_spec = importlib.util.spec_from_file_location("migrate_user_task_metadata_csv", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
migrate_user_task_metadata_csv = _mod.migrate_user_task_metadata_csv


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_rewrites_user_task_names_and_preserves_other_columns(tmp_path):
    ut = UserTask(task_name="blood_transfusion/uuid-123")
    src, dst = tmp_path / "meta.csv", tmp_path / "meta_migrated.csv"
    _write_csv(
        src,
        [
            {
                "tabarena_task_name": f"Task-{ut.task_id}",
                "task_id_str": ut.task_id_str,
                "num_instances": "748",  # preserved verbatim (no numeric reformatting)
            }
        ],
    )

    n_updated = migrate_user_task_metadata_csv(src_csv=src, dst_csv=dst)

    assert n_updated == 1
    out = _read_csv(dst)
    assert out[0]["tabarena_task_name"] == ut.slug
    assert out[0]["num_instances"] == "748"
    # source untouched
    assert _read_csv(src)[0]["tabarena_task_name"] == f"Task-{ut.task_id}"


def test_leaves_non_user_task_rows_unchanged(tmp_path):
    src, dst = tmp_path / "meta.csv", tmp_path / "meta_migrated.csv"
    _write_csv(src, [{"tabarena_task_name": "some-openml-task", "task_id_str": "363612"}])

    n_updated = migrate_user_task_metadata_csv(src_csv=src, dst_csv=dst)

    assert n_updated == 0
    assert _read_csv(dst)[0]["tabarena_task_name"] == "some-openml-task"


def test_in_place_overwrites_source(tmp_path):
    ut = UserTask(task_name="ds/uuid")
    src = tmp_path / "meta.csv"
    _write_csv(src, [{"tabarena_task_name": f"Task-{ut.task_id}", "task_id_str": ut.task_id_str}])

    n_updated = migrate_user_task_metadata_csv(src_csv=src, dst_csv=src)

    assert n_updated == 1
    assert _read_csv(src)[0]["tabarena_task_name"] == ut.slug
    assert not src.with_name(src.name + ".tmp").exists()  # temp file cleaned up


def test_missing_required_column_raises(tmp_path):
    src, dst = tmp_path / "meta.csv", tmp_path / "meta_migrated.csv"
    _write_csv(src, [{"task_id_str": "x", "other": "y"}])  # no tabarena_task_name

    with pytest.raises(ValueError, match="must contain columns"):
        migrate_user_task_metadata_csv(src_csv=src, dst_csv=dst)
