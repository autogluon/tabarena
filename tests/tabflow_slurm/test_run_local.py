"""The local runner keeps several items in flight for remote-compute systems and logs each one separately."""

from __future__ import annotations

import json
import sys
import time

import pytest

pytest.importorskip("tabflow_slurm.run_local", reason="tabflow_slurm is not installed")

from tabflow_slurm import run_local


def _write_jobs(tmp_path, n_items: int):
    items = [{"experiment": "Sys_c1", "dataset": f"d{i}", "fold": 0, "repeat": 0} for i in range(n_items)]
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"defaults": {}, "jobs": [{"items": items[:2]}, {"items": items[2:]}]}))
    return path


@pytest.fixture
def fake_items(monkeypatch):
    """Each item sleeps 0.5 s and prints its dataset; dataset ``d1`` exits 3."""

    def command(defaults, item):
        code = 3 if item["dataset"] == "d1" else 0
        return [
            sys.executable,
            "-c",
            f"import time; time.sleep(0.5); print('ran {item['dataset']}'); raise SystemExit({code})",
        ]

    monkeypatch.setattr(run_local, "_build_item_command", command)


def test_parallel_items_run_concurrently_and_log_separately(tmp_path, fake_items, capsys):
    json_path = _write_jobs(tmp_path, 6)
    start = time.monotonic()
    code = run_local.run(str(json_path), continue_on_error=True, num_workers=3, item_log_dir=str(tmp_path / "logs"))
    elapsed = time.monotonic() - start

    assert code == 1  # d1 failed
    assert elapsed < 6 * 0.5  # sequential would take at least 3 s
    logs = sorted((tmp_path / "logs").glob("*.log"))
    assert len(logs) == 6
    assert "ran d4" in next(p for p in logs if "d4" in p.name).read_text()
    out = capsys.readouterr().out
    assert "with 3 in flight" in out
    assert "5/6 succeeded, 1 failed" in out


def test_parallel_stops_starting_items_after_a_failure(tmp_path, fake_items, capsys):
    json_path = _write_jobs(tmp_path, 6)
    code = run_local.run(str(json_path), continue_on_error=False, num_workers=2, item_log_dir=str(tmp_path / "logs"))
    assert code == 1
    assert len(list((tmp_path / "logs").glob("*.log"))) < 6
    assert "Stopped starting new items" in capsys.readouterr().out


def test_parallel_needs_subprocess_mode(tmp_path):
    with pytest.raises(ValueError, match="subprocess"):
        run_local.run(str(_write_jobs(tmp_path, 2)), continue_on_error=True, execution_mode="in_process", num_workers=2)
