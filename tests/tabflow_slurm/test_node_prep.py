from __future__ import annotations

import pytest

pytest.importorskip("tabflow_slurm.node_prep", reason="tabflow_slurm is not installed")

from tabflow_slurm.node_prep import main, pretouch


def test_pretouch_reads_within_budget_writes_marker_and_skips_when_fresh(tmp_path, monkeypatch, capsys):
    pkg = tmp_path / "site" / "fakepretouchpkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "big.bin").write_bytes(b"x" * 1000)
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    marker = tmp_path / "jit" / "pretouch.done"

    read = pretouch(
        packages=["fakepretouchpkg", "no_such_package_xyz"], paths=["no/such/path"], max_bytes=500, marker=marker
    )
    assert read == 500  # the budget stops the read
    assert marker.exists()

    assert pretouch(packages=["fakepretouchpkg"], paths=[], max_bytes=500, marker=marker) == 0
    assert "skipped" in capsys.readouterr().out

    # The CLI never fails the job, even on a bad argument value.
    assert main(["pretouch", "--packages", "fakepretouchpkg", "--max-bytes", "10"]) == 0
