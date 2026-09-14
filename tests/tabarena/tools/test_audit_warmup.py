from __future__ import annotations

import pickle

from tabarena.tools.audit_warmup import summarize_results


def test_summarize_results_reads_blocks_with_get(tmp_path):
    """Results written before the blocks existed count as missing; newer ones are aggregated per method."""
    old = {"framework": "M", "experiment_metadata": {"time_start": 0.0}}
    new = {
        "framework": "M",
        "experiment_metadata": {
            "time_warmup_s": 2.0,
            "warmup_report": {"status": "partial", "failed_steps": ["import:x:failed:ModuleNotFoundError"]},
            "timing_audit": {
                "fit": {"new_packages": ["x"], "cuda_initialized_before": False, "cuda_initialized_after": True},
                "predict": {"new_packages": [], "cuda_initialized_before": True, "cuda_initialized_after": True},
            },
        },
    }
    paths = []
    for i, result in enumerate([old, new]):
        path = tmp_path / f"t{i}" / "results.pkl"
        path.parent.mkdir()
        path.write_bytes(pickle.dumps(result))
        paths.append(path)

    summary = summarize_results(paths)["M"]

    assert summary["n"] == 2
    assert summary["missing_warmup_report"] == 1
    assert summary["missing_timing_audit"] == 1
    assert summary["warmup_status"] == {"partial": 1}
    assert summary["failed_steps"] == {"import:x:failed:ModuleNotFoundError": 1}
    assert summary["time_warmup_s_mean"] == 2.0
    assert summary["fit"] == {"new_packages": {"x": 1}, "cuda_initialized_inside": 1}
    assert summary["predict"] == {"new_packages": {}, "cuda_initialized_inside": 0}
