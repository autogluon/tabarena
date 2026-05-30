"""Tests for `tabarena.evaluation.benchmark_eval`.

The heavy engines (Ray post-processing, OpenML fetch, TabArena context) are
monkeypatched, so these only exercise the orchestration + the pure helpers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from tabarena.evaluation import EvalMethod, TabArenaEvalConfig, run_eval


def _config(tmp_path: Path, **kwargs) -> TabArenaEvalConfig:
    defaults = {
        "benchmark_name": "bench",
        "output_dir": tmp_path / "out" / "bench",
        "methods": [EvalMethod("RandomForest", ag_name_override="RF")],
        "figure_output_dir": tmp_path / "figs",
    }
    defaults.update(kwargs)
    return TabArenaEvalConfig(**defaults)


class TestEvalMethod:
    def test_ag_name_override_wins(self):
        assert EvalMethod("RandomForest", ag_name_override="RF").ag_name == "RF"


class TestConfig:
    def test_path_raw_is_output_dir_data(self):
        cfg = _config(Path("/base"), output_dir="/x/out/bench")
        assert cfg.path_raw == Path("/x/out/bench/data")

    def test_subsets_default_is_full(self, tmp_path):
        assert _config(tmp_path).subsets_to_run() == [[]]

    def test_subsets_passthrough(self, tmp_path):
        assert _config(tmp_path, subsets=[[], ["regression"]]).subsets_to_run() == [[], ["regression"]]

    def test_init_caches_sets_tabarena_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TABARENA_CACHE", raising=False)
        _config(tmp_path, tabarena_cache_path="/c").init_caches()
        assert os.environ["TABARENA_CACHE"] == "/c"


def test_run_eval_orchestration(tmp_path, monkeypatch):
    import tabarena.nips2025_utils.end_to_end as ee
    import tabarena.nips2025_utils.end_to_end_single as ees
    import tabarena.website.website_format as wf

    post_calls: list[dict] = []
    cache_calls: list[dict] = []
    monkeypatch.setattr(
        ees.EndToEndSingle,
        "from_path_raw_to_results",
        staticmethod(lambda **kw: post_calls.append(kw) or "single"),
    )
    monkeypatch.setattr(
        ees.EndToEndSingle,
        "from_cache",
        classmethod(lambda _cls, **kw: cache_calls.append(kw) or "single"),
    )

    compare_calls: list[tuple] = []

    def fake_compare(_self, output_dir, *, subset=None, tabarena_context_kwargs=None, **_kw):
        compare_calls.append((Path(output_dir), subset, tabarena_context_kwargs))
        return pd.DataFrame({"method": ["m"], "metric": [1.0]})

    monkeypatch.setattr(ee.EndToEndResults, "compare_on_tabarena", fake_compare)

    class _FakeLB:
        def to_markdown(self, **_kwargs):
            return ""

    monkeypatch.setattr(wf, "format_leaderboard", lambda _df, **_kw: _FakeLB())

    cfg = _config(
        tmp_path,
        methods=[
            EvalMethod("A", ag_name_override="AG_A", result_suffix=" [Rerun]"),
            EvalMethod("B", ag_name_override="AG_B", only_load_cache=True),
        ],
        subsets=[[], ["regression"]],
    )
    out = run_eval(cfg)

    # Only the non-cache-only method is post-processed, with the suffix baked in.
    assert len(post_calls) == 1
    assert post_calls[0]["name_prefix_raw"] == "AG_A"
    assert post_calls[0]["method"] == "AG_A"
    assert post_calls[0]["artifact_name"] == "bench"
    assert post_calls[0]["name_suffix"] == " [Rerun]"
    assert Path(post_calls[0]["path_raw"]) == cfg.path_raw

    # The cache-only method is loaded from cache (keyed by benchmark_name).
    assert cache_calls == [{"method": "AG_B", "artifact_name": "bench"}]

    # One comparison per subset, with the expected output dir + subset + context.
    figs = Path(cfg.figure_output_dir)
    assert [c[0] for c in compare_calls] == [figs / "subsets" / "full", figs / "subsets" / "regression"]
    assert [c[1] for c in compare_calls] == [None, ["regression"]]
    assert compare_calls[0][2] == {"include_unverified": True}

    # Leaderboards returned + saved as CSV.
    assert set(out) == {"full", "regression"}
    assert (figs / "leaderboards" / "full.csv").exists()
    assert (figs / "leaderboards" / "regression.csv").exists()
