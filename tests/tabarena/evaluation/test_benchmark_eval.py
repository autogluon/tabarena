"""Tests for `tabarena.evaluation.benchmark_eval`.

The heavy engines (Ray post-processing, OpenML fetch, TabArena context) are
monkeypatched, so these only exercise the orchestration + the pure helpers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabarena.evaluation import EvalMethod, TabArenaEvalConfig, run_eval
from tabarena.loaders import get_tabarena_cache_root, set_tabarena_cache_root


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


class TestMethodArtifact:
    def test_method_name_bakes_in_result_suffix(self):
        """The suffix is part of the cache identity, so a re-run of a registered baseline
        registers under a distinct name instead of colliding on the bare method name.
        """
        from tabarena.evaluation._eval_common import MethodArtifact

        kwargs = {"ag_name": "RF", "path_raw": Path("/raw"), "suite": "bench"}
        assert MethodArtifact(**kwargs).method_name == "RF"
        assert MethodArtifact(**kwargs, result_suffix=" [Rerun]").method_name == "RF [Rerun]"


class TestConfig:
    def test_path_raw_is_output_dir_data(self):
        cfg = _config(Path("/base"), output_dir="/x/out/bench")
        assert cfg.path_raw == Path("/x/out/bench/data")

    def test_subsets_default_is_full(self, tmp_path):
        assert _config(tmp_path).subsets_to_run() == [[]]

    def test_subsets_passthrough(self, tmp_path):
        assert _config(tmp_path, subsets=[[], ["regression"]]).subsets_to_run() == [[], ["regression"]]

    def test_only_valid_tasks_defaults_false(self, tmp_path):
        assert _config(tmp_path).only_valid_tasks is False

    def test_only_valid_tasks_passthrough(self, tmp_path):
        assert _config(tmp_path, only_valid_tasks=True).only_valid_tasks is True

    def test_init_caches_sets_tabarena_cache_root(self, tmp_path):
        try:
            _config(tmp_path, tabarena_cache_path="/c").init_caches()
            assert get_tabarena_cache_root() == Path("/c")
        finally:
            set_tabarena_cache_root(None)

    def test_init_caches_prefers_cache_config(self, tmp_path):
        from tabarena.caching import CacheConfig

        try:
            # cache_config wins over the legacy *_cache_path field.
            _config(
                tmp_path,
                cache_config=CacheConfig(tabarena="/from_config"),
                tabarena_cache_path="/legacy",
            ).init_caches()
            assert get_tabarena_cache_root() == Path("/from_config")
        finally:
            set_tabarena_cache_root(None)


def test_run_eval_orchestration(tmp_path, monkeypatch):
    import tabarena.contexts.tabarena.context as tc
    import tabarena.end_to_end.end_to_end as ee
    import tabarena.website.website_format as wf

    post_calls: list[dict] = []
    monkeypatch.setattr(
        ee.EndToEnd,
        "from_path_raw",
        staticmethod(lambda **kw: post_calls.append(kw)),
    )

    compare_calls: list[tuple] = []
    context_init_calls: list = []
    context_only_valid_tasks: list = []
    methods_sentinel = [object()]

    class _FakeResults:
        """Stands in for the EndToEndResults reloaded from cache (phase 2)."""

        def to_method_metadata_lst(self, **_kw):
            return methods_sentinel

    class _FakeContext:
        """Stands in for the TabArenaContext the run's methods are registered on."""

        def __init__(self, *, extra_methods=None, only_valid_tasks=False, **_kw):
            context_init_calls.append(extra_methods)
            context_only_valid_tasks.append(only_valid_tasks)

        def compare(self, output_dir, *, subset=None, **_kw):
            compare_calls.append((Path(output_dir), subset))
            return pd.DataFrame({"method": ["m"], "metric": [1.0]})

    # Phase 2 reloads every method from the cache via EndToEndResults.from_cache; capture the args.
    from_cache_calls: list = []
    monkeypatch.setattr(
        ee.EndToEndResults,
        "from_cache",
        classmethod(lambda _cls, methods, **kw: from_cache_calls.append(methods) or _FakeResults()),
    )
    # The run's vended methods are registered on a TabArenaContext (extra_methods=) and compared.
    monkeypatch.setattr(tc, "TabArenaContext", _FakeContext)

    class _FakeLB:
        def to_markdown(self, **_kwargs):
            return ""

    # run_eval imports format_leaderboard from its source at call time, so patch there.
    monkeypatch.setattr(wf, "format_leaderboard", lambda _df, **_kw: _FakeLB())

    cfg = _config(
        tmp_path,
        methods=[
            EvalMethod("A", ag_name_override="AG_A", result_suffix=" [Rerun]"),
            EvalMethod("B", ag_name_override="AG_B", only_load_cache=True),
        ],
        subsets=[[], ["regression"]],
        only_valid_tasks=True,
    )
    out = run_eval(cfg)

    # Phase 1: only the non-cache-only method is post-processed. The raw-folder match uses the
    # bare ag_name; the suffix is baked into both the result rows (name_suffix) and the cache
    # method identity (method), so a re-run registers under a distinct name from the original.
    assert len(post_calls) == 1
    assert post_calls[0]["name_prefix_raw"] == "AG_A"
    assert post_calls[0]["method"] == "AG_A [Rerun]"
    assert post_calls[0]["suite"] == "bench"
    assert post_calls[0]["name_suffix"] == " [Rerun]"
    assert Path(post_calls[0]["path_raw"]) == cfg.path_raw

    # Phase 2: every method is re-loaded from cache as (method_name, suite), exactly once.
    assert from_cache_calls == [[("AG_A [Rerun]", "bench"), ("AG_B", "bench")]]

    # The context is built once, with the run's vended methods registered via extra_methods=,
    # and the config's only_valid_tasks forwarded through.
    assert context_init_calls == [methods_sentinel]
    assert context_only_valid_tasks == [True]

    # One comparison per subset, with the expected output dir + subset.
    figs = Path(cfg.figure_output_dir)
    assert [c[0] for c in compare_calls] == [figs / "subsets" / "full", figs / "subsets" / "regression"]
    assert [c[1] for c in compare_calls] == [None, ["regression"]]

    # Leaderboards returned + saved as CSV.
    assert set(out) == {"full", "regression"}
    assert (figs / "leaderboards" / "full.csv").exists()
    assert (figs / "leaderboards" / "regression.csv").exists()
