"""Tests for `tabarena.evaluation.benchmark_eval`.

The heavy engines (Ray post-processing, OpenML fetch, TabArena context) are
monkeypatched, so these only exercise the orchestration + the pure helpers.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

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

    def test_display_name_from_registry(self):
        """The leaderboard / figure label is the registry's display name, not the raw config type."""
        assert EvalMethod("RandomForest").display_name == "RandomForest"

    def test_display_name_appends_result_suffix(self):
        """A re-run keeps its suffix in the label, so it stays distinguishable from the hosted method."""
        assert EvalMethod("RandomForest", result_suffix=" [Rerun]").display_name == "RandomForest [Rerun]"

    def test_display_name_override_wins(self):
        method = EvalMethod("RandomForest", result_suffix=" [Rerun]", display_name_override="RF (ours)")
        assert method.display_name == "RF (ours)"

    def test_custom_method_without_override_has_no_display_name(self):
        """A method the registry does not know keeps the default label (its config type)."""
        assert EvalMethod("NotARegisteredModel", ag_name_override="AG_X").display_name is None

    def test_unknown_registry_name_raises(self):
        with pytest.raises(ValueError, match="not recognized"):
            _ = EvalMethod("NotARegisteredModel").display_name


class TestMethodArtifact:
    def test_method_name_bakes_in_result_suffix(self):
        """The suffix is part of the cache identity, so a re-run of a registered baseline
        registers under a distinct name instead of colliding on the bare method name.
        """
        from tabarena.evaluation._eval_common import MethodArtifact

        kwargs = {"ag_name": "RF", "path_raw": Path("/raw"), "suite": "bench"}
        assert MethodArtifact(**kwargs).method_name == "RF"
        assert MethodArtifact(**kwargs, result_suffix=" [Rerun]").method_name == "RF [Rerun]"

    def test_display_name_defaults_to_none(self):
        from tabarena.evaluation._eval_common import MethodArtifact

        assert MethodArtifact(ag_name="RF", path_raw=Path("/raw"), suite="bench").display_name is None


def test_post_process_applies_display_name_to_cached_methods(monkeypatch):
    """The artifact's display name reaches methods loaded from the cache too (`only_load_cache`, or a
    cache built before the name was recorded), while methods without one keep their label.
    """
    import tabarena.end_to_end.end_to_end as ee
    from tabarena.evaluation._eval_common import MethodArtifact, post_process_to_results

    monkeypatch.setattr(ee.EndToEnd, "from_path_raw", staticmethod(lambda **_kw: None))
    loaded = SimpleNamespace(
        method_results_lst=[
            SimpleNamespace(method_metadata=SimpleNamespace(method="AG_A", suite="bench", display_name="AG_A")),
            SimpleNamespace(method_metadata=SimpleNamespace(method="AG_B", suite="bench", display_name="AG_B")),
        ]
    )
    monkeypatch.setattr(ee.EndToEndResults, "from_cache", classmethod(lambda _cls, methods, **_kw: loaded))

    artifacts = [
        MethodArtifact(
            ag_name="AG_A", path_raw=Path("/raw"), suite="bench", display_name="Model A", only_load_cache=True
        ),
        MethodArtifact(ag_name="AG_B", path_raw=Path("/raw"), suite="bench"),
    ]
    out = post_process_to_results(artifacts)
    assert [m.method_metadata.display_name for m in out.method_results_lst] == ["Model A", "AG_B"]


def test_warn_on_duplicate_display_names(capsys):
    from tabarena.evaluation.benchmark_eval import _warn_on_duplicate_display_names

    hosted = SimpleNamespace(method="TabPFN-3", suite="tabarena-2026-07-13", display_name="TabPFN-3")
    rerun = SimpleNamespace(method="TA-TabPFN-3", suite="bench", display_name="TabPFN-3")
    suffixed = SimpleNamespace(method="TA-TabPFN-3 [Rerun]", suite="bench", display_name="TabPFN-3 [Rerun]")
    context = SimpleNamespace(method_metadata_collection=SimpleNamespace(method_metadata_lst=[hosted, rerun, suffixed]))

    _warn_on_duplicate_display_names(context, [rerun, suffixed])
    out = capsys.readouterr().out
    assert "WARNING: display name 'TabPFN-3' of 'TA-TabPFN-3'" in out
    assert "'TabPFN-3 [Rerun]'" not in out


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

    def test_pareto_focus_new_methods_defaults_true(self, tmp_path):
        # The eval exists to place the run's methods, so they are emphasized in the Pareto figures.
        assert _config(tmp_path).pareto_focus_new_methods is True
        assert _config(tmp_path, pareto_focus_new_methods=False).pareto_focus_new_methods is False

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
    compare_kwargs: list[dict] = []
    context_init_calls: list = []
    context_only_valid_tasks: list = []
    methods_sentinel = [object()]

    class _FakeResults:
        """Stands in for the EndToEndResults reloaded from cache (phase 2)."""

        method_results_lst: list = []

        def to_method_metadata_lst(self, **_kw):
            return methods_sentinel

    class _FakeContext:
        """Stands in for the TabArenaContext the run's methods are registered on."""

        def __init__(self, *, extra_methods=None, only_valid_tasks=False, **_kw):
            context_init_calls.append(extra_methods)
            context_only_valid_tasks.append(only_valid_tasks)

        def compare(self, output_dir, *, subset=None, **kw):
            compare_calls.append((Path(output_dir), subset))
            compare_kwargs.append(kw)
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
    # A custom method (ag_name_override, unknown to the registry) has no display name to record.
    assert post_calls[0]["display_name"] is None

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
    # The run's methods are box-labeled in the Pareto figures by default (the config's flag).
    assert all(kw["pareto_focus_new_methods"] is True for kw in compare_kwargs)

    # Leaderboards returned + saved as CSV.
    assert set(out) == {"full", "regression"}
    assert (figs / "leaderboards" / "full.csv").exists()
    assert (figs / "leaderboards" / "regression.csv").exists()
