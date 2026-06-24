from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.models._in_memory_method_metadata import InMemoryMethodMetadata
from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._method_metadata_collection import MethodMetadataCollection
from tabarena.nips2025_utils.abstract_arena_context import AbstractArenaContext
from tabarena.nips2025_utils.end_to_end_single import EndToEndResultsSingle


def _results_frame(method: str, *, datasets=("d1",), folds=(0,), config_type="MM", ta_name="M", ta_suite="M"):
    rows = [
        {
            "method": method,
            "dataset": dataset,
            "fold": fold,
            "metric_error": 0.1,
            "metric": "roc_auc",
            "problem_type": "binary",
            "method_type": "config",
            "method_subtype": "default",
            "config_type": config_type,
            "ta_name": ta_name,
            "ta_suite": ta_suite,
        }
        for dataset in datasets
        for fold in folds
    ]
    return pd.DataFrame(rows)


def _task_metadata() -> TaskMetadataCollection:
    legacy = pd.DataFrame(
        {
            "tid": [0, 1],
            "dataset": ["d1", "d2"],
            "name": ["d1", "d2"],
            "n_folds": [1, 1],
            "n_repeats": [1, 1],
            "n_samples_train_per_fold": [100, 100],
            "n_samples_test_per_fold": [50, 50],
            "NumberOfInstances": [150, 150],
            "problem_type": ["binary", "binary"],
            "n_features": [10, 10],
            "n_classes": [2, 2],
            "target_feature": ["t", "t"],
        },
    )
    return TaskMetadataCollection.from_legacy_df(legacy)


class _DiskBackedMethod(MethodMetadata):
    """A non-in-memory method whose results come from an in-test frame (stands in for disk).

    ``is_in_memory`` stays False (inherited from ``MethodMetadata``), so this proves that
    ``only_valid_tasks`` scopes by registration via ``extra_methods=`` — not by in-memory status.
    """

    def __init__(self, results: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self._results = results

    def load_results(self) -> pd.DataFrame:
        return self._results


class TestInMemoryArtifacts:
    def test_is_in_memory_marker(self):
        assert MethodMetadata.is_in_memory is False
        assert InMemoryMethodMetadata.is_in_memory is True

    def test_load_results_returns_a_copy(self):
        df = _results_frame("MM (default)")
        im = InMemoryMethodMetadata(results=df, method="M", method_type="config", model_key="MM")
        out = im.load_results()
        assert out.equals(df)
        out.loc[0, "metric_error"] = 999.0
        assert im.load_results().loc[0, "metric_error"] == 0.1  # original untouched

    def test_to_info_dict_excludes_in_memory_slots(self):
        im = InMemoryMethodMetadata(results=_results_frame("MM (default)"), method="M", method_type="baseline")
        info = im.to_info_dict()
        assert "_results" not in info and "_repo" not in info
        assert info["method"] == "M"

    def test_disabled_disk_ops_raise(self):
        im = InMemoryMethodMetadata(results=_results_frame("MM (default)"), method="M", method_type="baseline")
        for op in ("to_yaml", "to_yaml_fileobj", "load_raw", "generate_repo", "method_downloader", "method_uploader"):
            with pytest.raises(NotImplementedError):
                getattr(im, op)()

    def test_load_processed_requires_repo(self):
        im = InMemoryMethodMetadata(results=_results_frame("MM (default)"), method="M", method_type="config")
        with pytest.raises(NotImplementedError, match="no in-memory repo"):
            im.load_processed()
        sentinel = object()
        im_with_repo = InMemoryMethodMetadata(
            results=_results_frame("MM (default)"),
            repo=sentinel,
            method="M",
            method_type="config",
        )
        assert im_with_repo.load_processed() is sentinel


class TestCollectionInfo:
    def test_info_does_not_embed_dataframe(self):
        im = InMemoryMethodMetadata(results=_results_frame("MM (default)"), method="M", method_type="baseline")
        info = MethodMetadataCollection([im]).info()
        assert "_results" not in info.columns
        assert "_repo" not in info.columns
        assert info.loc[0, "method"] == "M"


class TestFromResultsSinglePrefixIdentity:
    """new_result_prefix must be baked into identity so the website merge key still matches."""

    def _results_single(self) -> EndToEndResultsSingle:
        base = MethodMetadata(
            method="MyModel",
            suite="MyModel",
            method_type="config",
            ag_key="MM",
            model_key="MM",
            display_name="MyModel",
        )
        hpo = _results_frame("MM (default)", config_type="MM", ta_name="MyModel", ta_suite="MyModel")
        return EndToEndResultsSingle(method_metadata=base, model_results=hpo, hpo_results=hpo)

    def test_prefix_applied_to_identity_fields(self):
        im = self._results_single().to_method_metadata(new_result_prefix="[New] ")
        assert im.method == "[New] MyModel"
        assert im.suite == "[New] MyModel"
        assert im.display_name == "[New] MyModel"
        assert im.config_type == "[New] MM"  # model_key was prefixed too

    def test_frame_and_identity_share_the_website_merge_key(self):
        # leaderboard_to_website_format merges leaderboard <-> info on (ta_name, ta_suite),
        # where info's ta_name/ta_suite == method/suite. They must equal the frame's.
        im = self._results_single().to_method_metadata(new_result_prefix="[New] ")
        frame = im.load_results()
        assert list(frame["ta_name"].unique()) == [im.method]
        assert list(frame["ta_suite"].unique()) == [im.suite]
        assert list(frame["method"].unique()) == ["[New] MM (default)"]

    def test_no_prefix_is_identity(self):
        im = self._results_single().to_method_metadata()
        assert im.method == "MyModel"
        assert im.config_type == "MM"


class TestContextRegistration:
    def _ctx(self, *extra) -> AbstractArenaContext:
        return AbstractArenaContext(methods=[], task_metadata=_task_metadata(), extra_methods=list(extra))

    def _in_memory(self, method: str, datasets) -> InMemoryMethodMetadata:
        return InMemoryMethodMetadata(
            results=_results_frame(f"{method} (default)", datasets=datasets, ta_name=method, ta_suite=method),
            method=method,
            suite=method,
            method_type="config",
            model_key=method,
        )

    def _disk_backed(self, method: str, datasets) -> _DiskBackedMethod:
        return _DiskBackedMethod(
            results=_results_frame(f"{method} (default)", datasets=datasets, ta_name=method, ta_suite=method),
            method=method,
            suite=method,
            method_type="config",
            model_key=method,
        )

    def test_registered_method_is_listed_and_loadable(self):
        im = self._in_memory("NewA", datasets=("d1",))
        ctx = self._ctx(im)
        assert "NewA" in ctx.methods
        loaded = ctx.load_results()
        assert set(loaded["method"]) == {"NewA (default)"}

    def test_registered_new_results_concats_registered_new_methods(self):
        # Both an in-memory and a disk-backed method registered via extra_methods= contribute;
        # only_valid_tasks scopes by registration, not by in-memory status.
        im = self._in_memory("NewA", datasets=("d1",))
        disk = self._disk_backed("NewB", datasets=("d1", "d2"))
        assert disk.is_in_memory is False
        ctx = self._ctx(im, disk)
        new = ctx._registered_new_results()
        assert set(new["method"]) == {"NewA (default)", "NewB (default)"}

    def test_registered_new_results_none_without_extra_methods(self):
        assert self._ctx()._registered_new_results() is None


class TestResolveOnlyValidTasks:
    def _ctx_with(self, im) -> AbstractArenaContext:
        return AbstractArenaContext(methods=[], task_metadata=_task_metadata(), extra_methods=[im])

    def _im(self) -> InMemoryMethodMetadata:
        return InMemoryMethodMetadata(
            results=_results_frame("NewA (default)", datasets=("d1",), ta_name="NewA", ta_suite="NewA"),
            method="NewA",
            suite="NewA",
            method_type="config",
            model_key="NewA",
        )

    def test_false_is_no_restriction(self):
        ctx = self._ctx_with(self._im())
        assert ctx._resolve_only_valid_tasks(False, None) == (None, None)

    def test_true_uses_registered_in_memory_methods(self):
        ctx = self._ctx_with(self._im())
        df_filter, names = ctx._resolve_only_valid_tasks(True, None)
        assert names is None
        assert set(df_filter["dataset"]) == {"d1"}  # only the task NewA ran

    def test_true_prefers_explicit_new_results(self):
        ctx = self._ctx_with(self._im())
        new_results = _results_frame("X (default)", datasets=("d2",))
        df_filter, _ = ctx._resolve_only_valid_tasks(True, new_results)
        assert set(df_filter["dataset"]) == {"d2"}

    def test_true_without_any_source_raises(self):
        ctx = AbstractArenaContext(methods=[], task_metadata=_task_metadata())
        with pytest.raises(ValueError, match="only_valid_tasks=True needs"):
            ctx._resolve_only_valid_tasks(True, None)

    def test_method_metadata_object_resolves_to_its_tasks(self):
        im = self._im()
        ctx = self._ctx_with(im)
        df_filter, names = ctx._resolve_only_valid_tasks(im, None)
        assert names is None
        assert set(df_filter["dataset"]) == {"d1"}

    def test_string_names_pass_through(self):
        ctx = self._ctx_with(self._im())
        df_filter, names = ctx._resolve_only_valid_tasks(["NewA (default)"], None)
        assert df_filter is None
        assert names == ["NewA (default)"]

    def test_numpy_array_of_names_passes_through_as_list(self):
        ctx = self._ctx_with(self._im())
        df_filter, names = ctx._resolve_only_valid_tasks(np.array(["NewA (default)"]), None)
        assert df_filter is None
        assert names == ["NewA (default)"]

    def test_mixed_list_raises(self):
        im = self._im()
        ctx = self._ctx_with(im)
        with pytest.raises(TypeError, match="not mixed"):
            ctx._resolve_only_valid_tasks([im, "NewA (default)"], None)


class TestInitOnlyValidTasks:
    """`only_valid_tasks=True` at init pre-filters task_metadata to the new methods' tasks."""

    def _im(self, method: str, datasets) -> InMemoryMethodMetadata:
        return InMemoryMethodMetadata(
            results=_results_frame(f"{method} (default)", datasets=datasets, ta_name=method, ta_suite=method),
            method=method,
            suite=method,
            method_type="config",
            model_key=method,
        )

    def test_default_keeps_full_task_metadata(self):
        ctx = AbstractArenaContext(
            methods=[],
            task_metadata=_task_metadata(),
            extra_methods=[self._im("NewA", datasets=("d1",))],
        )
        assert ctx.only_valid_tasks is False
        assert set(ctx.task_metadata_collection.dataset_names()) == {"d1", "d2"}

    def test_prefilters_to_registered_method_tasks(self):
        ctx = AbstractArenaContext(
            methods=[],
            task_metadata=_task_metadata(),
            extra_methods=[self._im("NewA", datasets=("d1",))],
            only_valid_tasks=True,
        )
        assert ctx.only_valid_tasks is True
        # d2 had no new-method results, so it is pruned from the context's task_metadata.
        assert ctx.task_metadata_collection.dataset_names() == ["d1"]
        # The legacy DataFrame view derives from the (now filtered) collection.
        assert set(ctx.task_metadata["dataset"]) == {"d1"}

    def test_prefilters_with_disk_backed_method(self):
        # only_valid_tasks does not require in-memory methods: a disk-backed method registered
        # via extra_methods= defines the valid tasks just the same.
        disk = _DiskBackedMethod(
            results=_results_frame("Disk (default)", datasets=("d1",), ta_name="Disk", ta_suite="Disk"),
            method="Disk",
            suite="Disk",
            method_type="config",
            model_key="Disk",
        )
        assert disk.is_in_memory is False
        ctx = AbstractArenaContext(
            methods=[],
            task_metadata=_task_metadata(),
            extra_methods=[disk],
            only_valid_tasks=True,
        )
        assert ctx.task_metadata_collection.dataset_names() == ["d1"]

    def test_results_filter_frame_tracks_prefilter(self):
        ctx = AbstractArenaContext(
            methods=[],
            task_metadata=_task_metadata(),
            extra_methods=[self._im("NewA", datasets=("d1",))],
            only_valid_tasks=True,
        )
        # The (dataset, fold) frame compare uses to scope results matches the kept tasks.
        df_filter = ctx._task_metadata_results_filter()
        assert set(zip(df_filter["dataset"], df_filter["fold"], strict=False)) == {("d1", 0)}

    def test_without_new_methods_raises(self):
        with pytest.raises(ValueError, match="only_valid_tasks=True needs"):
            AbstractArenaContext(methods=[], task_metadata=_task_metadata(), only_valid_tasks=True)

    def test_no_overlap_with_task_metadata_raises(self):
        # A method that ran a task outside the context's universe must not widen it.
        with pytest.raises(ValueError, match="share no"):
            AbstractArenaContext(
                methods=[],
                task_metadata=_task_metadata(),
                extra_methods=[self._im("Bad", datasets=("d3",))],
                only_valid_tasks=True,
            )


class TestFromNewMethodsFactory:
    """`from_new_methods` pairs extra_methods + only_valid_tasks=True in one intent-revealing call."""

    def _im(self, method: str, datasets) -> InMemoryMethodMetadata:
        return InMemoryMethodMetadata(
            results=_results_frame(f"{method} (default)", datasets=datasets, ta_name=method, ta_suite=method),
            method=method,
            suite=method,
            method_type="config",
            model_key=method,
        )

    def test_registers_methods_and_prefilters_tasks(self):
        ctx = AbstractArenaContext.from_new_methods(
            [self._im("NewA", datasets=("d1",))],
            methods=[],
            task_metadata=_task_metadata(),
        )
        assert "NewA" in ctx.methods
        assert ctx.only_valid_tasks is True
        # d2 had no new-method results -> pruned (same as the explicit two-kwarg form).
        assert ctx.task_metadata_collection.dataset_names() == ["d1"]

    def test_returns_the_concrete_subclass(self):
        ctx = AbstractArenaContext.from_new_methods(
            [self._im("NewA", datasets=("d1",))],
            methods=[],
            task_metadata=_task_metadata(),
        )
        # Self return type: a factory on the base yields the base; subclasses yield themselves.
        assert type(ctx) is AbstractArenaContext

    def test_forwards_extra_kwargs(self):
        ctx = AbstractArenaContext.from_new_methods(
            [self._im("NewA", datasets=("d1",))],
            methods=[],
            task_metadata=_task_metadata(),
            backend="native",
        )
        assert ctx.backend == "native"

    def test_without_new_methods_raises(self):
        with pytest.raises(ValueError, match="only_valid_tasks=True needs"):
            AbstractArenaContext.from_new_methods([], methods=[], task_metadata=_task_metadata())


class TestRegister:
    """`register` adds run results post-construction and (by default) scopes task_metadata."""

    def _im(self, method: str, datasets) -> InMemoryMethodMetadata:
        return InMemoryMethodMetadata(
            results=_results_frame(f"{method} (default)", datasets=datasets, ta_name=method, ta_suite=method),
            method=method,
            suite=method,
            method_type="config",
            model_key=method,
        )

    def test_register_converts_and_scopes(self, monkeypatch):
        # Stub the heavy raw->methods conversion; register's job is the wiring + scoping.
        im = self._im("NewA", datasets=("d1",))
        monkeypatch.setattr(
            "tabarena.nips2025_utils.end_to_end.EndToEnd.from_raw_to_methods",
            lambda **kwargs: [im],
        )
        ctx = AbstractArenaContext(methods=[], task_metadata=_task_metadata())
        out = ctx.register(["raw"], new_result_prefix="[New] ")
        assert out == [im]
        assert "NewA" in ctx.methods
        assert ctx.only_valid_tasks is True
        assert ctx.task_metadata_collection.dataset_names() == ["d1"]  # d2 pruned

    def test_register_without_scoping_keeps_full_task_metadata(self, monkeypatch):
        im = self._im("NewA", datasets=("d1",))
        monkeypatch.setattr(
            "tabarena.nips2025_utils.end_to_end.EndToEnd.from_raw_to_methods",
            lambda **kwargs: [im],
        )
        ctx = AbstractArenaContext(methods=[], task_metadata=_task_metadata())
        ctx.register(["raw"], scope_to_valid_tasks=False)
        assert "NewA" in ctx.methods
        assert ctx.only_valid_tasks is False
        assert set(ctx.task_metadata_collection.dataset_names()) == {"d1", "d2"}


class _StubExperiment:
    """Minimal Experiment stand-in: a Job only needs a `.name`-bearing experiment to carry."""

    model_constraints = None

    def __init__(self, name: str = "exp1"):
        self.name = name


def _jobs(dataset: str = "d1"):
    from tabarena.benchmark.experiment import Job

    return [Job.create(_StubExperiment(), dataset, fold=0, repeat=0)]


class TestRunJobs:
    """`run_jobs` scopes+materializes the collection, runs the jobs, then registers."""

    @staticmethod
    def _patch_runner(monkeypatch, captured: dict, *, results):
        import tabarena.benchmark.experiment as exp_mod

        class FakeRunner:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_jobs(self, jobs):
                captured["ran"] = jobs
                return results

        monkeypatch.setattr(exp_mod, "ExperimentBatchRunner", FakeRunner)

    def test_scopes_runner_and_registers(self, monkeypatch, tmp_path):
        ctx = AbstractArenaContext(methods=[], task_metadata=_task_metadata())
        captured: dict = {}
        self._patch_runner(monkeypatch, captured, results=["raw-result"])
        monkeypatch.setattr(ctx, "register", lambda results, **kw: captured.update(registered=results, register_kw=kw))

        jobs = _jobs("d1")
        out = ctx.run_jobs(
            jobs,
            expname=str(tmp_path),
            new_result_prefix="[New] ",
            debug_mode=True,  # an explicit runner_kwarg (overriding ExperimentBatchRunner's default)
        )
        assert out == ["raw-result"]
        assert captured["ran"] == jobs  # jobs forwarded to runner.run_jobs
        assert captured["expname"] == str(tmp_path)
        assert captured["debug_mode"] is True  # runner_kwargs forwarded
        # The runner is built over the collection scoped to (and materialized for) the jobs' splits.
        assert captured["task_metadata"].dataset_fold_repeats() == [("d1", 0, 0)]
        assert captured["registered"] == ["raw-result"]
        assert captured["register_kw"]["new_result_prefix"] == "[New] "

    def test_omitting_runner_kwargs_leaves_runner_defaults(self, monkeypatch, tmp_path):
        # run_jobs forwards only what it is given; an unspecified debug_mode is not injected,
        # so ExperimentBatchRunner keeps its own default (False).
        ctx = AbstractArenaContext(methods=[], task_metadata=_task_metadata())
        captured: dict = {}
        self._patch_runner(monkeypatch, captured, results=[])
        monkeypatch.setattr(ctx, "register", lambda *a, **k: None)

        ctx.run_jobs(_jobs(), expname=str(tmp_path))
        assert "debug_mode" not in captured  # not forwarded -> runner uses its own default

    def test_register_false_skips_registration(self, monkeypatch, tmp_path):
        ctx = AbstractArenaContext(methods=[], task_metadata=_task_metadata())
        self._patch_runner(monkeypatch, {}, results=["raw-result"])
        called = {"register": False}
        monkeypatch.setattr(ctx, "register", lambda *a, **k: called.__setitem__("register", True))

        out = ctx.run_jobs(_jobs(), expname=str(tmp_path), register=False)
        assert out == ["raw-result"]
        assert called["register"] is False

    def test_empty_jobs_short_circuits(self, monkeypatch, tmp_path):
        # No runner construction and no registration for an empty job list.
        ctx = AbstractArenaContext(methods=[], task_metadata=_task_metadata())
        monkeypatch.setattr(ctx, "register", lambda *a, **k: pytest.fail("should not register"))
        assert ctx.run_jobs([], expname=str(tmp_path)) == []

    def test_run_job_delegates_to_run_jobs(self, monkeypatch):
        ctx = AbstractArenaContext(methods=[], task_metadata=_task_metadata())
        seen: dict = {}
        monkeypatch.setattr(ctx, "run_jobs", lambda jobs, **kw: seen.update(jobs=jobs, kw=kw) or ["r"])
        job = _jobs()[0]
        assert ctx.run_job(job, expname="x") == ["r"]
        assert seen["jobs"] == [job]
        assert seen["kw"] == {"expname": "x"}


class TestBuildAndRunJobs:
    """`build_and_run_jobs` = build_jobs(subset=..., **build_kwargs) then run_jobs(...) in one call.

    Scoping (`subset` + `build_kwargs`) reaches build_jobs; `**runner_kwargs` reach run_jobs.
    """

    def test_routes_scoping_to_build_and_runner_kwargs_to_run(self, monkeypatch, tmp_path):
        ctx = AbstractArenaContext(methods=[], task_metadata=_task_metadata())
        seen: dict = {}
        sentinel = [_jobs()[0]]
        monkeypatch.setattr(
            ctx,
            "build_jobs",
            lambda experiments, **kw: seen.update(build_experiments=experiments, build_kw=kw) or sentinel,
        )
        monkeypatch.setattr(ctx, "run_jobs", lambda jobs, **kw: seen.update(run_jobs=jobs, run_kw=kw) or ["raw"])

        out = ctx.build_and_run_jobs(
            ["exp"],
            expname=str(tmp_path),
            subset="lite",
            build_kwargs={"dataset_names": ["d1"]},  # extra build-time filter -> build_jobs
            new_result_prefix="[New] ",
            debug_mode=True,  # a runner kwarg -> run_jobs
            user_tasks=["t"],  # a runner kwarg -> run_jobs
        )
        assert out == ["raw"]
        assert seen["build_experiments"] == ["exp"]
        assert seen["build_kw"] == {
            "task_subset": None,
            "subset": "lite",
            "dataset_names": ["d1"],
            "pre_materialize": False,
        }
        assert seen["run_jobs"] is sentinel
        assert seen["run_kw"] == {
            "expname": str(tmp_path),
            "register": True,
            "new_result_prefix": "[New] ",
            "debug_mode": True,
            "user_tasks": ["t"],
        }


class TestCompareReturnFlags:
    """`compare`'s `return_results` (also return df) and `new_methods_only` (scope lb + df) flags."""

    @staticmethod
    def _ctx_and_results(monkeypatch):
        import tabarena.nips2025_utils.compare as compare_mod

        # Stub the lower-level scoring with a leaderboard over baseline + new method.
        monkeypatch.setattr(
            compare_mod,
            "compare",
            lambda **kw: pd.DataFrame({"method": ["CatBoost", "[New] M"], "elo": [1000.0, 1100.0]}),
        )
        ctx = AbstractArenaContext(methods=[], task_metadata=_task_metadata())
        ctx._new_method_names = {"[New] M"}
        new_results = pd.DataFrame(
            {
                "method": ["[New] M", "[New] M", "CatBoost"],
                "dataset": ["d1", "d2", "d1"],
                "fold": [0, 0, 0],
                "metric_error": [0.1, 0.2, 0.3],
            },
        )
        return ctx, new_results

    def test_no_flags_returns_only_full_leaderboard(self, monkeypatch, tmp_path):
        ctx, new_results = self._ctx_and_results(monkeypatch)
        out = ctx.compare(output_dir=tmp_path, new_results=new_results, filter_to_task_metadata=False)
        assert isinstance(out, pd.DataFrame)  # not a tuple
        assert list(out["method"]) == ["CatBoost", "[New] M"]

    def test_return_results_gives_full_lb_and_df(self, monkeypatch, tmp_path):
        ctx, new_results = self._ctx_and_results(monkeypatch)
        lb, results = ctx.compare(
            output_dir=tmp_path, new_results=new_results, filter_to_task_metadata=False, return_results=True
        )
        assert list(lb["method"]) == ["CatBoost", "[New] M"]  # full leaderboard
        assert list(results["method"]) == ["[New] M", "[New] M", "CatBoost"]  # full per-split frame

    def test_new_methods_only_scopes_both_lb_and_df(self, monkeypatch, tmp_path):
        ctx, new_results = self._ctx_and_results(monkeypatch)
        lb, results = ctx.compare(
            output_dir=tmp_path,
            new_results=new_results,
            filter_to_task_metadata=False,
            return_results=True,
            new_methods_only=True,
        )
        assert list(lb["method"]) == ["[New] M"]  # leaderboard scoped to the new method
        assert list(results["method"]) == ["[New] M", "[New] M"]  # baseline rows dropped
        assert list(results["metric_error"]) == [0.1, 0.2]

    def test_new_methods_only_without_return_results_scopes_just_the_leaderboard(self, monkeypatch, tmp_path):
        ctx, new_results = self._ctx_and_results(monkeypatch)
        out = ctx.compare(
            output_dir=tmp_path, new_results=new_results, filter_to_task_metadata=False, new_methods_only=True
        )
        assert isinstance(out, pd.DataFrame)  # still just the leaderboard
        assert list(out["method"]) == ["[New] M"]

    def test_new_methods_only_raises_when_no_new_methods_registered(self, monkeypatch, tmp_path):
        ctx, new_results = self._ctx_and_results(monkeypatch)
        ctx._new_method_names = set()  # nothing registered
        with pytest.raises(ValueError, match="no new methods are registered"):
            ctx.compare(
                output_dir=tmp_path, new_results=new_results, filter_to_task_metadata=False, new_methods_only=True
            )

    def test_return_single_returns_the_one_row_and_results(self, monkeypatch, tmp_path):
        ctx, new_results = self._ctx_and_results(monkeypatch)
        row, results = ctx.compare(
            output_dir=tmp_path,
            new_results=new_results,
            filter_to_task_metadata=False,
            return_results=True,
            return_single=True,
        )
        assert isinstance(row, pd.Series)  # a single leaderboard row, not a frame
        assert row["method"] == "[New] M"
        assert row["elo"] == 1100.0
        assert list(results["method"]) == ["[New] M", "[New] M"]

    def test_return_single_without_results_returns_just_the_row(self, monkeypatch, tmp_path):
        ctx, new_results = self._ctx_and_results(monkeypatch)
        row = ctx.compare(
            output_dir=tmp_path, new_results=new_results, filter_to_task_metadata=False, return_single=True
        )
        assert isinstance(row, pd.Series)
        assert row["method"] == "[New] M"

    def test_return_single_raises_when_multiple_new_methods(self, monkeypatch, tmp_path):
        import tabarena.nips2025_utils.compare as compare_mod

        monkeypatch.setattr(
            compare_mod,
            "compare",
            lambda **kw: pd.DataFrame({"method": ["[New] A", "[New] B"], "elo": [1.0, 2.0]}),
        )
        ctx = AbstractArenaContext(methods=[], task_metadata=_task_metadata())
        ctx._new_method_names = {"[New] A", "[New] B"}
        new_results = pd.DataFrame(
            {"method": ["[New] A", "[New] B"], "dataset": ["d1", "d1"], "fold": [0, 0], "metric_error": [0.1, 0.2]},
        )
        with pytest.raises(ValueError, match="return_single=True but 2 new-method row"):
            ctx.compare(output_dir=tmp_path, new_results=new_results, filter_to_task_metadata=False, return_single=True)


class TestTempDir:
    """`output_dir=None` / `expname=None` use a throwaway temp dir, created for the call and removed after."""

    def test_compare_temp_dir_is_used_and_cleaned(self, monkeypatch):
        import os

        import tabarena.nips2025_utils.compare as compare_mod

        seen: dict = {}

        def fake_compare(**kw):
            seen["path"] = str(kw["output_dir"])
            seen["existed_during"] = os.path.isdir(seen["path"])
            return pd.DataFrame({"method": ["X"]})

        monkeypatch.setattr(compare_mod, "compare", fake_compare)
        ctx = AbstractArenaContext(methods=[], task_metadata=_task_metadata())
        out = ctx.compare(output_dir=None, filter_to_task_metadata=False)
        assert isinstance(out, pd.DataFrame)
        assert seen["existed_during"] is True  # a real temp dir existed during the call
        assert not os.path.isdir(seen["path"])  # and was cleaned up afterwards

    def test_run_jobs_temp_dir_is_used_and_cleaned(self, monkeypatch):
        import os

        import tabarena.benchmark.experiment as exp_mod

        seen: dict = {}

        class FakeRunner:
            def __init__(self, **kwargs):
                seen["expname"] = kwargs["expname"]

            def run_jobs(self, jobs):
                seen["existed_during"] = os.path.isdir(seen["expname"])
                return []

        monkeypatch.setattr(exp_mod, "ExperimentBatchRunner", FakeRunner)
        ctx = AbstractArenaContext(methods=[], task_metadata=_task_metadata())
        monkeypatch.setattr(ctx, "register", lambda *a, **k: None)
        ctx.run_jobs(_jobs(), expname=None)
        assert seen["existed_during"] is True
        assert not os.path.isdir(seen["expname"])  # cleaned up after the run
