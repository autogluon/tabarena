"""Unit tests for the per-model planning layer (`tabflow_slurm.setup.plan`).

Build-only: these exercise `TabArenaBenchmarkPlan.build_setups` and the
override/grouping logic. They never call `setup_jobs` (which needs Ray + real
task metadata).
"""

from __future__ import annotations

import pytest
from tabarena.benchmark.experiment import TabArenaExperimentBundle
from tabarena.benchmark.task.metadata import TabArenaMetadataBundle

# Import a real submodule (see test_setup.py for why a bare namespace won't skip).
pytest.importorskip("tabflow_slurm.setup", reason="tabflow_slurm is not installed")

from tabflow_slurm.setup.paths import PathSetup  # noqa: E402
from tabflow_slurm.setup.plan import (  # noqa: E402
    ModelJob,
    SingleModel,
    TabArenaBenchmarkPlan,
    _apply_overrides,
)
from tabflow_slurm.setup.resources import ResourcesSetup  # noqa: E402
from tabflow_slurm.setup.scheduler import SlurmSetup  # noqa: E402


def _slurm(**kw) -> SlurmSetup:
    kw.setdefault("gpu_partition", "gpu_part")
    kw.setdefault("cpu_partition", "cpu_part")
    kw.setdefault("extra_gres", None)
    return SlurmSetup(**kw)


def _resources(**kw) -> ResourcesSetup:
    kw.setdefault("time_limit", 3600)
    kw.setdefault("num_cpus", 8)
    kw.setdefault("num_gpus", 0)
    kw.setdefault("memory_limit", 32)
    return ResourcesSetup(**kw)


def _plan(model_jobs: list[ModelJob], **kwargs) -> TabArenaBenchmarkPlan:
    defaults = {
        "benchmark_name": "my_bench",
        "model_jobs": model_jobs,
        "tasks_to_run_setup": TabArenaMetadataBundle(task_metadata=[]),
        "experiment_bundle": TabArenaExperimentBundle(n_random_configs=0, preprocessing_pipelines=["default"]),
        "path_setup": PathSetup(workspace="/ws", python_path="/py"),
        "scheduler_setup": _slurm(),
        "resources_setup": _resources(),
    }
    defaults.update(kwargs)
    return TabArenaBenchmarkPlan(**defaults)


# ---------------------------------------------------------------------------
# SingleModel
# ---------------------------------------------------------------------------


class TestSingleModel:
    def test_from_input_passthrough(self):
        sm = SingleModel("X", 5)
        assert SingleModel.from_input(sm) is sm

    def test_from_input_string(self):
        sm = SingleModel.from_input("X")
        assert sm == SingleModel(name="X", n_configs=0)

    def test_from_input_tuple(self):
        assert SingleModel.from_input(("X", 5)) == SingleModel(name="X", n_configs=5)

    def test_from_input_invalid(self):
        with pytest.raises(TypeError):
            SingleModel.from_input(123)

    def test_to_entry(self):
        assert SingleModel("X", "all").to_entry() == ("X", "all")


# ---------------------------------------------------------------------------
# ModelJob normalization
# ---------------------------------------------------------------------------


class TestModelJobNormalization:
    def test_single_tuple_wrapped(self):
        job = ModelJob(models=("X", 5))
        assert job.models == [SingleModel("X", 5)]

    def test_single_string_wrapped(self):
        job = ModelJob(models="X")
        assert job.models == [SingleModel("X", 0)]

    def test_list_of_entries(self):
        job = ModelJob(models=[("X", 5), "Y", SingleModel("Z", 1)])
        assert job.models == [SingleModel("X", 5), SingleModel("Y", 0), SingleModel("Z", 1)]

    def test_model_entries_to_tuples(self):
        job = ModelJob(models=[("X", 5), "Y"])
        assert job._model_entries() == [("X", 5), ("Y", 0)]


# ---------------------------------------------------------------------------
# _apply_overrides
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    def test_empty_returns_same_object(self):
        base = _resources()
        assert _apply_overrides(base, {}, "resources") is base

    def test_applies_override(self):
        base = _resources(num_gpus=0)
        out = _apply_overrides(base, {"num_gpus": 2}, "resources")
        assert out.num_gpus == 2
        assert base.num_gpus == 0  # base untouched

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown resources override"):
            _apply_overrides(_resources(), {"not_a_field": 1}, "resources")


# ---------------------------------------------------------------------------
# TabArenaBenchmarkPlan.build_setups
# ---------------------------------------------------------------------------


class TestBuildSetups:
    def test_identical_settings_merge_into_one_setup(self):
        plan = _plan([ModelJob(models=("A", "all")), ModelJob(models=("B", 0))])
        setups = plan.build_setups()
        assert len(setups) == 1
        assert setups[0].experiment_bundle.models == [("A", "all"), ("B", 0)]

    def test_differing_resources_split_into_two_setups(self):
        plan = _plan(
            [
                ModelJob(models=("A", "all")),
                ModelJob(models=("B", 0), resources={"num_gpus": 1}),
            ]
        )
        setups = plan.build_setups()
        assert len(setups) == 2
        gpu_counts = sorted(s.resources_setup.num_gpus for s in setups)
        assert gpu_counts == [0, 1]

    def test_scheduler_override_lands_and_base_untouched(self):
        plan = _plan([ModelJob(models=("A", 0), scheduler={"gpu_partition": "X"})])
        setup = plan.build_setups()[0]
        assert setup.scheduler_setup.gpu_partition == "X"
        assert plan.scheduler_setup.gpu_partition == "gpu_part"

    def test_tasks_override_lands(self):
        plan = _plan([ModelJob(models=("A", 0), tasks={"n_train_samples_to_run": (0, 1000)})])
        setup = plan.build_setups()[0]
        assert setup.tasks_to_run_setup.n_train_samples_to_run == (0, 1000)

    def test_experiment_override_lands_and_models_preserved(self):
        plan = _plan([ModelJob(models=("A", 0), experiment={"model_agnostic_preprocessing": False})])
        setup = plan.build_setups()[0]
        assert setup.experiment_bundle.model_agnostic_preprocessing is False
        assert setup.experiment_bundle.models == [("A", 0)]

    def test_experiment_override_models_key_forbidden(self):
        plan = _plan([ModelJob(models=("A", 0), experiment={"models": [("B", 0)]})])
        with pytest.raises(ValueError, match="must not set 'models'"):
            plan.build_setups()

    def test_named_group_sets_safe_benchmark_name(self):
        plan = _plan([ModelJob(models=("A", 0), name="cpu")])
        assert plan.build_setups()[0].parallel_safe_benchmark_name == "my_bench_cpu"

    def test_unnamed_groups_auto_named_by_index(self):
        plan = _plan(
            [
                ModelJob(models=("A", 0)),
                ModelJob(models=("B", 0), resources={"num_gpus": 1}),
            ]
        )
        names = [s.parallel_safe_benchmark_name for s in plan.build_setups()]
        assert names == ["my_bench_group0", "my_bench_group1"]

    def test_conflicting_names_in_merged_group_raise(self):
        plan = _plan([ModelJob(models=("A", 0), name="x"), ModelJob(models=("B", 0), name="y")])
        with pytest.raises(ValueError, match="conflicting"):
            plan.build_setups()

    def test_one_named_one_unnamed_merge_uses_the_name(self):
        plan = _plan([ModelJob(models=("A", 0)), ModelJob(models=("B", 0), name="cpu")])
        setups = plan.build_setups()
        assert len(setups) == 1
        assert setups[0].parallel_safe_benchmark_name == "my_bench_cpu"

    def test_benchmark_name_threaded_through(self):
        setup = _plan([ModelJob(models=("A", 0))]).build_setups()[0]
        assert setup.benchmark_name == "my_bench"

    def test_ignore_cache_is_per_model(self):
        plan = _plan([ModelJob(models=("A", 0), ignore_cache=True)])
        assert plan.build_setups()[0].ignore_cache is True

    def test_ignore_cache_difference_splits_groups(self):
        plan = _plan(
            [
                ModelJob(models=("A", 0)),
                ModelJob(models=("B", 0), ignore_cache=True),
            ]
        )
        setups = plan.build_setups()
        assert len(setups) == 2
        assert sorted(s.ignore_cache for s in setups) == [False, True]

    def test_num_ray_cpus_forwarded_from_build_setups(self):
        setup = _plan([ModelJob(models=("A", 0))]).build_setups(num_ray_cpus=3)[0]
        assert setup.num_ray_cpus == 3
