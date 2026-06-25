"""Unit tests for the tabflow_slurm benchmark-setup components.

Ported from the pre-refactor `test_setup_slurm_base_v2.py`, which tested a single
monolithic `BenchmarkSetup2026`. That class is now split into focused pieces:
    - `PathSetup`                          (setup/paths.py)
    - `SlurmSetup`                         (setup/scheduler.py)
    - `ResourcesSetup.time_limit_per_config` (setup/resources.py)
    - `TabArenaBenchmarkSetup`             (setup/benchmark.py)

The task-metadata loading/filtering moved into `TaskMetadataCollection`
(tabarena core) and is covered by `tst/benchmark/task/test_collection_subset_tasks.py`.
"""

from __future__ import annotations

import json

import pytest

from tabarena.benchmark.experiment import TabArenaExperimentBundle
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.contexts.abstract_arena_context import AbstractArenaContext

# Import a real submodule (not the bare `tabflow_slurm` namespace): when the package
# is not installed, the repo-root workspace dir is importable as an empty namespace
# package, so `importorskip("tabflow_slurm")` would NOT skip. A submodule does.
pytest.importorskip("tabflow_slurm.setup", reason="tabflow_slurm is not installed")

from tabflow_slurm.setup.benchmark import TabArenaBenchmarkSetup
from tabflow_slurm.setup.paths import PathSetup
from tabflow_slurm.setup.resources import ResourcesSetup
from tabflow_slurm.setup.scheduler import SlurmSetup

# ---------------------------------------------------------------------------
# PathSetup
# ---------------------------------------------------------------------------


class TestPathSetup:
    def test_run_script_path_defaults_to_bundled_runner(self):
        ps = PathSetup(workspace="/ws", python_path="/py")
        assert ps.run_script_path.endswith("run_tabarena_experiment.py")

    def test_submit_script_path_defaults_to_bundled_template(self):
        ps = PathSetup(workspace="/ws", python_path="/py")
        assert ps.submit_script_path.endswith("submit_template.sh")

    def test_run_script_path_custom(self):
        ps = PathSetup(workspace="/ws", python_path="/py", run_script="/custom/run.py")
        assert ps.run_script_path == "/custom/run.py"

    def test_openml_cache_path_auto(self):
        ps = PathSetup(workspace="/ws", python_path="/py", openml_cache="auto")
        assert ps.openml_cache_path == "auto"

    def test_openml_cache_path_defaults_to_auto(self):
        ps = PathSetup(workspace="/ws", python_path="/py")
        assert ps.openml_cache_path == "auto"

    def test_openml_cache_path_none_is_under_workspace(self):
        ps = PathSetup(workspace="/ws", python_path="/py", openml_cache=None)
        assert ps.openml_cache_path == "/ws/.openml-cache"

    def test_openml_cache_path_custom(self):
        ps = PathSetup(workspace="/ws", python_path="/py", openml_cache="/data/cache")
        assert ps.openml_cache_path == "/data/cache"

    def test_get_job_batch_dir_contains_safe_name(self):
        ps = PathSetup(workspace="/ws", python_path="/py")
        path = ps.get_job_batch_dir(benchmark_name="bench", safe_benchmark_name="bench_p1")
        assert "bench_p1" in path

    def test_get_job_batch_dir_under_workspace_setup_out(self):
        ps = PathSetup(workspace="/ws", python_path="/py")
        path = ps.get_job_batch_dir(benchmark_name="bench", safe_benchmark_name="bench")
        assert path.startswith("/ws/setup_out/bench/")

    def test_get_slurm_job_json_path_contains_safe_name_and_is_json(self):
        ps = PathSetup(workspace="/ws", python_path="/py")
        path = ps.get_slurm_job_json_path(benchmark_name="bench", safe_benchmark_name="bench_p1")
        assert "bench_p1" in path
        assert path.endswith(".json")

    def test_get_output_path_under_workspace(self):
        ps = PathSetup(workspace="/ws", python_path="/py")
        assert ps.get_output_path("my_bench") == "/ws/output/my_bench"

    def test_get_slurm_log_output_path_under_workspace(self):
        ps = PathSetup(workspace="/ws", python_path="/py")
        assert ps.get_slurm_log_output_path("my_bench") == "/ws/slurm_out/my_bench"

    def test_get_setup_out_path_under_workspace(self):
        ps = PathSetup(workspace="/ws", python_path="/py")
        assert str(ps.get_setup_out_path("my_bench")) == "/ws/setup_out/my_bench"

    def test_ensure_runtime_dirs_creates_directories(self, tmp_path):
        # openml_cache=None -> the cache dir is created under the workspace too.
        ps = PathSetup(workspace=tmp_path, python_path="/py", openml_cache=None)
        ps.ensure_runtime_dirs("bench")
        assert (tmp_path / "output" / "bench").is_dir()
        assert (tmp_path / "slurm_out" / "bench").is_dir()
        assert (tmp_path / "setup_out" / "bench").is_dir()
        assert (tmp_path / ".openml-cache").is_dir()


# ---------------------------------------------------------------------------
# SlurmSetup
# ---------------------------------------------------------------------------


def _slurm(**kw) -> SlurmSetup:
    """SlurmSetup with the (now required) cluster partitions filled in."""
    kw.setdefault("gpu_partition", "gpu_part")
    kw.setdefault("cpu_partition", "cpu_part")
    kw.setdefault("extra_gres", "localtmp:100")
    return SlurmSetup(**kw)


class TestSlurmSetup:
    def test_partitions_and_gres_are_required(self):
        # gpu_partition / cpu_partition / extra_gres have no defaults (cluster-specific).
        with pytest.raises(TypeError):
            SlurmSetup()

    def test_default_mem_per_handle_is_false(self):
        assert _slurm().mem_per_handle is False

    def test_default_exclusive_node_is_false(self):
        assert _slurm().exclusive_node is False

    def test_default_time_limit_overhead(self):
        assert _slurm().time_limit_overhead == 1

    def test_default_bundle_size(self):
        assert _slurm().bundle_size == 5

    def test_default_setup_ray_for_slurm(self):
        assert _slurm().setup_ray_for_slurm_shared_resources_environment is True

    def test_custom_values(self):
        ss = SlurmSetup(
            gpu_partition="g",
            cpu_partition="c",
            extra_gres=None,
            mem_per_handle=True,
            exclusive_node=True,
            time_limit_overhead=2,
            bundle_size=10,
        )
        assert ss.gpu_partition == "g"
        assert ss.cpu_partition == "c"
        assert ss.extra_gres is None
        assert ss.mem_per_handle is True
        assert ss.exclusive_node is True
        assert ss.time_limit_overhead == 2
        assert ss.bundle_size == 10


# ---------------------------------------------------------------------------
# SchedulerSetup bundling (bundle_size + large-dataset auto-rule)
# ---------------------------------------------------------------------------


def _split(*, n_features=10, n_samples_train_per_fold=1000):
    from tabarena.benchmark.task.metadata import SplitMetadata

    return SplitMetadata(
        repeat=0,
        fold=0,
        num_instances_train=n_samples_train_per_fold,
        num_instances_test=100,
        num_instance_groups_train=n_samples_train_per_fold,
        num_instance_groups_test=100,
        num_classes_train=2,
        num_classes_test=2,
        num_features_train=n_features,
        num_features_test=n_features,
    )


def _sched_job(*, name="cfg", dataset="d", fold=0):
    """A scheduler-facing job; bundling only reads `experiment.name` and `task`."""
    from types import SimpleNamespace

    from tabarena.benchmark.experiment import Job

    return Job.create(SimpleNamespace(name=name), dataset, fold=fold)


def _shape_collection(specs: dict[str, dict]):
    """A collection of one-split datasets with the given shapes ({dataset: shape kwargs})."""
    import pandas as pd

    from tabarena.benchmark.task.metadata import TaskMetadataCollection

    df = pd.DataFrame(
        {
            "tid": list(range(1, len(specs) + 1)),
            "dataset": list(specs),
            "problem_type": ["binary"] * len(specs),
            "n_folds": [1] * len(specs),
            "n_repeats": [1] * len(specs),
            "n_features": [shape.get("n_features", 10) for shape in specs.values()],
            "n_classes": [2] * len(specs),
            "NumberOfInstances": [100] * len(specs),
            "n_samples_train_per_fold": [
                float(shape.get("n_samples_train_per_fold", 1000)) for shape in specs.values()
            ],
            "n_samples_test_per_fold": [100.0] * len(specs),
        },
    )
    return TaskMetadataCollection.from_legacy_df(df)


class TestEffectiveBundleSize:
    def test_default_large_thresholds(self):
        ss = _slurm()
        assert ss.large_dataset_n_features == 5_000
        assert ss.large_dataset_n_samples == 100_000

    def test_small_dataset_uses_bundle_size(self):
        ss = _slurm(bundle_size=5)
        assert ss._effective_bundle_size(_sched_job(), split=_split(n_features=10, n_samples_train_per_fold=1000)) == 5

    def test_wide_dataset_collapses_to_one(self):
        ss = _slurm(bundle_size=5)
        assert ss._effective_bundle_size(_sched_job(), split=_split(n_features=5001)) == 1

    def test_large_sample_dataset_collapses_to_one(self):
        ss = _slurm(bundle_size=5)
        assert ss._effective_bundle_size(_sched_job(), split=_split(n_samples_train_per_fold=100_001)) == 1

    def test_at_threshold_is_not_large(self):
        ss = _slurm(bundle_size=5)
        assert (
            ss._effective_bundle_size(_sched_job(), split=_split(n_features=5000, n_samples_train_per_fold=100_000))
            == 5
        )

    def test_per_dataset_override_wins_over_auto_rule(self):
        ss = _slurm(bundle_size=5, bundle_size_per_dataset={"big": 3})
        assert ss._effective_bundle_size(_sched_job(dataset="big"), split=_split(n_features=5001)) == 3

    def test_feature_check_disabled(self):
        ss = _slurm(bundle_size=5, large_dataset_n_features=None)
        assert ss._effective_bundle_size(_sched_job(), split=_split(n_features=5001)) == 5

    def test_sample_check_disabled(self):
        ss = _slurm(bundle_size=5, large_dataset_n_samples=None)
        assert ss._effective_bundle_size(_sched_job(), split=_split(n_samples_train_per_fold=100_001)) == 5


class TestBundleItems:
    def test_large_datasets_get_singleton_bundles(self):
        ss = _slurm(bundle_size=5)
        collection = _shape_collection({"small": {}, "wide": {"n_features": 6000}})
        jobs = [_sched_job(name=f"cfg_{i}", dataset="small") for i in range(3)] + [
            _sched_job(name=f"cfg_{i}", dataset="wide") for i in range(2)
        ]
        array_tasks, max_configs = ss.bundle_items(jobs, collection)
        sizes = sorted(len(j["items"]) for j in array_tasks)
        # 3 small batched into one bundle, 2 wide as singletons.
        assert sizes == [1, 1, 3]
        assert max_configs == 3
        # Items carry the self-describing coordinates.
        first = next(j for j in array_tasks if len(j["items"]) == 3)["items"][0]
        assert set(first) == {"experiment", "dataset", "fold", "repeat"}


# ---------------------------------------------------------------------------
# ResourcesSetup.time_limit_per_config
# ---------------------------------------------------------------------------


def _resources(**kw) -> ResourcesSetup:
    """ResourcesSetup with the (now required) compute fields filled in."""
    kw.setdefault("num_cpus", 8)
    kw.setdefault("num_gpus", 0)
    kw.setdefault("memory_limit", 32)
    return ResourcesSetup(**kw)


class TestTimeLimitPerConfig:
    def test_no_preprocessing_overhead(self):
        rs = _resources(
            time_limit=3600,
            time_limit_for_model_agnostic_preprocessing=None,
            time_limit_with_model_agnostic_preprocessing=False,
        )
        assert rs.time_limit_per_config == 3600

    def test_adds_preprocessing_time(self):
        rs = _resources(
            time_limit=3600,
            time_limit_for_model_agnostic_preprocessing=300,
            time_limit_with_model_agnostic_preprocessing=False,
        )
        assert rs.time_limit_per_config == 3900

    def test_adds_constant_overhead_when_flag_true(self):
        rs = _resources(
            time_limit=3600,
            time_limit_for_model_agnostic_preprocessing=None,
            time_limit_with_model_agnostic_preprocessing=True,
        )
        assert rs.time_limit_per_config == 3600 + 60 * 15

    def test_both_overheads_combined(self):
        rs = _resources(
            time_limit=3600,
            time_limit_for_model_agnostic_preprocessing=300,
            time_limit_with_model_agnostic_preprocessing=True,
        )
        assert rs.time_limit_per_config == 3600 + 300 + 60 * 15

    def test_zero_time_limit(self):
        rs = _resources(
            time_limit=0,
            time_limit_for_model_agnostic_preprocessing=None,
            time_limit_with_model_agnostic_preprocessing=False,
        )
        assert rs.time_limit_per_config == 0


# ---------------------------------------------------------------------------
# TabArenaBenchmarkSetup._safe_benchmark_name
# ---------------------------------------------------------------------------


def _context(tasks: TaskMetadataCollection | None = None) -> AbstractArenaContext:
    """A minimal, baseline-free arena context over the given (or empty) collection."""
    return AbstractArenaContext(methods=[], task_metadata=tasks if tasks is not None else TaskMetadataCollection([]))


def _benchmark_setup(**kwargs) -> TabArenaBenchmarkSetup:
    defaults = {
        "benchmark_name": "my_bench",
        "context": _context(),
        "experiment_bundle": TabArenaExperimentBundle(n_random_configs=0, preprocessing_pipelines=["default"]),
        "path_setup": PathSetup(workspace="/ws", python_path="/py"),
        "scheduler_setup": _slurm(),
        "resources_setup": _resources(time_limit=3600),
    }
    defaults.update(kwargs)
    return TabArenaBenchmarkSetup(**defaults)


class TestSafeBenchmarkName:
    def test_no_parallel_name_falls_back_to_benchmark_name(self):
        bs = _benchmark_setup(benchmark_name="my_bench", parallel_safe_benchmark_name=None)
        assert bs._safe_benchmark_name == "my_bench"

    def test_parallel_safe_name_used_directly(self):
        bs = _benchmark_setup(benchmark_name="my_bench", parallel_safe_benchmark_name="my_bench_run1")
        assert bs._safe_benchmark_name == "my_bench_run1"


# ---------------------------------------------------------------------------
# TabArenaBenchmarkSetup.get_jobs_to_run (ignore_cache path: no Ray)
# ---------------------------------------------------------------------------


def _two_dataset_collection() -> TaskMetadataCollection:
    import pandas as pd

    df = pd.DataFrame(
        {
            "tid": [1, 2],
            "dataset": ["ds_a", "ds_b"],
            "problem_type": ["binary", "binary"],
            "n_folds": [2, 1],
            "n_repeats": [1, 1],
            "n_features": [5, 5],
            "n_classes": [2, 2],
            "NumberOfInstances": [100, 100],
            "n_samples_train_per_fold": [80.0, 80.0],
            "n_samples_test_per_fold": [20.0, 20.0],
        },
    )
    return TaskMetadataCollection.from_legacy_df(df)


def _passthrough_experiment(name: str):
    from autogluon.tabular.models import LGBModel

    from tabarena.benchmark.experiment import AGModelBagExperiment

    return AGModelBagExperiment(
        name=name,
        model_cls=LGBModel,
        model_hyperparameters={},
        num_bag_folds=2,
        time_limit=60,
    )


class TestGetJobsToRun:
    def test_enumerates_saves_batch_and_bundles(self, tmp_path):
        from tabarena.benchmark.experiment import JobBatch

        bs = _benchmark_setup(
            context=_context(_two_dataset_collection()),
            experiment_bundle=TabArenaExperimentBundle(
                models=[_passthrough_experiment("exp_a")],
                n_random_configs=0,
                preprocessing_pipelines=["default"],
            ),
            path_setup=PathSetup(workspace=str(tmp_path), python_path="/py"),
            scheduler_setup=_slurm(bundle_size=2),
            ignore_cache=True,  # skip the Ray cache check
        )
        jobs, max_configs_per_job = bs.get_jobs_to_run()

        # 1 experiment x 3 splits (ds_a: 2 folds, ds_b: 1 fold) -> 3 items in size-2 bundles.
        items = [item for job in jobs for item in job["items"]]
        assert [(i["experiment"], i["dataset"], i["fold"], i["repeat"]) for i in items] == [
            ("exp_a", "ds_a", 0, 0),
            ("exp_a", "ds_a", 1, 0),
            ("exp_a", "ds_b", 0, 0),
        ]
        assert max_configs_per_job == 2

        # The self-contained JobBatch artifact is on disk and loadable.
        batch = JobBatch.load(bs._job_batch_dir)
        assert [j.task.as_triple() for j in batch.jobs] == [("ds_a", 0, 0), ("ds_a", 1, 0), ("ds_b", 0, 0)]
        assert [e.name for e in batch.experiments] == ["exp_a"]
        assert set(batch.task_metadata.dataset_names()) == {"ds_a", "ds_b"}

        # The per-job defaults point the runner at the batch.
        assert bs._build_default_args()["job_batch_dir"] == bs._job_batch_dir


# ---------------------------------------------------------------------------
# SlurmSetup._build_sbatch_prefix (pure computation)
# ---------------------------------------------------------------------------


def _prefix(
    *,
    num_gpus: int = 0,
    num_cpus: int = 8,
    memory_limit: int | None = 32,
    time_limit: int = 3600,
    configs_per_job: int = 5,
    time_limit_overhead: int = 1,
    extra_gres: str | None = "localtmp:100",
    exclusive_node: bool = False,
    mem_per_handle: bool = False,
    slurm_log_output: str = "/logs/bench",
    slurm_script_path: str = "/scripts/submit.sh",
) -> str:
    slurm = SlurmSetup(
        gpu_partition="gpu_part",
        cpu_partition="cpu_part",
        extra_gres=extra_gres,
        exclusive_node=exclusive_node,
        mem_per_handle=mem_per_handle,
        time_limit_overhead=time_limit_overhead,
    )
    resources = ResourcesSetup(
        time_limit=time_limit,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        memory_limit=memory_limit,
    )
    return slurm._build_sbatch_prefix(
        resources_setup=resources,
        configs_per_job=configs_per_job,
        slurm_log_output=slurm_log_output,
        slurm_script_path=slurm_script_path,
    )


class TestBuildSbatchPrefix:
    def test_cpu_job_uses_cpu_partition(self):
        assert "cpu_part" in _prefix(num_gpus=0)

    def test_gpu_job_uses_gpu_partition(self):
        assert "gpu_part" in _prefix(num_gpus=1)

    def test_gpu_gres_included(self):
        assert "gpu:2" in _prefix(num_gpus=2)

    def test_no_gres_flag_when_cpu_and_no_extra_gres(self):
        assert "--gres" not in _prefix(num_gpus=0, extra_gres=None)

    def test_extra_gres_included(self):
        assert "localtmp:50" in _prefix(num_gpus=0, extra_gres="localtmp:50")

    def test_time_limit_computed_in_hours(self):
        # 3600s/config // 3600 * 5 configs + 1h overhead = 6h
        assert "--time=6:00:00" in _prefix(time_limit=3600, configs_per_job=5, time_limit_overhead=1)

    def test_time_limit_overhead_applied(self):
        # 7200 // 3600 * 1 + 2 = 4h
        assert "--time=4:00:00" in _prefix(time_limit=7200, configs_per_job=1, time_limit_overhead=2)

    def test_exclusive_node_uses_mem_0_nodes_1(self):
        cmd = _prefix(exclusive_node=True)
        assert "--mem=0" in cmd
        assert "--nodes=1" in cmd
        assert "--exclusive" in cmd

    def test_exclusive_node_omits_cpus_per_task(self):
        assert "--cpus-per-task" not in _prefix(exclusive_node=True)

    def test_non_exclusive_includes_cpus_per_task(self):
        assert "--cpus-per-task=4" in _prefix(exclusive_node=False, num_cpus=4)

    def test_non_exclusive_cpu_memory_per_job(self):
        assert "--mem=32G" in _prefix(num_gpus=0, mem_per_handle=False, memory_limit=32)

    def test_mem_per_handle_cpu_uses_mem_per_cpu(self):
        assert "--mem-per-cpu=8G" in _prefix(num_gpus=0, num_cpus=4, mem_per_handle=True, memory_limit=32)

    def test_mem_per_handle_gpu_uses_mem_per_gpu(self):
        assert "--mem-per-gpu=20G" in _prefix(num_gpus=2, mem_per_handle=True, memory_limit=40)


# ---------------------------------------------------------------------------
# SlurmSetup._write_job_batches_and_build_commands (per-bundle-size time budget)
# ---------------------------------------------------------------------------


def _job(size: int, bundle_size: int | None = None) -> dict:
    """An array-task bundle with `size` (placeholder) items.

    `bundle_size` is the target size the scheduler groups on; it defaults to
    `size` (a full bundle). Pass a larger `bundle_size` to model a remainder
    bundle (fewer items than the size group it belongs to).
    """
    return {
        "bundle_size": size if bundle_size is None else bundle_size,
        "items": [{"experiment": f"cfg_{i}", "dataset": "d", "fold": 0, "repeat": 0} for i in range(size)],
    }


def _build_commands(jobs: list[dict], *, tmp_path, time_limit=3600, time_limit_overhead=1, **slurm_kw) -> list[str]:
    slurm = _slurm(time_limit_overhead=time_limit_overhead, **slurm_kw)
    resources = ResourcesSetup(time_limit=time_limit, num_cpus=8, num_gpus=0, memory_limit=32)
    return slurm._write_job_batches_and_build_commands(
        all_jobs=jobs,
        defaults={"k": "v"},
        base_json_path=str(tmp_path / "jobs.json"),
        resources_setup=resources,
        slurm_log_output=str(tmp_path / "logs"),
        slurm_script_path=str(tmp_path / "submit.sh"),
    )


class TestWriteJobBatches:
    def test_single_size_keeps_base_path_and_one_command(self, tmp_path):
        cmds = _build_commands([_job(5), _job(5)], tmp_path=tmp_path)
        assert len(cmds) == 1
        assert str(tmp_path / "jobs.json") in cmds[0]
        # 3600s // 3600 * 5 configs + 1h overhead = 6h.
        assert "--time=6:00:00" in cmds[0]

    def test_mixed_sizes_get_separate_arrays_with_own_time(self, tmp_path):
        # Two singleton (large-dataset) tasks + one 5-config bundle.
        cmds = _build_commands([_job(1), _job(1), _job(5)], tmp_path=tmp_path)
        assert len(cmds) == 2
        by_time = {next(p for p in c.split() if p.startswith("--time=")): c for c in cmds}
        # Singletons budgeted for 1 config (2h), not the 5-config budget (6h).
        assert "--time=2:00:00" in by_time
        assert "--time=6:00:00" in by_time
        # The two arrays write to distinct, size-suffixed JSON files.
        assert "_size1.json" in by_time["--time=2:00:00"]
        assert "_size5.json" in by_time["--time=6:00:00"]
        assert (tmp_path / "jobs_size1.json").exists()
        assert (tmp_path / "jobs_size5.json").exists()

    def test_singleton_array_holds_only_its_two_tasks(self, tmp_path):
        _build_commands([_job(1), _job(1), _job(5)], tmp_path=tmp_path)
        with (tmp_path / "jobs_size1.json").open() as f:
            assert len(json.load(f)["jobs"]) == 2

    def test_remainder_bundle_stays_in_its_size_group(self, tmp_path):
        # Two full size-10 bundles + one size-10 remainder holding only 8 items:
        # grouping is by target bundle size, so it is a single array, not two.
        cmds = _build_commands([_job(10), _job(10), _job(8, bundle_size=10)], tmp_path=tmp_path)
        assert len(cmds) == 1
        assert str(tmp_path / "jobs.json") in cmds[0]
        # Budgeted for the 10-config target (11h), and the remainder over-allocates.
        assert "--time=11:00:00" in cmds[0]
        # All three tasks land in the one shipped file, stripped of the hint.
        with (tmp_path / "jobs.json").open() as f:
            shipped = json.load(f)["jobs"]
        assert len(shipped) == 3
        assert all(set(job) == {"items"} for job in shipped)

    def test_oversized_single_size_splits_into_batches(self, tmp_path):
        cmds = _build_commands([_job(5) for _ in range(3)], tmp_path=tmp_path, max_array_size=2)
        assert len(cmds) == 2
        assert (tmp_path / "jobs_batch0.json").exists()
        assert (tmp_path / "jobs_batch1.json").exists()

    def test_slurm_log_output_in_command(self):
        assert "/my/logs/bench" in _prefix(slurm_log_output="/my/logs/bench")

    def test_script_path_in_command(self):
        assert "/path/to/submit.sh" in _prefix(slurm_script_path="/path/to/submit.sh")

    def test_returns_string(self):
        assert isinstance(_prefix(), str)

    def test_gpu_job_gres_combined_with_extra(self):
        cmd = _prefix(num_gpus=1, extra_gres="localtmp:100")
        assert "gpu:1" in cmd
        assert "localtmp:100" in cmd
