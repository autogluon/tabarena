"""Unit tests for the tabflow_slurm benchmark-setup components.

Ported from the pre-refactor `test_setup_slurm_base_v2.py`, which tested a single
monolithic `BenchmarkSetup2026`. That class is now split into focused pieces:
    - `PathSetup`                          (setup/paths.py)
    - `SlurmSetup`                         (setup/scheduler.py)
    - `ResourcesSetup.time_limit_per_config` (setup/resources.py)
    - `TabArenaBenchmarkSetup`             (setup/benchmark.py)

The task-metadata loading/filtering moved into `TabArenaMetadataBundle`
(tabarena core) and is covered by `tst/benchmark/task/test_metadata_bundle.py`.
"""

from __future__ import annotations

import pytest
from tabarena.benchmark.experiment import TabArenaExperimentBundle
from tabarena.benchmark.task.metadata import TabArenaMetadataBundle

# Import a real submodule (not the bare `tabflow_slurm` namespace): when the package
# is not installed, the repo-root workspace dir is importable as an empty namespace
# package, so `importorskip("tabflow_slurm")` would NOT skip. A submodule does.
pytest.importorskip("tabflow_slurm.setup", reason="tabflow_slurm is not installed")

from tabflow_slurm.setup.benchmark import TabArenaBenchmarkSetup  # noqa: E402
from tabflow_slurm.setup.candidates import JobCandidate  # noqa: E402
from tabflow_slurm.setup.paths import PathSetup  # noqa: E402
from tabflow_slurm.setup.resources import ResourcesSetup  # noqa: E402
from tabflow_slurm.setup.scheduler import SlurmSetup  # noqa: E402

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

    def test_get_configs_path_contains_safe_name_and_is_yaml(self):
        ps = PathSetup(workspace="/ws", python_path="/py")
        path = ps.get_configs_path(benchmark_name="bench", safe_benchmark_name="bench_p1")
        assert "bench_p1" in path
        assert path.endswith(".yaml")

    def test_get_configs_path_under_workspace_setup_out(self):
        ps = PathSetup(workspace="/ws", python_path="/py")
        path = ps.get_configs_path(benchmark_name="bench", safe_benchmark_name="bench")
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


def _candidate(*, dataset_name="d", n_features=10, n_samples_train_per_fold=1000) -> JobCandidate:
    return JobCandidate(
        task_id="1",
        dataset_name=dataset_name,
        fold=0,
        repeat=0,
        config_index=0,
        config={"name": "cfg"},
        n_features=n_features,
        n_classes=2,
        n_samples_train_per_fold=n_samples_train_per_fold,
        problem_type="binary",
    )


class TestEffectiveBundleSize:
    def test_default_large_thresholds(self):
        ss = _slurm()
        assert ss.large_dataset_n_features == 5_000
        assert ss.large_dataset_n_samples == 100_000

    def test_small_dataset_uses_bundle_size(self):
        ss = _slurm(bundle_size=5)
        assert ss._effective_bundle_size(_candidate(n_features=10, n_samples_train_per_fold=1000)) == 5

    def test_wide_dataset_collapses_to_one(self):
        ss = _slurm(bundle_size=5)
        assert ss._effective_bundle_size(_candidate(n_features=5001)) == 1

    def test_large_sample_dataset_collapses_to_one(self):
        ss = _slurm(bundle_size=5)
        assert ss._effective_bundle_size(_candidate(n_samples_train_per_fold=100_001)) == 1

    def test_at_threshold_is_not_large(self):
        ss = _slurm(bundle_size=5)
        assert ss._effective_bundle_size(_candidate(n_features=5000, n_samples_train_per_fold=100_000)) == 5

    def test_per_dataset_override_wins_over_auto_rule(self):
        ss = _slurm(bundle_size=5, bundle_size_per_dataset={"big": 3})
        assert ss._effective_bundle_size(_candidate(dataset_name="big", n_features=5001)) == 3

    def test_feature_check_disabled(self):
        ss = _slurm(bundle_size=5, large_dataset_n_features=None)
        assert ss._effective_bundle_size(_candidate(n_features=5001)) == 5

    def test_sample_check_disabled(self):
        ss = _slurm(bundle_size=5, large_dataset_n_samples=None)
        assert ss._effective_bundle_size(_candidate(n_samples_train_per_fold=100_001)) == 5


class TestBundleItems:
    def test_large_datasets_get_singleton_bundles(self):
        ss = _slurm(bundle_size=5)
        small = [_candidate(dataset_name="small") for _ in range(3)]
        wide = [_candidate(dataset_name="wide", n_features=6000) for _ in range(2)]
        jobs, max_configs = ss.bundle_items([*small, *wide])
        sizes = sorted(len(j["items"]) for j in jobs)
        # 3 small batched into one bundle, 2 wide as singletons.
        assert sizes == [1, 1, 3]
        assert max_configs == 3


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


def _benchmark_setup(**kwargs) -> TabArenaBenchmarkSetup:
    defaults = {
        "benchmark_name": "my_bench",
        "tasks_to_run_setup": TabArenaMetadataBundle(task_metadata=[]),
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
