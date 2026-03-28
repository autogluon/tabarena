from __future__ import annotations

import pytest

# setup_slurm_base_v2 imports ray and other heavy deps at the top level.
# Skip the entire module gracefully when those are unavailable.
pytest.importorskip(
    "ray", reason="ray not installed — skipping setup_slurm_base_v2 tests"
)
setup_mod = pytest.importorskip(
    "tabflow_slurm.setup_slurm_base_v2",
    reason="tabflow_slurm.setup_slurm_base_v2 import failed",
)

from tabarena.benchmark.task.user_task import (  # noqa: E402
    SplitMetadata,
    TabArenaTaskMetadata,
)
from tabflow_slurm.setup_slurm_base_v2 import (  # noqa: E402
    BenchmarkSetup2026,
    PathSetup,
    SlurmSetup,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_meta(repeat: int = 0, fold: int = 0) -> SplitMetadata:
    return SplitMetadata(
        repeat=repeat,
        fold=fold,
        num_instances_train=80,
        num_instances_test=20,
        num_instance_groups_train=80,
        num_instance_groups_test=20,
        num_classes_train=2,
        num_classes_test=2,
        num_features_train=5,
        num_features_test=5,
    )


def _task_meta(
    problem_type: str = "binary",
    task_id_str: str | None = "360",
    n_splits: int = 1,
    dataset_name: str = "test_ds",
) -> TabArenaTaskMetadata:
    splits: dict = {}
    for i in range(n_splits):
        sm = _split_meta(repeat=0, fold=i)
        splits[sm.split_index] = sm
    return TabArenaTaskMetadata(
        dataset_name=dataset_name,
        problem_type=problem_type,
        is_classification=(problem_type != "regression"),
        target_name="target",
        eval_metric="roc_auc",
        splits_metadata=splits,
        split_time_horizon=None,
        split_time_horizon_unit=None,
        stratify_on=None,
        time_on=None,
        group_on=None,
        group_time_on=None,
        group_labels=None,
        multiclass_min_n_classes_over_splits=2,
        multiclass_max_n_classes_over_splits=2,
        class_consistency_over_splits=True,
        num_instances=100,
        num_features=5,
        num_classes=2,
        num_instance_groups=100,
        tabarena_task_name="test_task",
        task_id_str=task_id_str,
    )


def _benchmark_setup(**kwargs) -> BenchmarkSetup2026:
    """Minimal BenchmarkSetup2026 with sensible defaults for unit tests."""
    defaults = {
        "benchmark_name": "test_benchmark",
        "task_metadata": [_task_meta()],
        "num_cpus": 4,
        "num_gpus": 0,
        "memory_limit": 16,
        "time_limit": 3600,
    }
    defaults.update(kwargs)
    return BenchmarkSetup2026(**defaults)


# ---------------------------------------------------------------------------
# PathSetup
# ---------------------------------------------------------------------------


class TestPathSetup:
    def test_python_path_contains_venv_name(self):
        ps = PathSetup(base_path="/base/", venv_name="my_venv")
        assert "my_venv" in ps.python_path

    def test_python_path_contains_python_binary(self):
        ps = PathSetup(base_path="/base/", venv_name="my_venv")
        assert ps.python_path.endswith("/python")

    def test_python_path_uses_base_path(self):
        ps = PathSetup(base_path="/custom/base/", venv_name="v1")
        assert ps.python_path.startswith("/custom/base/")

    def test_openml_cache_path_auto(self):
        ps = PathSetup(openml_cache_from_base_path="auto")
        assert ps.openml_cache_path == "auto"

    def test_openml_cache_path_custom(self):
        ps = PathSetup(base_path="/base/", openml_cache_from_base_path=".oml-cache")
        assert ps.openml_cache_path == "/base/.oml-cache"

    def test_openml_cache_path_combines_base_and_relative(self):
        ps = PathSetup(base_path="/data/", openml_cache_from_base_path="cache/")
        assert ps.openml_cache_path == "/data/cache/"

    def test_run_script_path_contains_py_filename(self):
        ps = PathSetup(base_path="/b/", tabarena_repo_name="myrepo")
        assert ps.run_script_path.endswith("run_tabarena_experiment.py")

    def test_run_script_path_contains_repo_name(self):
        ps = PathSetup(base_path="/b/", tabarena_repo_name="myrepo")
        assert "myrepo" in ps.run_script_path

    def test_configs_base_path_contains_repo_name(self):
        ps = PathSetup(tabarena_repo_name="myrepo")
        assert "myrepo" in ps.configs_base_path

    def test_get_slurm_job_json_path_contains_benchmark_name(self):
        ps = PathSetup(base_path="/b/")
        path = ps.get_slurm_job_json_path("my_bench")
        assert "my_bench" in path

    def test_get_slurm_job_json_path_is_json(self):
        ps = PathSetup(base_path="/b/")
        path = ps.get_slurm_job_json_path("bench")
        assert path.endswith(".json")

    def test_get_configs_path_contains_benchmark_name(self):
        ps = PathSetup(base_path="/b/")
        path = ps.get_configs_path("my_bench")
        assert "my_bench" in path

    def test_get_configs_path_is_yaml(self):
        ps = PathSetup(base_path="/b/")
        path = ps.get_configs_path("bench")
        assert path.endswith(".yaml")

    def test_get_output_path_contains_benchmark_name(self):
        ps = PathSetup(base_path="/b/")
        path = ps.get_output_path("my_bench")
        assert "my_bench" in path

    def test_get_output_path_uses_base_path(self):
        ps = PathSetup(base_path="/b/")
        path = ps.get_output_path("bench")
        assert path.startswith("/b/")

    def test_get_slurm_log_output_path_contains_benchmark_name(self):
        ps = PathSetup(base_path="/b/")
        path = ps.get_slurm_log_output_path("my_bench")
        assert "my_bench" in path

    def test_get_slurm_script_path_contains_script_name(self):
        ps = PathSetup(base_path="/b/", tabarena_repo_name="repo")
        path = ps.get_slurm_script_path("submit.sh")
        assert "submit.sh" in path

    def test_get_slurm_script_path_uses_base_path(self):
        ps = PathSetup(base_path="/b/", tabarena_repo_name="repo")
        path = ps.get_slurm_script_path("submit.sh")
        assert path.startswith("/b/")

    def test_default_base_path(self):
        ps = PathSetup()
        assert ps.base_path == "/work/dlclarge2/purucker-tabarena/"

    def test_default_openml_cache_is_not_auto(self):
        ps = PathSetup()
        assert ps.openml_cache_path != "auto"


# ---------------------------------------------------------------------------
# SlurmSetup
# ---------------------------------------------------------------------------


class TestSlurmSetup:
    def test_default_script_name(self):
        ss = SlurmSetup()
        assert ss.script_name == "submit_template.sh"

    def test_default_mem_per_handle_is_false(self):
        ss = SlurmSetup()
        assert ss.mem_per_handle is False

    def test_default_exclusive_node_is_false(self):
        ss = SlurmSetup()
        assert ss.exclusive_node is False

    def test_default_time_limit_overhead(self):
        ss = SlurmSetup()
        assert ss.time_limit_overhead == 1

    def test_default_configs_per_job(self):
        ss = SlurmSetup()
        assert ss.configs_per_job == 5

    def test_default_setup_ray_for_slurm(self):
        ss = SlurmSetup()
        assert ss.setup_ray_for_slurm_shared_resources_environment is True

    def test_custom_values(self):
        ss = SlurmSetup(
            script_name="custom.sh",
            mem_per_handle=True,
            exclusive_node=True,
            time_limit_overhead=2,
            configs_per_job=10,
        )
        assert ss.script_name == "custom.sh"
        assert ss.mem_per_handle is True
        assert ss.exclusive_node is True
        assert ss.time_limit_overhead == 2
        assert ss.configs_per_job == 10


# ---------------------------------------------------------------------------
# BenchmarkSetup2026.time_limit_per_config
# ---------------------------------------------------------------------------


class TestTimeLimitPerConfig:
    def test_no_preprocessing_overhead(self):
        bs = _benchmark_setup(
            time_limit=3600,
            time_limit_for_model_agnostic_preprocessing=None,
            time_limit_with_model_agnostic_preprocessing=False,
        )
        assert bs.time_limit_per_config == 3600

    def test_adds_preprocessing_time(self):
        bs = _benchmark_setup(
            time_limit=3600,
            time_limit_for_model_agnostic_preprocessing=300,
            time_limit_with_model_agnostic_preprocessing=False,
        )
        assert bs.time_limit_per_config == 3900

    def test_adds_constant_overhead_when_flag_true(self):
        bs = _benchmark_setup(
            time_limit=3600,
            time_limit_for_model_agnostic_preprocessing=None,
            time_limit_with_model_agnostic_preprocessing=True,
        )
        assert bs.time_limit_per_config == 3600 + 60 * 15

    def test_both_overheads_combined(self):
        bs = _benchmark_setup(
            time_limit=3600,
            time_limit_for_model_agnostic_preprocessing=300,
            time_limit_with_model_agnostic_preprocessing=True,
        )
        assert bs.time_limit_per_config == 3600 + 300 + 60 * 15

    def test_zero_time_limit(self):
        bs = _benchmark_setup(
            time_limit=0,
            time_limit_for_model_agnostic_preprocessing=None,
            time_limit_with_model_agnostic_preprocessing=False,
        )
        assert bs.time_limit_per_config == 0


# ---------------------------------------------------------------------------
# BenchmarkSetup2026._parallel_safe_benchmark_name
# ---------------------------------------------------------------------------


class TestParallelSafeBenchmarkName:
    def test_no_parallel_name_returns_benchmark_name(self):
        bs = _benchmark_setup(benchmark_name="my_bench", parallel_benchmark_name=None)
        assert bs._parallel_safe_benchmark_name == "my_bench"

    def test_parallel_name_appended_with_underscore(self):
        bs = _benchmark_setup(benchmark_name="my_bench", parallel_benchmark_name="run1")
        assert bs._parallel_safe_benchmark_name == "my_bench_run1"

    def test_parallel_name_none_does_not_add_suffix(self):
        bs = _benchmark_setup(benchmark_name="bench", parallel_benchmark_name=None)
        assert (
            "_" not in bs._parallel_safe_benchmark_name
            or bs._parallel_safe_benchmark_name == "bench"
        )


# ---------------------------------------------------------------------------
# BenchmarkSetup2026._get_slurm_base_command (static, pure computation)
# ---------------------------------------------------------------------------

BASE_CMD_DEFAULTS = {
    "num_gpus": 0,
    "num_cpus": 8,
    "time_limit_per_config": 3600,
    "configs_per_job": 5,
    "time_limit_overhead": 1,
    "gpu_partition": "gpu_part",
    "cpu_partition": "cpu_part",
    "slurm_log_output": "/logs/bench",
    "slurm_script_path": "/scripts/submit.sh",
    "slurm_extra_gres": "localtmp:100",
    "slurm_exclusive_node": False,
    "memory_limit": 32,
    "slurm_mem_per_handle": False,
}


def _cmd(**overrides) -> str:
    kwargs = {**BASE_CMD_DEFAULTS, **overrides}
    return BenchmarkSetup2026._get_slurm_base_command(**kwargs)


class TestGetSlurmBaseCommand:
    def test_cpu_job_uses_cpu_partition(self):
        cmd = _cmd(num_gpus=0)
        assert "cpu_part" in cmd

    def test_gpu_job_uses_gpu_partition(self):
        cmd = _cmd(num_gpus=1)
        assert "gpu_part" in cmd

    def test_gpu_gres_included(self):
        cmd = _cmd(num_gpus=2)
        assert "gpu:2" in cmd

    def test_no_gpu_gres_not_in_cpu_job(self):
        cmd = _cmd(num_gpus=0, slurm_extra_gres=None)
        assert "--gres" not in cmd

    def test_extra_gres_included(self):
        cmd = _cmd(num_gpus=0, slurm_extra_gres="localtmp:50")
        assert "localtmp:50" in cmd

    def test_extra_gres_none_omits_gres_flag(self):
        cmd = _cmd(num_gpus=0, slurm_extra_gres=None)
        assert "--gres" not in cmd

    def test_time_limit_computed_in_hours(self):
        # 3600s/config * 5 configs + 1h overhead = 6h
        cmd = _cmd(
            time_limit_per_config=3600,
            configs_per_job=5,
            time_limit_overhead=1,
        )
        assert "--time=6:00:00" in cmd

    def test_time_limit_overhead_applied(self):
        cmd = _cmd(
            time_limit_per_config=7200,
            configs_per_job=1,
            time_limit_overhead=2,
        )
        assert "--time=4:00:00" in cmd

    def test_exclusive_node_uses_mem_0_nodes_1(self):
        cmd = _cmd(slurm_exclusive_node=True)
        assert "--mem=0" in cmd
        assert "--nodes=1" in cmd
        assert "--exclusive" in cmd

    def test_exclusive_node_omits_cpus_per_task(self):
        cmd = _cmd(slurm_exclusive_node=True)
        assert "--cpus-per-task" not in cmd

    def test_non_exclusive_includes_cpus_per_task(self):
        cmd = _cmd(slurm_exclusive_node=False, num_cpus=4)
        assert "--cpus-per-task=4" in cmd

    def test_non_exclusive_cpu_memory_per_job(self):
        cmd = _cmd(
            num_gpus=0,
            slurm_exclusive_node=False,
            slurm_mem_per_handle=False,
            memory_limit=32,
        )
        assert "--mem=32G" in cmd

    def test_mem_per_handle_cpu_uses_mem_per_cpu(self):
        cmd = _cmd(
            num_gpus=0,
            num_cpus=4,
            slurm_exclusive_node=False,
            slurm_mem_per_handle=True,
            memory_limit=32,
        )
        assert "--mem-per-cpu=8G" in cmd

    def test_mem_per_handle_gpu_uses_mem_per_gpu(self):
        cmd = _cmd(
            num_gpus=2,
            slurm_exclusive_node=False,
            slurm_mem_per_handle=True,
            memory_limit=40,
        )
        assert "--mem-per-gpu=20G" in cmd

    def test_slurm_log_output_in_command(self):
        cmd = _cmd(slurm_log_output="/my/logs/bench")
        assert "/my/logs/bench" in cmd

    def test_script_path_in_command(self):
        cmd = _cmd(slurm_script_path="/path/to/submit.sh")
        assert "/path/to/submit.sh" in cmd

    def test_returns_string(self):
        cmd = _cmd()
        assert isinstance(cmd, str)

    def test_gpu_job_gres_combined_with_extra(self):
        cmd = _cmd(num_gpus=1, slurm_extra_gres="localtmp:100")
        # Both GPU gres and extra gres should appear in the --gres flag.
        assert "gpu:1" in cmd
        assert "localtmp:100" in cmd


# ---------------------------------------------------------------------------
# BenchmarkSetup2026._load_task_metadata
# ---------------------------------------------------------------------------


class TestLoadTaskMetadata:
    def test_list_input_passthrough(self):
        meta = [_task_meta()]
        bs = _benchmark_setup(task_metadata=meta)
        result = bs._load_task_metadata()
        assert len(result) == 1
        assert result[0].dataset_name == "test_ds"

    def test_multi_split_task_unrolled(self):
        meta = [_task_meta(n_splits=3)]
        bs = _benchmark_setup(task_metadata=meta)
        result = bs._load_task_metadata()
        assert len(result) == 3

    def test_problem_type_filter_excludes_non_matching(self):
        meta = [
            _task_meta(problem_type="binary", dataset_name="bin_ds"),
            _task_meta(problem_type="regression", dataset_name="reg_ds"),
        ]
        bs = _benchmark_setup(task_metadata=meta, problem_types_to_run=["binary"])
        result = bs._load_task_metadata()
        assert all(m.problem_type == "binary" for m in result)
        assert not any(m.dataset_name == "reg_ds" for m in result)

    def test_problem_type_filter_includes_all_types_by_default(self):
        meta = [
            _task_meta(problem_type="binary"),
            _task_meta(problem_type="multiclass"),
            _task_meta(problem_type="regression"),
        ]
        bs = _benchmark_setup(
            task_metadata=meta,
            problem_types_to_run=["binary", "multiclass", "regression"],
        )
        result = bs._load_task_metadata()
        assert len(result) == 3

    def test_split_indices_none_keeps_all(self):
        meta = [_task_meta(n_splits=4)]
        bs = _benchmark_setup(task_metadata=meta, split_indices_to_run=None)
        result = bs._load_task_metadata()
        assert len(result) == 4

    def test_split_indices_lite_keeps_only_r0f0(self):
        meta = [_task_meta(n_splits=4)]
        bs = _benchmark_setup(task_metadata=meta, split_indices_to_run="lite")
        result = bs._load_task_metadata()
        assert len(result) == 1
        assert result[0].split_index == "r0f0"

    def test_split_indices_list_filters_correctly(self):
        # Create a task with 3 splits (r0f0, r0f1, r0f2).
        meta = [_task_meta(n_splits=3)]
        bs = _benchmark_setup(task_metadata=meta, split_indices_to_run=["r0f0", "r0f2"])
        result = bs._load_task_metadata()
        assert len(result) == 2
        assert {m.split_index for m in result} == {"r0f0", "r0f2"}

    def test_missing_task_id_str_raises(self):
        meta = [_task_meta(task_id_str=None)]
        bs = _benchmark_setup(task_metadata=meta)
        with pytest.raises(ValueError, match="task_id_str"):
            bs._load_task_metadata()

    def test_multiple_tasks_combined(self):
        meta = [
            _task_meta(dataset_name="a", n_splits=2),
            _task_meta(dataset_name="b", n_splits=3),
        ]
        bs = _benchmark_setup(task_metadata=meta)
        result = bs._load_task_metadata()
        assert len(result) == 5

    def test_empty_task_list(self):
        bs = _benchmark_setup(task_metadata=[])
        result = bs._load_task_metadata()
        assert result == []

    def test_each_result_has_exactly_one_split(self):
        meta = [_task_meta(n_splits=4)]
        bs = _benchmark_setup(task_metadata=meta)
        result = bs._load_task_metadata()
        for item in result:
            assert item.n_splits == 1

    def test_dataframe_input_parsed(self):

        meta_obj = _task_meta(n_splits=1)
        df = meta_obj.to_dataframe()
        bs = _benchmark_setup(task_metadata=df)
        result = bs._load_task_metadata()
        assert len(result) == 1
        assert result[0].dataset_name == meta_obj.dataset_name
