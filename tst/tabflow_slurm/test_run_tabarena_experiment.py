from __future__ import annotations

import argparse

import pytest
from tabflow_slurm.run_tabarena_experiment import (
    _parse_int_list,
    _parse_int_list_or_none,
    _parse_int_or_none,
    _parse_task_id,
    _str2bool,
    setup_slurm_job,
)

# ---------------------------------------------------------------------------
# _str2bool
# ---------------------------------------------------------------------------


class TestStr2Bool:
    @pytest.mark.parametrize("value", ["yes", "true", "t", "1", "YES", "True", "T"])
    def test_truthy_strings(self, value):
        assert _str2bool(value) is True

    @pytest.mark.parametrize("value", ["no", "false", "f", "0", "NO", "False", "F"])
    def test_falsy_strings(self, value):
        assert _str2bool(value) is False

    def test_bool_true_passthrough(self):
        assert _str2bool(True) is True

    def test_bool_false_passthrough(self):
        assert _str2bool(False) is False

    @pytest.mark.parametrize("value", ["maybe", "yes_no", "2", "tru", "fals", ""])
    def test_invalid_raises(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            _str2bool(value)

    def test_invalid_message_mentions_boolean(self):
        with pytest.raises(argparse.ArgumentTypeError, match="[Bb]oolean"):
            _str2bool("not_a_bool")


# ---------------------------------------------------------------------------
# _parse_int_list
# ---------------------------------------------------------------------------


class TestParseIntList:
    def test_single_element(self):
        assert _parse_int_list("5") == [5]

    def test_multiple_elements(self):
        assert _parse_int_list("1,2,3") == [1, 2, 3]

    def test_negative_value(self):
        assert _parse_int_list("-1,0,1") == [-1, 0, 1]

    def test_returns_list_of_ints(self):
        result = _parse_int_list("10,20")
        assert all(isinstance(v, int) for v in result)

    def test_single_zero(self):
        assert _parse_int_list("0") == [0]

    def test_longer_list(self):
        assert _parse_int_list("0,1,2,3,4,5,6,7") == list(range(8))

    def test_non_int_raises(self):
        with pytest.raises((ValueError, TypeError)):
            _parse_int_list("a,b,c")

    def test_float_raises(self):
        with pytest.raises((ValueError, TypeError)):
            _parse_int_list("1.5,2.0")


# ---------------------------------------------------------------------------
# _parse_int_list_or_none
# ---------------------------------------------------------------------------


class TestParseIntListOrNone:
    @pytest.mark.parametrize("value", ["none", "None", "NONE", "null", "Null", "NULL"])
    def test_none_variants_return_none(self, value):
        assert _parse_int_list_or_none(value) is None

    def test_python_none_returns_none(self):
        assert _parse_int_list_or_none(None) is None

    def test_single_int(self):
        assert _parse_int_list_or_none("3") == [3]

    def test_multiple_ints(self):
        assert _parse_int_list_or_none("1,2,3") == [1, 2, 3]

    def test_result_type_is_list(self):
        result = _parse_int_list_or_none("10,20")
        assert isinstance(result, list)

    def test_empty_string_raises(self):
        with pytest.raises((ValueError, TypeError)):
            _parse_int_list_or_none("")


# ---------------------------------------------------------------------------
# _parse_int_or_none
# ---------------------------------------------------------------------------


class TestParseIntOrNone:
    @pytest.mark.parametrize("value", ["none", "None", "NONE", "null", "Null", "NULL"])
    def test_none_variants_return_none(self, value):
        assert _parse_int_or_none(value) is None

    def test_python_none_returns_none(self):
        assert _parse_int_or_none(None) is None

    def test_positive_int(self):
        assert _parse_int_or_none("7") == 7

    def test_zero(self):
        assert _parse_int_or_none("0") == 0

    def test_negative_int(self):
        assert _parse_int_or_none("-5") == -5

    def test_returns_int_type(self):
        assert isinstance(_parse_int_or_none("42"), int)

    def test_float_string_raises(self):
        with pytest.raises((ValueError, TypeError)):
            _parse_int_or_none("3.14")

    def test_non_numeric_raises(self):
        with pytest.raises((ValueError, TypeError)):
            _parse_int_or_none("abc")


# ---------------------------------------------------------------------------
# _parse_task_id
# ---------------------------------------------------------------------------


class TestParseTaskId:
    def test_numeric_string_returns_int(self):
        result = _parse_task_id("360")
        assert result == 360
        assert isinstance(result, int)

    def test_zero_string_returns_int(self):
        result = _parse_task_id("0")
        assert result == 0
        assert isinstance(result, int)

    def test_large_numeric_string(self):
        result = _parse_task_id("12345678")
        assert result == 12345678

    def test_negative_numeric_string(self):
        result = _parse_task_id("-1")
        assert result == -1

    def test_user_task_string_parsed(self, tmp_path):
        # A valid UserTask task_id_str is formatted as "UserTask|<id>|<name>|<path>".
        # Passing a non-integer string that matches the UserTask format should return a
        # UserTask object rather than raising an exception.
        from tabarena.benchmark.task.user_task import UserTask

        task = UserTask(task_name="my_task", task_cache_path=tmp_path)
        result = _parse_task_id(task.task_id_str)
        assert isinstance(result, UserTask)
        assert result.task_name == "my_task"

    def test_arbitrary_non_int_non_usertask_raises(self):
        # Plain non-parseable strings that aren't UserTask ids should raise.
        with pytest.raises(Exception):
            _parse_task_id("not_a_valid_task_id")


# ---------------------------------------------------------------------------
# setup_slurm_job  (no Ray, so returns None immediately)
# ---------------------------------------------------------------------------


class TestSetupSlurmJob:
    def test_auto_openml_cache_returns_none(self):
        result = setup_slurm_job(
            openml_cache_dir="auto",
            num_cpus=1,
            num_gpus=0,
            memory_limit=4,
            setup_ray_for_slurm_shared_resources_environment=False,
        )
        assert result is None

    def test_custom_openml_cache_sets_directory(self, tmp_path):
        import openml

        cache_dir = str(tmp_path / "oml_cache")
        result = setup_slurm_job(
            openml_cache_dir=cache_dir,
            num_cpus=1,
            num_gpus=0,
            memory_limit=4,
            setup_ray_for_slurm_shared_resources_environment=False,
        )
        assert result is None
        # Verify the OpenML cache was pointed at the custom directory.
        assert str(openml.config.get_cache_directory()).startswith(cache_dir)

    def test_no_ray_setup_skips_ray(self, capsys):
        result = setup_slurm_job(
            openml_cache_dir="auto",
            num_cpus=2,
            num_gpus=0,
            memory_limit=8,
            setup_ray_for_slurm_shared_resources_environment=False,
        )
        assert result is None
        captured = capsys.readouterr()
        # Should NOT mention Ray setup when skipping it.
        assert "Ray" not in captured.out
