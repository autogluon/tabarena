from __future__ import annotations

import pandas as pd
import pytest

from tabarena.benchmark.experiment.experiment_runner_api import (
    _build_cache_prefix,
    _clean_repetitions_mode_args_for_matrix,
    _parse_repetitions_mode_and_args,
    run_experiments_new,
)
from tabarena.utils.cache import AbstractCacheFunction

RES_2_2 = [(f, r) for f in range(2) for r in range(2)]


class _RecordingCache(AbstractCacheFunction):
    """Minimal cache class that records its constructions and never has a cache hit."""

    instances: list = []

    def __init__(self, cache_name, cache_path, **kwargs):
        super().__init__(include_self_in_call=kwargs.get("include_self_in_call", False))
        self.cache_name = cache_name
        self.cache_path = cache_path
        type(self).instances.append((cache_name, cache_path, kwargs))

    @property
    def cache_file(self):
        return None

    @property
    def exists(self) -> bool:
        return False

    def save_cache(self, data) -> None:
        pass

    def load_cache(self):
        return None


@pytest.mark.parametrize(
    ("repetitions_mode", "repetitions_mode_args", "tasks", "expected_output"),
    [
        # Presets
        ("TabArena-Lite", None, [0, 1], [[(0, 0)], [(0, 0)]]),
        ("TabArena-Lite", (4, 3), [0, 1], [[(0, 0)], [(0, 0)]]),
        # Matrix
        ("matrix", None, [0, 1], AssertionError),  # input None
        ("matrix", [[None], [None]], [0], AssertionError),  # not same size as tasks
        ("matrix", [(), [None]], [0], AssertionError),  # not tuple (in a later element)
        ("matrix", [(1,)], [0], AssertionError),  # not len 2
        ("matrix", [(1, "str")], [0], AssertionError),  # not both str
        ("matrix", [([], "str")], [0], AssertionError),  # if list-based, not both str
        ("matrix", [([], [0])], [0], AssertionError),  # empty list
        ("matrix", [([0], ["str"])], [0], AssertionError),  # not int list
        ("matrix", "str", [0], AssertionError),  # not a tuple if not a list
        ("matrix", ([0], []), [0], AssertionError),  # empty list without list of tuples
        ("matrix", (2, 2), [0, 1], [RES_2_2, RES_2_2]),
        ("matrix", ([0, 1], [0, 1]), [0, 1], [RES_2_2, RES_2_2]),
        ("matrix", ([2], [0, 1]), [0, 1], [[(2, 0), (2, 1)], [(2, 0), (2, 1)]]),
        # Individual
        ("individual", None, [0, 1], AssertionError),  # input None
        ("individual", "str", [0], AssertionError),  # not a list
        ("individual", [[None], [None]], [0], AssertionError),  # not same size as tasks
        ("individual", [(0, 0), (0, "str")], [0], AssertionError),  # not tuples of int
        ("individual", [(0, 0), (0, 1, "str")], [0], AssertionError),  # wrong length
        ("individual", [(0, 0), None], [0], AssertionError),  # not tuples
        ("individual", [[(0, 1)], None], [0], AssertionError),  # not list of tuples
        ("individual", [[(0, 1)], [None]], [0], AssertionError),  # not list of tuples
        (
            "individual",
            [[(0, 1)], [(None,)]],
            [0],
            AssertionError,
        ),  # not list of tuples
        (
            "individual",
            [[(0, 1)], [(None, "str")]],
            [0],
            AssertionError,
        ),  # not list of tuples
        ("individual", [(0, 0)], [0, 1], [[(0, 0)], [(0, 0)]]),
        ("individual", [(0, 0), (2, 3)], [0, 1], [[(0, 0), (2, 3)], [(0, 0), (2, 3)]]),
        (
            "individual",
            [[(0, 0), (0, 1)], [(1, 0), (1, 1)]],
            [0, 1],
            [[(0, 0), (0, 1)], [(1, 0), (1, 1)]],
        ),
    ],
)
def test_parse_repetitions_mode_and_args(
    repetitions_mode,
    repetitions_mode_args,
    tasks,
    expected_output,
):
    if isinstance(expected_output, type) and issubclass(expected_output, BaseException):
        with pytest.raises(expected_output):
            _parse_repetitions_mode_and_args(
                repetitions_mode=repetitions_mode,
                repetitions_mode_args=repetitions_mode_args,
                tasks=tasks,
            )
    else:
        assert expected_output == _parse_repetitions_mode_and_args(
            repetitions_mode=repetitions_mode,
            repetitions_mode_args=repetitions_mode_args,
            tasks=tasks,
        )


# ---------------------------------------------------------------------------
# _clean_repetitions_mode_args_for_matrix
# ---------------------------------------------------------------------------


class TestCleanRepetitionsModeArgsForMatrix:
    # --- Valid inputs ---

    def test_two_ints_expand_to_ranges(self):
        folds, repeats = _clean_repetitions_mode_args_for_matrix((3, 2))
        assert folds == [0, 1, 2]
        assert repeats == [0, 1]

    def test_single_fold_single_repeat(self):
        folds, repeats = _clean_repetitions_mode_args_for_matrix((1, 1))
        assert folds == [0]
        assert repeats == [0]

    def test_lists_passed_through(self):
        folds, repeats = _clean_repetitions_mode_args_for_matrix(([0, 2, 4], [1, 3]))
        assert folds == [0, 2, 4]
        assert repeats == [1, 3]

    def test_single_element_lists(self):
        folds, repeats = _clean_repetitions_mode_args_for_matrix(([5], [0]))
        assert folds == [5]
        assert repeats == [0]

    def test_large_ints_expand(self):
        folds, repeats = _clean_repetitions_mode_args_for_matrix((8, 3))
        assert folds == list(range(8))
        assert repeats == list(range(3))

    def test_returns_tuple_of_two_lists(self):
        result = _clean_repetitions_mode_args_for_matrix((2, 3))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    # --- Invalid inputs ---

    def test_not_a_tuple_raises(self):
        with pytest.raises(AssertionError):
            _clean_repetitions_mode_args_for_matrix([2, 3])

    def test_tuple_too_short_raises(self):
        with pytest.raises(AssertionError):
            _clean_repetitions_mode_args_for_matrix((2,))

    def test_tuple_too_long_raises(self):
        with pytest.raises(AssertionError):
            _clean_repetitions_mode_args_for_matrix((2, 3, 4))

    def test_mixed_int_and_list_raises(self):
        with pytest.raises(AssertionError):
            _clean_repetitions_mode_args_for_matrix((2, [0, 1]))

    def test_mixed_list_and_int_raises(self):
        with pytest.raises(AssertionError):
            _clean_repetitions_mode_args_for_matrix(([0, 1], 2))

    def test_empty_first_list_raises(self):
        with pytest.raises(AssertionError):
            _clean_repetitions_mode_args_for_matrix(([], [0]))

    def test_empty_second_list_raises(self):
        with pytest.raises(AssertionError):
            _clean_repetitions_mode_args_for_matrix(([0], []))

    def test_non_int_in_first_list_raises(self):
        with pytest.raises(AssertionError):
            _clean_repetitions_mode_args_for_matrix((["a"], [0]))

    def test_non_int_in_second_list_raises(self):
        with pytest.raises(AssertionError):
            _clean_repetitions_mode_args_for_matrix(([0], ["a"]))

    def test_neither_ints_nor_lists_raises(self):
        with pytest.raises(AssertionError):
            _clean_repetitions_mode_args_for_matrix(("a", "b"))

    def test_zero_first_int_returns_empty_first_list(self):
        # (0, N) means 0 folds → empty range → empty list
        folds, repeats = _clean_repetitions_mode_args_for_matrix((0, 2))
        assert folds == []
        assert repeats == [0, 1]


# ---------------------------------------------------------------------------
# run_experiments_new — input validation (no ML training)
# ---------------------------------------------------------------------------


def _make_minimal_experiment(name: str = "lgbm_test"):
    from autogluon.tabular.models import LGBModel

    from tabarena.benchmark.experiment import AGModelBagExperiment

    return AGModelBagExperiment(
        name=name,
        model_cls=LGBModel,
        model_hyperparameters={},
        num_bag_folds=2,
        time_limit=60,
    )


class TestRunExperimentsNewValidation:
    def test_non_experiment_in_model_experiments_raises(self, tmp_path):
        with pytest.raises(AssertionError):
            run_experiments_new(
                output_dir=str(tmp_path),
                model_experiments=["not_an_experiment"],
                tasks=[360],
                repetitions_mode="individual",
                repetitions_mode_args=[(0, 0)],
            )

    def test_duplicate_experiment_names_raises(self, tmp_path):
        exp1 = _make_minimal_experiment("dup_name")
        exp2 = _make_minimal_experiment("dup_name")
        with pytest.raises(AssertionError):
            run_experiments_new(
                output_dir=str(tmp_path),
                model_experiments=[exp1, exp2],
                tasks=[360],
                repetitions_mode="individual",
                repetitions_mode_args=[(0, 0)],
            )

    def test_non_int_non_usertask_in_tasks_raises(self, tmp_path):
        with pytest.raises(AssertionError):
            run_experiments_new(
                output_dir=str(tmp_path),
                model_experiments=[_make_minimal_experiment()],
                tasks=["not_valid"],
                repetitions_mode="individual",
                repetitions_mode_args=[(0, 0)],
            )

    def test_unknown_repetitions_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown"):
            run_experiments_new(
                output_dir=str(tmp_path),
                model_experiments=[_make_minimal_experiment()],
                tasks=[360],
                repetitions_mode="unknown_mode",
                repetitions_mode_args=None,
            )


# ---------------------------------------------------------------------------
# run_experiments_new — cache_mode="only" with no cache (no ML runs)
# ---------------------------------------------------------------------------


class TestRunExperimentsNewCacheOnly:
    def test_empty_cache_returns_empty_list(self, tmp_path):
        result = run_experiments_new(
            output_dir=str(tmp_path),
            model_experiments=[_make_minimal_experiment()],
            tasks=[360],
            repetitions_mode="individual",
            repetitions_mode_args=[(0, 0)],
            cache_mode="only",
        )
        assert result == []

    def test_multiple_tasks_no_cache_returns_empty(self, tmp_path):
        result = run_experiments_new(
            output_dir=str(tmp_path),
            model_experiments=[_make_minimal_experiment()],
            tasks=[360, 361],
            repetitions_mode="individual",
            repetitions_mode_args=[(0, 0)],
            cache_mode="only",
        )
        assert result == []

    def test_multiple_experiments_no_cache_returns_empty(self, tmp_path):
        exp1 = _make_minimal_experiment("exp1")
        exp2 = _make_minimal_experiment("exp2")
        result = run_experiments_new(
            output_dir=str(tmp_path),
            model_experiments=[exp1, exp2],
            tasks=[360],
            repetitions_mode="individual",
            repetitions_mode_args=[(0, 0)],
            cache_mode="only",
        )
        assert result == []

    def test_tabarena_lite_no_cache_returns_empty(self, tmp_path):
        result = run_experiments_new(
            output_dir=str(tmp_path),
            model_experiments=[_make_minimal_experiment()],
            tasks=[360],
            repetitions_mode="TabArena-Lite",
            cache_mode="only",
        )
        assert result == []

    def test_matrix_mode_no_cache_returns_empty(self, tmp_path):
        result = run_experiments_new(
            output_dir=str(tmp_path),
            model_experiments=[_make_minimal_experiment()],
            tasks=[360],
            repetitions_mode="matrix",
            repetitions_mode_args=(2, 1),
            cache_mode="only",
        )
        assert result == []

    def test_returns_list_type(self, tmp_path):
        result = run_experiments_new(
            output_dir=str(tmp_path),
            model_experiments=[_make_minimal_experiment()],
            tasks=[360],
            repetitions_mode="individual",
            repetitions_mode_args=[(0, 0)],
            cache_mode="only",
        )
        assert isinstance(result, list)

    def test_user_task_no_cache_returns_empty(self, tmp_path):
        from tabarena.benchmark.task.user_task import UserTask

        task = UserTask(task_name="test_empty_task", task_cache_path=tmp_path / "oml")
        result = run_experiments_new(
            output_dir=str(tmp_path / "out"),
            model_experiments=[_make_minimal_experiment()],
            tasks=[task],
            repetitions_mode="individual",
            repetitions_mode_args=[(0, 0)],
            cache_mode="only",
        )
        assert result == []

    def test_local_base_cache_path_uses_output_dir(self, tmp_path):
        # base_cache_path == output_dir.
        # Result is empty because no cache exists; but it shouldn't raise.
        result = run_experiments_new(
            output_dir=str(tmp_path / "subdir"),
            model_experiments=[_make_minimal_experiment()],
            tasks=[360],
            repetitions_mode="individual",
            repetitions_mode_args=[(0, 0)],
            cache_mode="only",
        )
        assert result == []


class TestBuildCachePrefix:
    def test_name_first_with_repeat(self):
        prefix = _build_cache_prefix(
            method_name="m",
            cache_task_key=360,
            fold=2,
            repeat=1,
        )
        assert prefix == "data/m/360/1_2"


class TestExperimentBatchRunnerDelegation:
    """ExperimentBatchRunner.run() now delegates to run_experiments_new."""

    @staticmethod
    def _runner(tmp_path, **kwargs):
        from tabarena.benchmark.experiment import ExperimentBatchRunner

        task_metadata = pd.DataFrame({"tid": [0, 1], "dataset": ["d0", "d1"]})
        return ExperimentBatchRunner(
            expname=str(tmp_path),
            task_metadata=task_metadata,
            **kwargs,
        )

    def test_only_cache_no_cache_returns_empty(self, tmp_path):
        # only_cache => cache_mode="only"; with no cache present, returns [] without
        # loading any task or training (exercises the full delegation wiring).
        runner = self._runner(tmp_path, only_cache=True)
        result = runner.run(
            methods=[_make_minimal_experiment()],
            datasets=["d0", "d1"],
            folds=[0],
        )
        assert result == []

    def test_only_cache_with_repeats(self, tmp_path):
        runner = self._runner(tmp_path, only_cache=True)
        result = runner.run(
            methods=[_make_minimal_experiment()],
            datasets=["d0"],
            folds=[0, 1],
            repeats=[0, 1],
        )
        assert result == []

    def test_custom_cache_cls_is_forwarded(self, tmp_path):
        # A non-Pickle cache class is threaded through to run_experiments_new and used.
        _RecordingCache.instances.clear()
        runner = self._runner(tmp_path, only_cache=True, cache_cls=_RecordingCache)
        result = runner.run(
            methods=[_make_minimal_experiment()],
            datasets=["d0"],
            folds=[0],
        )
        assert result == []
        assert _RecordingCache.instances, "custom cache_cls was not used"
        # ExperimentBatchRunner's default cache_cls_kwargs forwards include_self_in_call=True.
        assert all(kw.get("include_self_in_call") is True for *_, kw in _RecordingCache.instances)

    def test_unknown_dataset_raises(self, tmp_path):
        runner = self._runner(tmp_path, only_cache=True)
        with pytest.raises(ValueError, match="present in task_metadata"):
            runner.run(
                methods=[_make_minimal_experiment()],
                datasets=["does_not_exist"],
                folds=[0],
            )


class TestRunExperimentsNewCacheCls:
    def test_custom_cache_cls_used(self, tmp_path):
        _RecordingCache.instances.clear()
        result = run_experiments_new(
            output_dir=str(tmp_path),
            model_experiments=[_make_minimal_experiment()],
            tasks=[360],
            repetitions_mode="individual",
            repetitions_mode_args=[(0, 0)],
            cache_mode="only",
            cache_cls=_RecordingCache,
        )
        assert result == []
        assert _RecordingCache.instances, "custom cache_cls was not used"

    def test_cache_cls_kwargs_forwarded(self, tmp_path):
        _RecordingCache.instances.clear()
        run_experiments_new(
            output_dir=str(tmp_path),
            model_experiments=[_make_minimal_experiment()],
            tasks=[360],
            repetitions_mode="individual",
            repetitions_mode_args=[(0, 0)],
            cache_mode="only",
            cache_cls=_RecordingCache,
            cache_cls_kwargs={"include_self_in_call": True},
        )
        # Explicit cache_cls_kwargs wins over the include_self_in_call default (False).
        assert all(kw.get("include_self_in_call") is True for *_, kw in _RecordingCache.instances)
