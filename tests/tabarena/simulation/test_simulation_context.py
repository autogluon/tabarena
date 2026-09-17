from __future__ import annotations

from tabarena.simulation.context_artificial import load_repo_artificial
from tabarena.utils import task_to_tid_fold


def test_get_tasks_as_dataset_fold():
    """``get_tasks(as_dataset_fold=True)`` maps every task through the tid -> dataset dict."""
    context = load_repo_artificial()._zeroshot_context
    tasks = context.get_tasks()
    assert len(tasks) == 6
    tid_to_dataset = {tid: dataset for dataset, tid in context.dataset_to_tid_dict.items()}
    expected = [(tid_to_dataset[tid], fold) for tid, fold in (task_to_tid_fold(task=task) for task in tasks)]
    assert context.get_tasks(as_dataset_fold=True) == expected
    assert context.get_tasks(datasets=["ada"], as_dataset_fold=True) == [("ada", 0), ("ada", 1), ("ada", 2)]
    assert context._task_to_dataset_fold(tasks[0]) == expected[0]
