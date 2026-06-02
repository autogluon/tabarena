from typing import Tuple

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def catchtime(name: str, logger = None) -> float:
    start = perf_counter()
    print_fun = print if logger is None else logger.info
    try:
        print_fun(f"start: {name}")
        yield lambda: perf_counter() - start
    finally:
        print_fun(f"Time for {name}: {perf_counter() - start:.4f} secs")


def task_to_tid(task: str) -> int:
    return int(task.rsplit("_", 1)[0])


def task_to_fold(task: str) -> int:
    return int(task.rsplit("_", 1)[1])


def task_to_tid_fold(task: str) -> Tuple[int, int]:
    dataset_fold = task.rsplit("_", 1)
    tid = int(dataset_fold[0])
    fold = int(dataset_fold[1])
    return tid, fold


def tid_fold_to_task(tid: int, fold: int) -> str:
    return f"{tid}_{fold}"
