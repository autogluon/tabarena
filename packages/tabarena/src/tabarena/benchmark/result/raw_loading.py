"""IO helpers for raw per-run result artifacts (``results.pkl`` files).

A benchmark run writes one ``results.pkl`` per (method, task, split); these helpers locate,
load, and summarize those files. :func:`load_all_artifacts` / :func:`load_raw` deserialize the
full results (including predictions, so memory scales with what is loaded);
:func:`scan_raw_info` instead extracts only the small per-result info dict of each file and
discards the predictions, so whole runs can be inspected at near-zero memory cost.
"""

from __future__ import annotations

import io
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tabarena.benchmark.result.ag_bag_result import AGBagResult
from tabarena.benchmark.result.baseline_result import BaselineResult
from tabarena.benchmark.result.config_result import ConfigResult
from tabarena.utils.parallel_for import parallel_for
from tabarena.utils.pickle_utils import fetch_all_pickles, read_pickle_bytes

if TYPE_CHECKING:
    from collections.abc import Iterable


# TODO: This is a hack to ensure old result artifacts still load properly after renaming tabrepo to tabarena.
# TODO: We should ensure all artifacts are saved as a dictionary so this doesn't need to be here.
class _RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("tabrepo"):
            module = module.replace("tabrepo", "tabarena", 1)
        return super().find_class(module, name)


def _rename_load(file_obj):
    try:
        return _RenameUnpickler(file_obj).load()
    except EOFError as e:
        raise e from Exception(
            f"Failed to load artifact {file_obj}. This may be due to a corrupted file or"
            " an incompatible format. Please ensure the file is a valid artifact and"
            " try again.",
        )


def load_and_align(path, convert_to_holdout: bool = False) -> BaselineResult:
    # Transparently handles both raw and gzip-compressed ``.pkl`` artifacts.
    data: dict | BaselineResult = _rename_load(io.BytesIO(read_pickle_bytes(path)))

    data_aligned = BaselineResult.from_dict(data)
    if convert_to_holdout:
        return result_to_holdout(result=data_aligned)
    return data_aligned


def load_all_artifacts(
    file_paths: list[str | Path],
    engine: str = "sequential",
    convert_to_holdout: bool = False,
    progress_bar: bool = True,
) -> list[BaselineResult]:
    file_paths_lst = []
    for file_path in file_paths:
        file_paths_lst.append(
            {
                "path": str(file_path),
                "convert_to_holdout": convert_to_holdout,
            },
        )

    results_lst: list[BaselineResult] = parallel_for(
        f=load_and_align,
        inputs=file_paths_lst,
        engine=engine,
        progress_bar=progress_bar,
        desc="Loading raw artifacts",
    )
    return results_lst


def result_to_holdout(result: BaselineResult) -> BaselineResult:
    assert isinstance(result, AGBagResult)
    result_holdout = result.bag_artifacts(as_baseline=False)
    if len(result_holdout) > 0:
        assert len(result_holdout) == 1
        result_holdout = result_holdout[0]
    else:
        result_holdout = None
    return result_holdout


def results_to_holdout(result_lst: list[BaselineResult]) -> list[BaselineResult]:
    return [result_to_holdout(result) for result in result_lst]


def fetch_raw_result_paths(
    path_raw: str | Path | list[str | Path],
    name_pattern: str | None = None,
    num_workers: int | None = None,
) -> list[Path]:
    """Find all ``results.pkl`` files under ``path_raw`` (recursively).

    Raises with the most common file names found when no ``results.pkl`` exists, so a wrong
    directory is diagnosed instead of silently yielding zero results.
    """
    suffix = "results.pkl"
    file_paths = fetch_all_pickles(
        dir_path=path_raw,
        suffix=suffix,
        name_pattern=name_pattern,
        num_workers=num_workers,
    )
    if len(file_paths) == 0:
        roots = path_raw if isinstance(path_raw, list) else [path_raw]
        all_files = [p for root in roots for p in Path(root).rglob("*") if p.is_file()]
        counter = Counter(p.name for p in all_files).most_common(10)
        common_str = ", ".join(f"{name} ({count})" for name, count in counter) or "None"
        raise AssertionError(
            f"No valid {suffix!r} files found under path: {path_raw!r} "
            f"(name_pattern={name_pattern!r})!\nMost common file names:\n{common_str}",
        )
    return file_paths


def load_raw(
    path_raw: str | Path | list[str | Path] | None = None,
    name_pattern: str | None = None,
    engine: str = "ray",
    as_holdout: bool = False,
    num_workers: int | None = None,
) -> list[BaselineResult]:
    """Load all raw results from the ``results.pkl`` files under ``path_raw``.

    Holds every loaded result (including predictions) in memory at once; for a
    memory-light metadata-only pass over large runs use :func:`scan_raw_info` instead.
    """
    file_paths_method = fetch_raw_result_paths(
        path_raw=path_raw,
        name_pattern=name_pattern,
        num_workers=num_workers,
    )
    return load_all_artifacts(file_paths=file_paths_method, engine=engine, convert_to_holdout=as_holdout)


def get_info_from_result(result: BaselineResult) -> dict:
    """Extract the small per-result info dict that drives ``MethodMetadata`` inference."""
    cur_task_metadata = result.task_metadata
    cur_result = dict()
    cur_result["framework"] = result.framework
    cur_result["metric"] = result.result["metric"]
    cur_result["problem_type"] = result.problem_type
    cur_result.update(cur_task_metadata)
    is_bag = False

    ag_key = None
    model_type = None
    name_prefix = None
    num_gpus = 0
    method_type = "baseline"
    if isinstance(result, ConfigResult):
        hyperparameters = result.hyperparameters
        hyperparameters["model_cls"]
        model_type = hyperparameters["model_type"]
        ag_key = hyperparameters["ag_key"]
        name_prefix = hyperparameters["name_prefix"]
        num_gpus = result.result["method_metadata"].get("num_gpus", 0)
        method_type = "config"

        if isinstance(result, AGBagResult) and result.num_children > 1:
            is_bag = True

    cur_result["is_bag"] = is_bag
    cur_result["ag_key"] = ag_key
    cur_result["model_type"] = model_type
    cur_result["method_type"] = method_type
    cur_result["name_prefix"] = name_prefix
    cur_result["num_gpus"] = num_gpus

    return cur_result


def _load_result_info_batch(paths: list[str]) -> list[dict]:
    return [get_info_from_result(load_and_align(path=path)) for path in paths]


def scan_raw_info(
    path_raw: str | Path | list[str | Path] | None = None,
    file_paths: Iterable[str | Path] | None = None,
    name_pattern: str | None = None,
    engine: str = "ray",
    num_workers: int | None = None,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """Scan raw results and return one :func:`get_info_from_result` row per ``results.pkl``.

    Each file is loaded, reduced to its info dict, and discarded immediately, so only the
    small info rows ever accumulate — unlike :func:`load_raw`, which keeps every result's
    predictions in memory. With ``engine="ray"``, both the directory walk and the file scan
    run in parallel, with files batched per task so per-task overhead stays negligible on
    runs with tens of thousands of files. Pass either ``path_raw`` (searched recursively)
    or an explicit ``file_paths`` list.
    """
    if num_workers is None:
        num_workers = len(os.sched_getaffinity(0))
    if file_paths is None:
        file_paths = fetch_raw_result_paths(
            path_raw=path_raw,
            name_pattern=name_pattern,
            num_workers=num_workers if engine == "ray" else None,
        )
    file_paths = list(file_paths)
    batch_size = min(max(len(file_paths) // (num_workers * 4), 1), 100)
    batches = [file_paths[i : i + batch_size] for i in range(0, len(file_paths), batch_size)]
    info_batches = parallel_for(
        f=_load_result_info_batch,
        inputs=[{"paths": [str(p) for p in batch]} for batch in batches],
        engine=engine,
        progress_bar=progress_bar,
        desc="Scanning raw result info",
    )
    return pd.DataFrame([info for batch in info_batches for info in batch])
