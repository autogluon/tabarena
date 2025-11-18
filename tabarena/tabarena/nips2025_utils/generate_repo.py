from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabarena.repository import EvaluationRepository
from tabarena.utils.pickle_utils import fetch_all_pickles
from tabarena.benchmark.result import BaselineResult, ConfigResult, ExperimentResults

from .load_artifacts import load_all_artifacts


def generate_repo(experiment_path: str, task_metadata: pd.DataFrame, name_suffix: str | None = None) -> EvaluationRepository:
    file_paths = fetch_all_pickles(dir_path=experiment_path)
    file_paths = sorted([str(f) for f in file_paths])
    print(len(file_paths))

    return generate_repo_from_paths(result_paths=file_paths, task_metadata=task_metadata, name_suffix=name_suffix)


def generate_repo_from_paths(
    result_paths: list[str | Path],
    task_metadata: pd.DataFrame,
    engine: str = "ray",
    name_suffix: str | None = None,
    as_holdout: bool = False,
) -> EvaluationRepository:
    results_lst = load_all_artifacts(file_paths=result_paths, engine=engine, convert_to_holdout=as_holdout)
    return generate_repo_from_results_lst(
        results_lst=results_lst,
        task_metadata=task_metadata,
        name_suffix=name_suffix,
    )


def generate_repo_from_results_lst(
    results_lst: list[BaselineResult],
    task_metadata: pd.DataFrame,
    name_suffix: str | None = None,
) -> EvaluationRepository:
    results_lst = [r for r in results_lst if r is not None]
    tids = set(list(task_metadata["tid"].unique()))
    results_lst = [r for r in results_lst if r.result["task_metadata"]["tid"] in tids]

    if name_suffix is not None:
        for r in results_lst:
            r.update_name(name_suffix=name_suffix)
            if isinstance(r, ConfigResult):
                r.update_model_type(name_suffix=name_suffix)

    if len(results_lst) == 0:
        print(f"EMPTY")
        return None

    exp_results = ExperimentResults(task_metadata=task_metadata)

    repo: EvaluationRepository = exp_results.repo_from_results(results_lst=results_lst)
    return repo
