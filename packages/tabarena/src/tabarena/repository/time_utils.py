from __future__ import annotations

from typing import List, Optional, Dict
import numpy as np
import pandas as pd

from .abstract_repository import AbstractRepository


# FIXME: Should move into repo?
def get_runtime(
        repo: AbstractRepository,
        dataset: str,
        fold: int,
        config_metrics: pd.DataFrame | None = None,
        config_names: Optional[List[str]] = None,
        runtime_col: str = 'time_train_s',
        fail_if_missing: bool = True
) -> Dict[str, float]:
    """
    :param repo:
    :param dataset:
    :param fold:
    :param config_names:
    :param fail_if_missing: whether to raise an error if some configurations are missing
    :return: a dictionary with keys are elements in `config_names` and the values are runtimes of the configuration
    on the task `tid`_`fold`.
    """
    task = (dataset, fold)
    if config_metrics is None:
        config_metrics = repo.metrics(
            tasks=[task],
            configs=config_names,
            set_index=False,
        )

    if not config_names:
        config_names = repo.configs()

    config_metrics = config_metrics.set_index("framework", drop=True)
    runtime_series = config_metrics[runtime_col]
    runtime_configs = runtime_series.to_dict()

    missing_configurations = set(config_names).difference(runtime_configs.keys())
    if len(missing_configurations) > 0:
        if fail_if_missing:
            raise ValueError(
                f"not all configurations could be found in available data for the task {task}\n" \
                f"requested: {config_names}\n" \
                f"available: {list(runtime_configs.keys())}."
            )
        else:
            # todo take mean of framework
            if repo._config_fallback is not None:
                config_metrics_fallback = repo.metrics(tasks=[task], configs=[repo._config_fallback])
                fill_value = config_metrics_fallback.loc[(dataset, fold, repo._config_fallback), runtime_col]
            else:
                fill_value = np.mean(list(runtime_configs.values()))
            # print(f"Imputing missing value {fill_value} for configurations {missing_configurations} on task {task}")
            for configuration in missing_configurations:
                runtime_configs[configuration] = fill_value
    return runtime_configs


def sort_by_runtime(
    repo: AbstractRepository,
    config_names: List[str],
    ascending: bool = True,
) -> List[str]:
    df_metrics = repo._zeroshot_context.df_configs
    config_sorted = df_metrics.pivot_table(
        index="framework", columns="tid", values="time_train_s"
    ).median(axis=1).sort_values(ascending=ascending).index.tolist()
    return [c for c in config_sorted if c in set(config_names)]


def filter_configs_by_runtime(
        repo: AbstractRepository,
        dataset: str,
        fold: int,
        config_names: List[str],
        config_metrics: pd.DataFrame = None,
        max_cumruntime: Optional[float] = None
) -> List[str]:
    """
    :param repo:
    :param tid:
    :param fold:
    :param config_names:
    :param max_cumruntime:
    :return: A sublist of configuration from `config_names` such that the total cumulative runtime does not exceed
    `max_cumruntime`.
    """
    if not max_cumruntime:
        return config_names
    else:
        assert dataset in repo.datasets()
        assert fold in repo.dataset_to_folds(dataset=dataset)
        runtime_configs = get_runtime(repo=repo, dataset=dataset, fold=fold, config_names=config_names, config_metrics=config_metrics, fail_if_missing=False)
        cumruntime = np.cumsum([runtime_configs[config] for config in config_names])
        # str_runtimes = ", ".join([f"{name}: {time}" for name, time in zip(runtime_configs.keys(), cumruntime)])
        # print(f"Cumulative runtime:\n {str_runtimes}")

        # gets index where cumulative runtime is below the target and next index is above the target
        i = np.searchsorted(cumruntime, max_cumruntime)
        return config_names[:i]

