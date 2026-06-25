from __future__ import annotations

import numpy as np

from tabarena.repository.time_utils import filter_configs_by_runtime, get_runtime, sort_by_runtime
from tabarena.simulation.context_artificial import load_repo_artificial

repo = load_repo_artificial()


def test_get_runtime():
    config_names = repo.configs()
    runtime_dict = get_runtime(
        repo,
        dataset="ada",
        fold=1,
        config_names=config_names,
    )
    assert list(runtime_dict.keys()) == config_names
    assert np.allclose(list(runtime_dict.values()), [1.0, 2.0])


def test_get_runtime_time_infer_s():
    config_names = repo.configs()
    runtime_dict = get_runtime(
        repo,
        dataset="ada",
        fold=1,
        config_names=config_names,
        runtime_col="time_infer_s",
    )
    assert list(runtime_dict.keys()) == config_names
    assert np.allclose(list(runtime_dict.values()), [2.0, 4.0])


def test_sort_by_runtime():
    config_names = repo.configs()
    assert sort_by_runtime(repo, config_names) == ["NeuralNetFastAI_r1", "NeuralNetFastAI_r2"]


def test_filter_configs_by_runtime():
    config_names = repo.configs()
    for max_cumruntime, num_config_expected in [
        (None, len(config_names)),
        (0, len(config_names)),
        (0.5, 0),
        (2.0, 1),
        (3.01, len(config_names)),
        (6.0, len(config_names)),
        (np.inf, len(config_names)),
    ]:
        selected_configs = filter_configs_by_runtime(
            repo,
            dataset="ada",
            fold=1,
            config_names=config_names,
            max_cumruntime=max_cumruntime,
        )
        assert selected_configs == config_names[:num_config_expected]
