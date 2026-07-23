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


def test_get_runtime_missing_fallback_imputes_mean(caplog):
    """A missing config whose fallback has no result on the task (e.g. the repo was
    subset to a config list that dropped the fallback's data) imputes the mean of the
    available runtimes with a warning, instead of raising KeyError."""
    import logging

    repo_fb = load_repo_artificial()
    repo_fb.set_config_fallback(repo_fb.configs()[0])
    # subset to one non-fallback config: the fallback's rows are gone but the
    # attribute still names it
    sub = repo_fb.subset(configs=[repo_fb.configs()[1]])
    assert sub._config_fallback == repo_fb.configs()[0]

    with caplog.at_level(logging.WARNING):
        runtime_dict = get_runtime(
            sub,
            dataset="ada",
            fold=1,
            config_names=[*sub.configs(), "MissingConfig_r1"],
            fail_if_missing=False,
        )
    assert "MissingConfig_r1" in runtime_dict
    assert runtime_dict["MissingConfig_r1"] == 2.0  # mean of the one available runtime
    assert any("has no" in r.message and "imputing" in r.message for r in caplog.records)
