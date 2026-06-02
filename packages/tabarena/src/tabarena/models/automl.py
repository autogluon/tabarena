from __future__ import annotations

from tabarena.benchmark.experiment.experiment_constructor import AGExperiment


def generate_autogluon_experiments():
    return [
        AGExperiment(
            name="AutoGluon_v140_bq_4h8c",
            fit_kwargs=dict(
                time_limit=14400,
                presets="best",
            ),
        ),
        AGExperiment(
            name="AutoGluon_v140_bq_1h8c",
            fit_kwargs=dict(
                time_limit=3600,
                presets="best",
            ),
        ),
        AGExperiment(
            name="AutoGluon_v140_bq_5m8c",
            fit_kwargs=dict(
                time_limit=300,
                presets="best",
            ),
        ),
        AGExperiment(
            name="AutoGluon_v140_hq_4h8c",
            fit_kwargs=dict(
                time_limit=14400,
                presets="high",
            ),
        ),
        AGExperiment(
            name="AutoGluon_v140_hq_5m8c",
            fit_kwargs=dict(
                time_limit=300,
                presets="high",
            ),
        ),
        AGExperiment(
            name="AutoGluon_v140_hqil_4h8c",
            fit_kwargs=dict(
                time_limit=14400,
                presets="high",
                infer_limit=0.0001,
            ),
        ),
        AGExperiment(
            name="AutoGluon_v140_hqil_5m8c",
            fit_kwargs=dict(
                time_limit=300,
                presets="high",
                infer_limit=0.0001,
            ),
        ),
    ]


def generate_autogluon_extreme_experiments():
    return [
        AGExperiment(
            name="AutoGluon_v140_eq_4h8c",
            fit_kwargs=dict(
                time_limit=14400,
                presets="extreme",
            ),
        ),
        AGExperiment(
            name="AutoGluon_v140_eq_1h8c",
            fit_kwargs=dict(
                time_limit=3600,
                presets="extreme",
            ),
        ),
        AGExperiment(
            name="AutoGluon_v140_eq_5m8c",
            fit_kwargs=dict(
                time_limit=300,
                presets="extreme",
            ),
        ),
    ]
