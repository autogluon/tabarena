from __future__ import annotations

import copy
from collections import defaultdict

from tabarena.benchmark.experiment.experiment_constructor import AGModelExperiment


class AutoGluonExporter:
    def __init__(self, experiments: list[AGModelExperiment]):
        self.experiments = experiments

    def export_hyperparameters(self) -> dict[str, list[dict]]:
        """
        Convert TabArena AGModelExperiment objects into an AutoGluon-compatible
        hyperparameters dictionary.

        Returns
        -------
        dict[str, list[dict]]
            Example:
            {
                "GBM": [
                    {"extra_trees": True, "ag_args": {"priority": -1}},
                    {"learning_rate": 0.03, "ag_args": {"priority": -2}},
                ],
                "CAT": [
                    {"ag_args": {"priority": -3}},
                ],
            }
        """
        hyperparameters = defaultdict(list)

        for i, e in enumerate(self.experiments):
            model_cls = copy.deepcopy(e.method_kwargs["model_cls"])
            model_hyperparameters = copy.deepcopy(e.method_kwargs["model_hyperparameters"])

            priority = -i - 1

            ag_args = model_hyperparameters.setdefault("ag_args", {})
            ag_args.setdefault("priority", priority)

            ag_key = model_cls.ag_key
            hyperparameters[ag_key].append(model_hyperparameters)

        return dict(hyperparameters)

    def export_fit_kwargs(self) -> dict:
        """
        Return shared AutoGluon fit kwargs across all experiments.

        Raises
        ------
        AssertionError
            If experiments have non-matching fit_kwargs.
        """
        if not self.experiments:
            return {}

        fit_kwargs = copy.deepcopy(self.experiments[0].method_kwargs["fit_kwargs"])

        for i, e in enumerate(self.experiments[1:], start=1):
            if e.method_kwargs["fit_kwargs"] != fit_kwargs:
                raise AssertionError(
                    "All experiments must have identical fit_kwargs to export "
                    f"an AutoGluon preset, but experiment 0 and experiment {i} differ.\n"
                    f"experiment 0 fit_kwargs: {fit_kwargs}\n"
                    f"experiment {i} fit_kwargs: {e.method_kwargs["fit_kwargs"]}"
                )

        return fit_kwargs

    def export_preset(self) -> dict:
        """
        Export a dict of AutoGluon fit arguments.

        Returns
        -------
        dict
            Example:
            {
                "hyperparameters": {...},
                "num_bag_folds": 8,
                "num_stack_levels": 1,
                ...
            }
        """
        preset = copy.deepcopy(self.export_fit_kwargs())
        preset["hyperparameters"] = self.export_hyperparameters()
        return preset
