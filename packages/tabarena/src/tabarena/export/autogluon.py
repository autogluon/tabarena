from __future__ import annotations

import copy
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tabarena.benchmark.experiment.experiment_constructor import AGModelExperiment
    from tabarena.benchmark.validation_protocol import ValidationProtocol


class AutoGluonExporter:
    def __init__(self, experiments: list[AGModelExperiment]):
        self.experiments = experiments

    def export_hyperparameters(self) -> dict[str, list[dict]]:
        """Convert TabArena AGModelExperiment objects into an AutoGluon-compatible
        hyperparameters dictionary.

        Returns:
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

    def export_validation_protocol(self, validation_protocol: ValidationProtocol | None = None) -> ValidationProtocol:
        """The one validation protocol the exported preset bags under.

        ``validation_protocol`` wins when given; otherwise the experiments' shared protocol is used.
        Raises when no protocol is known (unstamped experiments) or the experiments disagree, since a
        plain ``TabularPredictor`` needs explicit counts.
        """
        if validation_protocol is not None:
            return validation_protocol
        protocols = {e.validation_protocol for e in self.experiments if e.validation_protocol is not None}
        if len(protocols) > 1:
            raise ValueError(
                "All experiments must share one validation protocol to export an AutoGluon preset, got: "
                f"{sorted(p.key() for p in protocols)}",
            )
        if not protocols:
            raise ValueError(
                "The experiments carry no validation protocol (an arena context stamps it at build_jobs); "
                "pass `validation_protocol=` to export the bagging counts of a preset.",
            )
        return protocols.pop()

    def export_fit_kwargs(self, validation_protocol: ValidationProtocol | None = None) -> dict:
        """Return shared AutoGluon fit kwargs across all experiments, incl. the protocol's bagging counts.

        ``num_bag_folds`` / ``num_bag_sets`` (and ``adapt_num_bag_folds_to_n_classes`` when the protocol
        asks for it) come from :meth:`export_validation_protocol`; a tiny-data regime cannot be
        expressed in a preset, so the protocol's default-regime counts are exported.

        Raises:
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
                    f"experiment {i} fit_kwargs: {e.method_kwargs['fit_kwargs']}",
                )

        protocol = self.export_validation_protocol(validation_protocol)
        fit_kwargs["num_bag_folds"] = protocol.num_bag_folds
        fit_kwargs["num_bag_sets"] = protocol.num_bag_sets
        if protocol.adapt_num_folds_to_n_classes:
            fit_kwargs["adapt_num_bag_folds_to_n_classes"] = True
        return fit_kwargs

    def export_preset(self, validation_protocol: ValidationProtocol | None = None) -> dict:
        """Export a dict of AutoGluon fit arguments.

        Returns:
        -------
        dict
            Example:
            {
                "hyperparameters": {...},
                "num_bag_folds": 8,
                "num_bag_sets": 1,
                ...
            }
        """
        preset = copy.deepcopy(self.export_fit_kwargs(validation_protocol=validation_protocol))
        preset["hyperparameters"] = self.export_hyperparameters()
        return preset
