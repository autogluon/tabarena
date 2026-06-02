from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tabarena.repository.abstract_repository import AbstractRepository


class ConfigurationListScorer:
    def __init__(self, tasks: list[str]):
        self.tasks: list[str] = tasks

    @classmethod
    def from_repo(cls, repo: AbstractRepository, **kwargs):
        raise NotImplementedError()

    def score(self, configs: list[str]) -> float:
        """:param configs: list of configuration to select from.
        :return: a score obtained after evaluating the list of configurations. Current strategies include:
        * `SingleBestConfigScorer`: picking the test-error of the configuration with the lowest validation score
        * `EnsembleSelectionConfigScorer`: returning the test-error when evaluating an ensemble of the configurations
        where the weights are computed with validations scores
        """
        raise NotImplementedError()

    def subset(self, tasks: list[str]) -> ConfigurationListScorer:
        raise NotImplementedError()
