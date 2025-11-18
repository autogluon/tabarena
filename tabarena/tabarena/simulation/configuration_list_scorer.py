from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..repository.abstract_repository import AbstractRepository


class ConfigurationListScorer:
    def __init__(self, tasks: List[str]):
        self.tasks: List[str] = tasks

    @classmethod
    def from_repo(cls, repo: "AbstractRepository", **kwargs):
        raise NotImplementedError()

    def score(self, configs: List[str]) -> float:
        """
        :param configs: list of configuration to select from.
        :return: a score obtained after evaluating the list of configurations. Current strategies include:
        * `SingleBestConfigScorer`: picking the test-error of the configuration with the lowest validation score
        * `EnsembleSelectionConfigScorer`: returning the test-error when evaluating an ensemble of the configurations
        where the weights are computed with validations scores
        """
        raise NotImplementedError()

    def subset(self, tasks: List[str]) -> "ConfigurationListScorer":
        raise NotImplementedError()