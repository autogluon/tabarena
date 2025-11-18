"""
Original code from https://github.com/dholzmueller/probmetrics
Credit to David Holzmüller
"""

from pathlib import Path
from typing import Union, Optional, List

import torch


# problem: should a distribution object store the tensor which it does computations on?
# In favor:
# then it knows whether it stores logits or probabilities etc.
# we only need to pass/return one object
# we can have distributions that are mixtures of other distributions
# Counterarguments:
# then batching might be more difficult (but do we need this?) -> could implement an indexing function?
# then we need to store a "constructor function" instead of just storing the constructed object.
# but maybe the "constructor function" can just be the class itself.
# or the heads just need to be implemented so that they know how to construct the distribution

def to_one_hot(y: torch.Tensor, n_classes: int, label_smoothing_eps: float = 0.0) -> torch.Tensor:
    """
    Creates a one-hot representation of y.
    :param y: Labels, shape (n_samples,).
    :param n_classes: Number of classes (needs to be specified because it might be larger than max(y)+1).
    :param label_smoothing_eps: Epsilon parameter for label smoothing.
    :return: Returns a one-hot representation of shape (n_samples, num_classes).
    """
    one_hot = torch.nn.functional.one_hot(y, n_classes).float()
    if label_smoothing_eps > 0.0:
        low = label_smoothing_eps / n_classes
        high = 1.0 - label_smoothing_eps + low
        return low + (high - low) * one_hot
    else:
        return one_hot


class Distribution:
    pass  # abstract base class

    # maybe have get_metrics() and save() function

    def get_metric_names(self) -> List[str]:
        pass  # indicate which metrics are applicable

    def save(self, folder: Union[str, Path]):
        # todo: implement
        pass  # need a way to serialize the distribution (for storing predictions). Typically just name and tensor?

    @staticmethod
    def load(folder: Union[str, Path]) -> 'Distribution':
        # todo: implement
        pass

    def get_n_samples(self) -> int:
        pass

    def __getitem__(self, *args) -> 'Distribution':
        raise NotImplementedError()


class CategoricalDistribution(Distribution):
    # probably need one version that takes logits and one that takes probabilities
    # or even more versions for different link functions?
    def get_logits(self) -> torch.Tensor:
        raise NotImplementedError()

    def get_probs(self) -> torch.Tensor:
        raise NotImplementedError()

    def get_modes(self) -> torch.Tensor:
        raise NotImplementedError()

    def get_n_classes(self) -> int:
        raise NotImplementedError()

    def get_n_samples(self) -> int:
        raise NotImplementedError()

    def is_dirac(self) -> bool:
        raise NotImplementedError()

    def __getitem__(self, *args) -> 'CategoricalDistribution':
        raise NotImplementedError()

    def get_metric_names(self) -> List[str]:
        pass  # todo: implement

    @staticmethod
    def from_labels(labels: torch.Tensor, n_classes: Optional[int] = None,
                    ls_eps: float = 0.0) -> 'CategoricalDistribution':
        if ls_eps == 0.0:
            return CategoricalDirac(labels=labels, n_classes=n_classes)

        if n_classes is None:
            n_classes = torch.max(labels).item() + 1
        return CategoricalProbs(probs=to_one_hot(labels, n_classes=n_classes, label_smoothing_eps=ls_eps))

    @staticmethod
    def from_mixture(distributions: List['CategoricalDistribution'], weights: List[float]) -> 'CategoricalDistribution':
        assert len(distributions) >= 1
        return CategoricalProbs(
            probs=sum([weight * dist.get_probs() for weight, dist in zip(weights, distributions)]))


class CategoricalLogits(CategoricalDistribution):
    # probably need one version that takes logits and one that takes probabilities
    # or even more versions for different link functions?
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.probs = None
        self.modes = None

    def get_logits(self) -> torch.Tensor:
        return self.logits

    def get_probs(self) -> torch.Tensor:
        if self.probs is None:
            self.probs = torch.nn.functional.softmax(self.logits, dim=-1)
        return self.probs

    def get_modes(self) -> torch.Tensor:
        # returns a tensor of shape (batch_dims)
        if self.modes is None:
            self.modes = self.logits.argmax(dim=-1)
        return self.modes

    def get_n_classes(self) -> int:
        return self.logits.shape[-1]

    def get_n_samples(self) -> int:
        return self.logits.shape[-2]

    def is_dirac(self) -> bool:
        return False

    def __getitem__(self, *args) -> 'CategoricalDistribution':
        return CategoricalLogits(self.logits.__getitem__(*args))


class CategoricalProbs(CategoricalDistribution):
    # probably need one version that takes logits and one that takes probabilities
    # or even more versions for different link functions?
    def __init__(self, probs: torch.Tensor):
        self.logits = None
        self.probs = probs
        self.modes = None

    def get_logits(self) -> torch.Tensor:
        if self.logits is None:
            self.logits = torch.log(self.probs + 1e-30)
        return self.logits

    def get_probs(self) -> torch.Tensor:
        return self.probs

    def get_modes(self) -> torch.Tensor:
        # returns a tensor of shape (batch_dims)
        if self.modes is None:
            self.modes = self.probs.argmax(dim=-1)
        return self.modes

    def get_n_classes(self) -> int:
        return self.probs.shape[-1]

    def get_n_samples(self) -> int:
        return self.probs.shape[-2]

    def is_dirac(self) -> bool:
        return False

    def __getitem__(self, *args) -> 'CategoricalDistribution':
        return CategoricalProbs(self.probs.__getitem__(*args))


class CategoricalDirac(CategoricalDistribution):
    def __init__(self, labels: torch.Tensor, n_classes: Optional[int]):
        """
        labels should have shape (n_samples,)
        :param labels:
        :param n_classes:
        """
        assert len(labels.shape) == 1
        if n_classes is None:
            n_classes = torch.max(labels).item() + 1
        self.n_classes = n_classes
        self.labels = labels
        self.probs = None
        self.logits = None

    def get_logits(self) -> torch.Tensor:
        if self.logits is None:
            self.probs = to_one_hot(self.labels, self.n_classes)
            self.logits = torch.log(self.probs + 1e-30)
        return self.logits

    def get_probs(self) -> torch.Tensor:
        if self.probs is None:
            self.probs = to_one_hot(self.labels, self.n_classes)
        return self.probs

    def get_modes(self) -> torch.Tensor:
        return self.labels

    def is_dirac(self) -> bool:
        return True

    def get_n_classes(self) -> int:
        return self.n_classes

    def get_n_samples(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, *args) -> 'CategoricalDirac':
        return CategoricalDirac(labels=self.labels.__getitem__(*args), n_classes=self.get_n_classes())
