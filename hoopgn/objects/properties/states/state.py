from abc import abstractmethod
from dataclasses import dataclass

import torch

from hoopgn.networks.layers.classifiers.state_classifier import StateClassifierConfig


@dataclass(kw_only=True)
class StateConfig:
    values: set[str]


class State:
    def __init__(
        self,
        config: StateConfig,
    ):
        self.config = config
        assert len(config.values) > 0, "State must have at least one value."
        assert None not in config.values, "State values cannot be None."

        self.one_hots: dict[str, torch.Tensor] = {
            label: self.make_one_hot(label) for label in config.values
        }
        self.zeros = self.make_zeros()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        label = self.label(x)
        if label in self.config.values:
            return self.one_hots[label]
        else:
            return self.zeros

    @abstractmethod
    def label(self, x: torch.Tensor) -> str | None:
        raise NotImplementedError()

    def make_zeros(self) -> torch.Tensor:
        return torch.zeros(len(self.config.values), dtype=torch.float32)

    def make_one_hot(
        self,
        label: str | None,
    ) -> torch.Tensor:
        """Get one-hot encoded vector for the given area name."""
        assert (
            label is not None and label in self.config.values
        ), "Label cannot be None."
        one_hot = self.make_zeros()
        index = list(self.config.values).index(label)
        one_hot[index] = 1.0
        return one_hot
