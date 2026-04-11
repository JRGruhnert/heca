from abc import abstractmethod
from dataclasses import dataclass

import torch

from hoopgn.networks.layers.classifiers.state_classifier import StateClassifierConfig


@dataclass(kw_only=True)
class XStateConfig:
    label: str
    values: set[str]
    classifier: StateClassifierConfig


class XState:
    def __init__(
        self,
        config: XStateConfig,
    ):
        self.config = config
        self.one_hots: dict[str, torch.Tensor] = {
            "default": torch.zeros(len(self.config.values), dtype=torch.float32)
        }

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.one_hot(self.label(x))

    @abstractmethod
    def label(self, x: torch.Tensor) -> str | None:
        raise NotImplementedError()

    def one_hot(self, label: str | None) -> torch.Tensor:
        if label is None:
            label = "default"
        assert label in self.config.values | {"default"}, "This is not a valid state"
        if label not in self.one_hots:
            onesis = torch.zeros(len(self.config.values), dtype=torch.float32)
            index = list(self.config.values).index(label)
            onesis[index] = 1.0
            self.one_hots[label] = onesis
        return self.one_hots[label]

    def make_one_hot(
        self,
        label: str | None,
    ) -> torch.Tensor:
        """Get one-hot encoded vector for the given area name."""
        one_hot = torch.zeros(len(self.config.values), dtype=torch.float32)
        if label in self.config.values:
            index = list(self.config.values).index(label)
            one_hot[index] = 1.0
        else:
            pass  # Area name not found, return all zeros
        return one_hot
