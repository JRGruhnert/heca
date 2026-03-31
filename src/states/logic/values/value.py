from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass
class ValueHandlerConfig:
    pass


class Value(ABC):
    def __init__(
        self,
        config: ValueHandlerConfig,
    ):
        self.config = config

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the processed value of the state as a tensor."""
        raise NotImplementedError("Subclasses must implement the value method.")

    @abstractmethod
    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the processed input value as a tensor."""
        raise NotImplementedError("Subclasses must implement the make_input method.")
