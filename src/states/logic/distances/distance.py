from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass
class DistanceConfig:
    pass


class Distance(ABC):
    def __init__(self, config: DistanceConfig):
        self.config = config

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> float:
        value = self.distance(a, b)
        assert isinstance(value, float), "Distance must be a float"
        assert 0.0 <= value <= 1.0, "Distance must be in [0.0, 1.0]"
        return value

    @abstractmethod
    def distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        raise NotImplementedError("Subclasses must implement the distance method.")
