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
        return self.distance(a, b)

    @abstractmethod
    def distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        raise NotImplementedError("Subclasses must implement the distance method.")
