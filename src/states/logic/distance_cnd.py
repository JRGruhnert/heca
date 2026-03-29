from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass
class DistanceConditionConfig:
    pass


class DistanceCondition(ABC):
    def __init__(self, config: DistanceConditionConfig):
        self.config = config

    @abstractmethod
    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        raise NotImplementedError("Subclasses must implement the distance method.")
