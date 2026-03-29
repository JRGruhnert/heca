from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass
class EvalConditionConfig:
    pass


class EvalCondition(ABC):
    """Abstract base class for success evaluation strategies."""

    @abstractmethod
    def evaluate(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")
