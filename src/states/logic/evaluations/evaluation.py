from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass
class EvaluationConfig:
    pass


class Evaluation(ABC):
    """Abstract base class for success evaluation strategies."""

    @abstractmethod
    def __call__(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition for the given state."""
        raise NotImplementedError("Subclasses must implement the __call__ method.")
