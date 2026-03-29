from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass
class AddonConfig:
    label: str


class Addon(ABC):
    """Abstract base class for additional logic components that are only available at runtime."""

    @abstractmethod
    def run(self, *args, **kwargs) -> torch.Tensor | None:
        """Execute the addon logic."""
        raise NotImplementedError("Subclasses must implement the run method.")
