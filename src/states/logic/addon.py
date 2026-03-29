from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass
class AddonConfig:
    pass


class Addon(ABC):
    @abstractmethod
    def run(self, *args, **kwargs) -> torch.Tensor | None:
        """Execute the addon logic."""
        raise NotImplementedError("Subclasses must implement the run method.")
