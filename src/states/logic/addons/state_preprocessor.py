from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from src.observation.demonstration import Demo


@dataclass
class StatePreprocessorConfig:
    pass


class StatePreprocessor(ABC):
    def __init__(self, config: StatePreprocessorConfig):
        self.config = config

    def __call__(
        self,
        value: tuple | torch.Tensor | list[float] | float | int,
    ) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value
        else:
            if isinstance(value, (float, int)):
                return torch.tensor([value], dtype=torch.float32)
            elif isinstance(value, tuple):
                start, end, reversed, selected_by_tapas = value
                return self.run(start, end, reversed, selected_by_tapas)
            else:
                raise ValueError(f"Unsupported value type: {type(value)}")

    @abstractmethod
    def run(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        raise NotImplementedError("Subclasses must implement this method.")
