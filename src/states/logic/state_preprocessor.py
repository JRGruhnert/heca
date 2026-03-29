from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class StatePreprocessorConfig:
    pass


class StatePreprocessor(ABC):
    def __init__(self, config: StatePreprocessorConfig):
        self.config = config

    def __call__(self, value: torch.Tensor | list[float] | float | int) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            if isinstance(value, (float, int)):
                value = [value]
            value = torch.tensor(value, dtype=torch.float32)
        return self.process(value)

    @abstractmethod
    def process(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the preprocess method.")

    @abstractmethod
    def run(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        raise NotImplementedError("Subclasses must implement this method.")
