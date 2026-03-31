from dataclasses import dataclass

import torch

from src.states.logic.addons.state_preprocessor import (
    StatePreprocessor,
    StatePreprocessorConfig,
)


@dataclass
class FlipStatePreprocessorConfig(StatePreprocessorConfig):
    pass


class FlipStatePreprocessor(StatePreprocessor):
    def __init__(self, config: FlipStatePreprocessorConfig):
        super().__init__(config)
        self.config = config

    def hoopgnv1(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        """Returns the mean of the given tensor values."""
        assert isinstance(start, torch.Tensor), "start must be a torch.Tensor"
        assert isinstance(end, torch.Tensor), "end must be a torch.Tensor"
        if (end == (1 - start)).all(dim=0).all():
            return torch.tensor([1.0])  # Flip state
        return None

    def process(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
