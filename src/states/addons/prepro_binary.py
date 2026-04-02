from dataclasses import dataclass

import torch

from src.states.addons.state_preprocessor import (
    StatePreprocessor,
    StatePreprocessorConfig,
)


@dataclass
class BinaryStatePreprocessorConfig(StatePreprocessorConfig):
    binary = BinaryConfig()


class BinaryStatePreprocessor(StatePreprocessor):
    def __init__(self, config: BinaryStatePreprocessorConfig):
        super().__init__(config)
        self.config = config

    def hoopgnv1(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        assert isinstance(start, torch.Tensor), "start must be a torch.Tensor"
        assert isinstance(end, torch.Tensor), "end must be a torch.Tensor"
        if bool(((start == 0.0) | (end == 1.0)).all()):
            candidate = end if reversed else start
            if ((end < start) | (start < end)).all():
                return candidate.mean(dim=0)
        return None  # Not constant enough
