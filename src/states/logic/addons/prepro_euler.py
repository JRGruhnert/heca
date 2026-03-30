from dataclasses import dataclass

import torch

from src.states.logic.state_preprocessor import (
    StatePreprocessor,
    StatePreprocessorConfig,
)


@dataclass
class EulerTapasAddonConfig(StatePreprocessorConfig):
    pass


class EulerStatePreprocessor(StatePreprocessor):

    def __init__(self, config: StatePreprocessorConfig):
        super().__init__(config)
        self.config = config

    def run(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        assert isinstance(start, torch.Tensor), "start must be a torch.Tensor"
        assert isinstance(end, torch.Tensor), "end must be a torch.Tensor"
        if selected_by_tapas:
            if reversed:
                return end.mean(dim=0)
            return start.mean(dim=0)
        return None  # Not selected by tapas
