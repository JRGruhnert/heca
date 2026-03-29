from dataclasses import dataclass

import torch

from src.states.logic.rotation.quaternion import Quaternion, QuaternionConfig
from src.states.logic.state_preprocessor import (
    StatePreprocessor,
    StatePreprocessorConfig,
)


@dataclass
class QuatStatePreprocessorConfig(StatePreprocessorConfig):
    quaternion: QuaternionConfig = QuaternionConfig()


class QuatStatePreprocessor(StatePreprocessor):
    def __init__(
        self,
        config: QuatStatePreprocessorConfig,
    ):
        super().__init__(config)
        self.config = config
        self.quaternion = Quaternion(config.quaternion)

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
                return self.quaternion.mean(end)
            return self.quaternion.mean(start)
        return None
