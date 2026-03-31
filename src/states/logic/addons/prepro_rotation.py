from dataclasses import dataclass

import torch

from src.states.logic.quaternion import Quaternion, QuaternionConfig
from src.states.logic.addons.state_preprocessor import (
    StatePreprocessor,
    StatePreprocessorConfig,
)


@dataclass
class QuaternionPreprocessorConfig(StatePreprocessorConfig):
    rotation: QuaternionConfig = QuaternionConfig()


class RotationStatePreprocessor(StatePreprocessor):
    def __init__(
        self,
        config: QuaternionPreprocessorConfig,
    ):
        super().__init__(config)
        self.config = config
        self.rotation = Quaternion(config.rotation)

    def hoopgnv1(
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
                return self.rotation.mean(end)
            return self.rotation.mean(start)
        return None
