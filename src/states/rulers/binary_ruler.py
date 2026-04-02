from dataclasses import dataclass

import torch

from src.states.rulers.ruler import Ruler, RulerConfig


@dataclass
class BinaryRulerConfig(RulerConfig):
    pass


class BinaryRuler(Ruler):
    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return torch.abs(current - x).item()
