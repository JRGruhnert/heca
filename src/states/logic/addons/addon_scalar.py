from dataclasses import dataclass

import torch

from src.states.logic.addons.addon_tapas import TapasAddon, TapasAddonConfig
from src.states.logic.threshold import RelativeThreshold, RelativeThresholdConfig


@dataclass
class ScalarTapasAddonConfig(TapasAddonConfig):
    threshold: RelativeThresholdConfig


class ScalarTapasAddon(TapasAddon):
    def __init__(self, config: ScalarTapasAddonConfig):
        super().__init__(config)
        self.config = config
        self.threshold = RelativeThreshold(config.threshold)

    def run(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        assert isinstance(start, torch.Tensor), "start must be a torch.Tensor"
        assert isinstance(end, torch.Tensor), "end must be a torch.Tensor"
        if reversed:
            std = end.std(dim=0)
            if (std < self.threshold.relative).all():
                return end.mean(dim=0)
        else:
            std = start.std(dim=0)
            if (std < self.threshold.relative).all():
                return start.mean(dim=0)
        return None  # Not constant enough
