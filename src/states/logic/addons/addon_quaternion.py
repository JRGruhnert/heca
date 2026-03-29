from dataclasses import dataclass

import torch

from src.states.logic.addons.addon_tapas import TapasAddon, TapasAddonConfig
from src.states.logic.rotation.quaternion import Quaternion, QuaternionConfig


@dataclass
class QuatTapasAddonConfig(TapasAddonConfig):
    quaternion: QuaternionConfig = QuaternionConfig()


class QuatTapasAddon(TapasAddon):
    def __init__(
        self,
        config: QuatTapasAddonConfig,
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
