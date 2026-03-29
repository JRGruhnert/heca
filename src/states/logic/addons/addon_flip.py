from dataclasses import dataclass

import torch

from src.states.logic.addons.addon_tapas import TapasAddon, TapasAddonConfig


@dataclass
class FlipTapasAddonConfig(TapasAddonConfig):
    pass


class FlipTapasAddon(TapasAddon):
    def __init__(self, config: FlipTapasAddonConfig):
        super().__init__(config)
        self.config = config

    def run(
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
