from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from src.states.logic.addon import Addon, AddonConfig


@dataclass
class TapasAddonConfig(AddonConfig):
    pass


class TapasAddon(Addon):
    def __init__(self, config: TapasAddonConfig):
        self.config = config

    def run(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        """Returns the Taskparameter or None"""
        raise NotImplementedError("Subclasses must implement this method.")
