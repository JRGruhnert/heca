from dataclasses import dataclass, field
import torch

from hoopgn.properties.state import StateConfig, State


@dataclass(kw_only=True)
class LockedStateConfig(StateConfig):
    values: set[str] = field(default_factory=lambda: {"None"})


class LockedState(State):
    def __init__(self, config: LockedStateConfig):
        super().__init__(config)
        self.config = config

    def label(self, x: torch.Tensor) -> str:
        return "None"
