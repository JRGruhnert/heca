from dataclasses import dataclass

import torch

from hoopgn.objects.properties.handlers.handler import (
    ValueHandler,
    ValueHandlerConfig,
)
from hoopgn.objects.properties.x_state import XState, XStateConfig


@dataclass(kw_only=True)
class OneHotValueConfig(ValueHandlerConfig):
    state: XStateConfig


class OneHotValue(ValueHandler):
    def __init__(
        self,
        config: OneHotValueConfig,
    ):
        super().__init__(config)
        self.state = XState(config.state)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        lx = self.state(x)
        return torch.cat([x, lx], dim=0)
