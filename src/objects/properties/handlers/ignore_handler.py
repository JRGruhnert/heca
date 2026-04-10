from dataclasses import dataclass

import torch

from src.objects.properties.handlers.handler import (
    ValueHandler,
    ValueHandlerConfig,
)


@dataclass(kw_only=True)
class IgnoreValueConfig(ValueHandlerConfig):
    pass


class IgnoreValue(ValueHandler):
    def __init__(
        self,
        config: IgnoreValueConfig,
    ):
        super().__init__(config)
        self.config = config

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
