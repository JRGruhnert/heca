from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from hoopgn.objects.properties.handlers.handler import (
    ValueHandler,
    ValueHandlerConfig,
)


@dataclass(kw_only=True)
class NormalizerConfig(ValueHandlerConfig):
    pass


class Normalizer(ValueHandler):
    def __init__(
        self,
        config: NormalizerConfig,
    ):
        self.config = config

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the processed value of the state as a tensor."""
