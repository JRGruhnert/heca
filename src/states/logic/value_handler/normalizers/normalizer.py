from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from src.states.logic.value_handler.value_handler import (
    ValueHandler,
    ValueHandlerConfig,
)


@dataclass
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
