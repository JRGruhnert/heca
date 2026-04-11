from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from hoopgn.objects.properties.handlers.handler import (
    ValueHandler,
    ValueHandlerConfig,
)


@dataclass(kw_only=True)
class StateValidatorConfig(ValueHandlerConfig):
    pass


class StateValidator(ValueHandler):
    def __init__(
        self,
        config: StateValidatorConfig,
    ):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        raise NotImplementedError("Subclasses must implement the __call__ method.")
