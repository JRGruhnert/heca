from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from src.states import state
from src.states.logic.value_handler.value_handler import (
    ValueHandler,
    ValueHandlerConfig,
)


@dataclass
class ValidatorConfig(ValueHandlerConfig):
    pass


class Validator(ValueHandler):
    def __init__(
        self,
        config: ValidatorConfig,
    ):
        self.config = config

    @abstractmethod
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the __call__ method.")
