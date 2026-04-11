from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass(kw_only=True)
class ValueHandlerConfig:
    pass


class ValueHandler(ABC):
    def __init__(
        self,
        config: ValueHandlerConfig,
    ):
        self.config = config

    @abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        assert all(
            isinstance(arg, torch.Tensor) for arg in args
        ), "All inputs must be torch.Tensor"
        raise NotImplementedError("Subclasses must implement the value method.")
