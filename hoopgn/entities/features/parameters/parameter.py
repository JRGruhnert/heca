from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from hoopgn.entities.properties.features.feature import (
    PropertyFeature,
    PropertyFeatureConfig,
)


@dataclass(kw_only=True)
class PropertyParameterConfig(PropertyFeatureConfig):
    pass


class PropertyParameter(PropertyFeature):
    def __init__(self, config: PropertyParameterConfig):
        super().__init__(config)
        self.config = config

    # TODO: This whole file is just there for legacy code.
    # Remove it as soon as starting with master
    def __call__(
        self,
        value: tuple | torch.Tensor | list[float] | float | int | None,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        else:
            if isinstance(value, (float, int)):
                return torch.tensor([value], dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported value type: {type(value)}")

    @abstractmethod
    def hoopgnv1(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        raise NotImplementedError("Subclasses must implement this method.")
