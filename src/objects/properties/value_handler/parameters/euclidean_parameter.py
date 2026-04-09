from dataclasses import dataclass

import torch

from src.objects.properties.value_handler.parameters.parameter import (
    StateParameter,
    StateParameterConfig,
)


@dataclass
class EuclideanParameterConfig(StateParameterConfig):
    pass


class EuclideanParameter(StateParameter):

    def __init__(self, config: EuclideanParameterConfig):
        super().__init__(config)
        self.config = config

    def hoopgnv1(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        assert isinstance(start, torch.Tensor), "start must be a torch.Tensor"
        assert isinstance(end, torch.Tensor), "end must be a torch.Tensor"
        if selected_by_tapas:
            if reversed:
                return end.mean(dim=0)
            return start.mean(dim=0)
        return None  # Not selected by tapas
