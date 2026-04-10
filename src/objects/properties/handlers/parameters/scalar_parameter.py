from dataclasses import dataclass

import torch

from src.objects.properties.handlers.parameters.parameter import (
    StateParameter,
    StateParameterConfig,
)
from src.objects.properties.threshold_boundary import (
    BoundaryThresholdConfig,
    BoundaryThreshold,
)


@dataclass(kw_only=True)
class ScalarParameterConfig(StateParameterConfig):
    threshold: BoundaryThresholdConfig


class ScalarParameter(StateParameter):
    def __init__(self, config: ScalarParameterConfig):
        super().__init__(config)
        self.config = config
        self.threshold = BoundaryThreshold(config.threshold)

    def hoopgnv1(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        assert isinstance(start, torch.Tensor), "start must be a torch.Tensor"
        assert isinstance(end, torch.Tensor), "end must be a torch.Tensor"
        if reversed:
            std = end.std(dim=0)
            if (std < self.threshold.relative).all():
                return end.mean(dim=0)
        else:
            std = start.std(dim=0)
            if (std < self.threshold.relative).all():
                return start.mean(dim=0)
        return None  # Not constant enough
