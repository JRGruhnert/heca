from dataclasses import dataclass

import torch

from hoopgn.objects.properties.features.parameters.parameter import (
    PropertyParameter,
    PropertyParameterConfig,
)


@dataclass(kw_only=True)
class EuclideanParameterConfig(PropertyParameterConfig):
    pass


class EuclideanParameter(PropertyParameter):

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
