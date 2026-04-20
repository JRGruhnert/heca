from dataclasses import dataclass

import torch

from hoopgn.properties.features.parameters.parameter import (
    PropertyParameter,
)
from hoopgn.properties.features.threshold_boundary import (
    BoundaryThreshold,
)


class ScalarParameter(PropertyParameter):
    @dataclass(kw_only=True)
    class Config(PropertyParameter.Config):
        threshold: BoundaryThreshold.Config

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.threshold = BoundaryThreshold(cfg.threshold)

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
