from dataclasses import dataclass
from functools import cached_property

import torch

from heca.properties.normalizers.boundary import BoundaryNormalizer
from heca.agents.experts.legacy.parameters.parameter import PropertyParameter


class RangeParameter(PropertyParameter):
    @dataclass(kw_only=True)
    class Config(PropertyParameter.Config):
        normalizer: BoundaryNormalizer.Config
        threshold: float = 0.05

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.normalizer = BoundaryNormalizer(cfg.normalizer)

    @cached_property
    def relative(self) -> torch.Tensor:
        """Returns the relative threshold for the state."""
        return self.cfg.threshold * (self.normalizer.lower - self.normalizer.upper)

    def hoopgnv1(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        assert isinstance(start, torch.Tensor), "start must be a torch.Tensor"
        assert isinstance(end, torch.Tensor), "end must be a torch.Tensor"
        std = start.std(dim=0)
        if (std < self.relative).all():
            return start.mean(dim=0)
        return None  # Not constant enough
