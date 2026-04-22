from dataclasses import dataclass

import torch

from hoopgn.properties.rotations.quaternion import Quaternion
from hoopgn.properties.features.parameters.parameter import (
    PropertyParameter,
)


class QuaternionParameter(PropertyParameter):
    @dataclass(kw_only=True)
    class Config(PropertyParameter.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.rotation = Quaternion()

    def hoopgnv1(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        assert isinstance(start, torch.Tensor), "start must be a torch.Tensor"
        assert isinstance(end, torch.Tensor), "end must be a torch.Tensor"
        if selected_by_tapas:
            return self.rotation.mean(start)
        return None
