from dataclasses import dataclass

import torch

from hoopgn.objects.properties.quaternion import Quaternion, QuaternionConfig
from hoopgn.objects.properties.features.parameters.parameter import (
    PropertyParameter,
    PropertyParameterConfig,
)


@dataclass(kw_only=True)
class QuaternionParameterConfig(PropertyParameterConfig):
    rotation: QuaternionConfig = QuaternionConfig()


class QuaternionParameter(PropertyParameter):
    def __init__(self, config: QuaternionParameterConfig):
        super().__init__(config)
        self.config = config
        self.rotation = Quaternion(config.rotation)

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
                return self.rotation.mean(end)
            return self.rotation.mean(start)
        return None
