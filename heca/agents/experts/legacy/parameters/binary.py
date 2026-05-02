from dataclasses import dataclass

import torch

from heca.agents.experts.legacy.parameters.parameter import PropertyParameter
from heca.misc.binary import Binary


class BinaryParameter(PropertyParameter):
    @dataclass(kw_only=True)
    class Config(PropertyParameter.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def hoopgnv1(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        assert isinstance(start, torch.Tensor), "start must be a torch.Tensor"
        assert isinstance(end, torch.Tensor), "end must be a torch.Tensor"
        if Binary.is_binary(start) and Binary.is_binary(end):
            if Binary.is_always_same(start, end):
                return start.mean(dim=0)
        return None  # Not constant enough
