from dataclasses import dataclass

import torch

from heca.properties.normalizers.normalizer import PropertyNormalizer


class DomainNormalizer(PropertyNormalizer):
    @dataclass(kw_only=True)
    class Config(PropertyNormalizer.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
