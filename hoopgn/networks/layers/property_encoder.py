from dataclasses import dataclass

import torch.nn as nn

from hoopgn.base import RegisterableClass


class PropertyEncoder(RegisterableClass, nn.Module):
    @dataclass(kw_only=True)
    class Signature(RegisterableClass.Signature):
        label: str

    @dataclass(kw_only=True)
    class Config(RegisterableClass.Config):
        dim_input: int
        middle_dim: int = 16
        dim_encoder: int = 32

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.fc = nn.Sequential(
            nn.Linear(cfg.dim_input, cfg.middle_dim),
            nn.ReLU(),
            nn.Linear(cfg.middle_dim, cfg.dim_encoder),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(x)
