from dataclasses import dataclass

from hoopgn import logger
import torch.nn as nn


class PropertyEncoder(nn.Module):
    @dataclass
    class Config:
        label: str
        dim_input: int
        middle_dim: int = 16
        dim_encoder: int = 32

    def __init__(self, cfg: Config):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cfg.dim_input, cfg.middle_dim),
            nn.ReLU(),
            nn.Linear(cfg.middle_dim, cfg.dim_encoder),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(x)


@dataclass
class PropertyEncoderRegistryConfig:
    dynamic_create: bool = True
    dim_encoder: int = 32


class PropertyEncoderRegistry(nn.ModuleDict):
    def __init__(self, config: PropertyEncoderRegistryConfig):
        super().__init__()
        self.config = config

    def register(self, cfg: PropertyEncoder.Config):
        if cfg.label in self:
            logger.info(f"Encoder for '{cfg.label}' already registered. Skipping.")
            return
        if self.config.dynamic_create and cfg.dim_encoder != self.config.dim_encoder:
            raise ValueError(
                f"Dimension mismatch for '{cfg.label}': "
                f"expected {self.config.dim_encoder}, got {cfg.dim_encoder}. "
                f"Currently only uniform Node Feature Sizes are supported."
            )
        self[cfg.label] = PropertyEncoder(cfg)

    def get(self, label: str) -> PropertyEncoder:
        encoder = self[label]
        assert isinstance(encoder, PropertyEncoder)
        return encoder
