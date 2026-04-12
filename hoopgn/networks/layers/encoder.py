from dataclasses import dataclass

from hoopgn import logger
import torch.nn as nn


@dataclass
class PropertyEncoderConfig:
    label: str
    dim_input: int
    middle_dim: int = 16
    dim_encoder: int = 32


class PropertyEncoder(nn.Module):
    def __init__(self, config: PropertyEncoderConfig):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.dim_input, config.middle_dim),
            nn.ReLU(),
            nn.Linear(config.middle_dim, config.dim_encoder),
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

    def register(self, config: PropertyEncoderConfig):
        if config.label in self:
            logger.info(f"Encoder for '{config.label}' already registered. Skipping.")
            return
        if self.config.dynamic_create and config.dim_encoder != self.config.dim_encoder:
            raise ValueError(
                f"Dimension mismatch for '{config.label}': "
                f"expected {self.config.dim_encoder}, got {config.dim_encoder}. "
                f"Currently only uniform Node Feature Sizes are supported."
            )
        self[config.label] = PropertyEncoder(config)

    def get(self, label: str) -> PropertyEncoder:
        encoder = self[label]
        assert isinstance(encoder, PropertyEncoder)
        return encoder
