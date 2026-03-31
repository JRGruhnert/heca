from dataclasses import dataclass

import torch.nn as nn


@dataclass
class StateEncoderConfig:
    label: str
    dim_input: int
    middle_dim: int = 16
    dim_encoder: int = 32


class StateEncoder(nn.Module):
    def __init__(self, config: StateEncoderConfig):
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
class StateEncoderRegistryConfig:
    dynamic_create: bool = True


class StateEncoderRegistry:
    def __init__(self, config: StateEncoderRegistryConfig):
        self._dict = nn.ModuleDict()
        self.config = config

    def get(self, config: StateEncoderConfig):
        if config.label not in self._dict:
            if self.config.dynamic_create:
                self._dict[config.label] = StateEncoder(config)
            else:
                raise KeyError(
                    f"Encoder with key '{config.label}' not found and dynamic create is disabled."
                )
        return self._dict[config.label]

    def modules(self):
        return self._dict.values()

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __setitem__(self, key, value):
        self._dict[key] = value
