from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class StateClassifierConfig:
    pass


class StateClassifier(nn.Module):
    def __init__(self, config: StateClassifierConfig):
        super().__init__()
        self.config = config
        self.fc = nn.Sequential()

    # TODO: get image as input and give output
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@dataclass
class StateClassifierRegistryConfig:
    dynamic_create: bool = True
    dim_encoder: int = 32


class StateClassifierRegistry:
    def __init__(self, config: StateClassifierRegistryConfig):
        self._dict = nn.ModuleDict()
        self.config = config

    def modules(self):
        return self._dict.values()

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __setitem__(self, key, value):
        self._dict[key] = value
