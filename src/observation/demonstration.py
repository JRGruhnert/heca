from dataclasses import dataclass

import torch


@dataclass
class DemoTrajectoryConfig:
    label: str
    id: int


class Demos:
    def __init__(
        self,
        config: DemoTrajectoryConfig,
    ):
        self.config = config
        self.precons_raw: dict[str, torch.Tensor] = {}
        self.postcons_raw: dict[str, torch.Tensor] = {}

    def _load_precons(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement method.")

    def _load_postcons(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement method.")
