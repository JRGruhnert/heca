from dataclasses import dataclass

import numpy as np
import torch
from calvin_env_modified.envs.observation import CalvinEnvObservation
from src.observation.demonstration import Demo
from src.observation.observation import StateValueDict
from src.objects.properties.condition import ConditionConfig
from src.objects.properties.property import StateConfig


@dataclass
class IgnoreLeafConfig(TreeNodeConfig):
    label: str = "IgnoreLeaf"
    id: int = -1
    precons: dict[str, ConditionConfig] | None = None
    postcons: dict[str, ConditionConfig] | None = None


class IgnoreLeaf(Leaf):
    def __init__(self, config: IgnoreLeafConfig = IgnoreLeafConfig()):
        super().__init__(config)
        self.config = config

    def reset(self, goal: StateValueDict, env: object):
        pass

    def predict(
        self,
        current: CalvinEnvObservation,
    ) -> np.ndarray | None:
        return None

    def _load_demo_precons(self) -> dict[str, torch.Tensor]:
        return {}

    def _load_demo_postcons(self) -> dict[str, torch.Tensor]:
        return {}

    def _load_tp_precons(self) -> dict[str, torch.Tensor]:
        return {}

    def _load_tp_postcons(self) -> dict[str, torch.Tensor]:
        return {}
