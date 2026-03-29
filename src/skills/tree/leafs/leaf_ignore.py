from dataclasses import dataclass

import numpy as np
import torch
from calvin_env_modified.envs.observation import CalvinEnvObservation
from build.lib.src.skills.empty import EmptySkillConfig
from src.observation.demonstration import Demos
from src.observation.observation import StateValueDict
from src.skills.tree.leafs.leaf import Leaf, LeafConfig
from src.states.logic.condition import ConditionConfig
from src.states.state import StateConfig


@dataclass
class IgnoreLeafConfig(LeafConfig):
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
