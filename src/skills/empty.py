from dataclasses import dataclass

import numpy as np
import torch
from calvin_env_modified.envs.observation import CalvinEnvObservation
from src.observation.observation import StateValueDict
from src.skills.skill import Skill, SkillConfig
from src.states.state import StateConfig


@dataclass
class EmptySkillConfig(SkillConfig):
    label: str = "EmptySkill"
    id: int = -1
    states: list[StateConfig] = []


class EmptySkill(Skill):
    def __init__(self, config: EmptySkillConfig = EmptySkillConfig()):
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
