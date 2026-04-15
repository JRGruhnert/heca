from dataclasses import dataclass

from hoopgn.environments.environment import EnvironmentConfig
from hoopgn.observation.td_scene import TDScene
from hoopgn.skills.skill import Skill, SkillConfig


@dataclass(kw_only=True)
class BranchConfig(SkillConfig):
    environment: EnvironmentConfig


class Branch(Skill):
    def __init__(self, config: BranchConfig):
        super().__init__(config)
        self.config = config

    def sample(self) -> tuple[TDScene, TDScene]:
        raise NotImplementedError("TODO: implement branch sampling logic.")
