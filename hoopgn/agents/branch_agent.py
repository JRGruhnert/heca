from dataclasses import dataclass

from hoopgn.environments.environment import Environment
from hoopgn.observation.td_scene import TDScene
from hoopgn.agents.agent import Skill


class Branch(Skill):
    @dataclass(kw_only=True)
    class Config(Skill.Config):
        environment: Environment.Config

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def sample(self) -> tuple[TDScene, TDScene]:
        raise NotImplementedError("TODO: implement branch sampling logic.")
