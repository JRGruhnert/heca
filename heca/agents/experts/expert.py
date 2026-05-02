from abc import abstractmethod
from dataclasses import dataclass
from heca.agents.agent import Agent
from heca.classes.register import Registerable
from heca.environments.scenes.scene import Scene
from heca.misc.td import TDScene


class ExpertAgent(Agent):
    @dataclass(frozen=True, kw_only=True)
    class Query(Registerable.Query):
        type: str

    @dataclass(kw_only=True)
    class Config(Agent.Config):
        scene: Scene.Query

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def execute(self, x: TDScene, y: TDScene) -> TDScene:
        raise NotImplementedError()

    def required_scenes(self) -> list[Scene.Query]:
        return [self.cfg.scene]
