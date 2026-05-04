from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import cached_property

from heca.classes.register import Registerable
from heca.entities.entity import Entity
from heca.entities.precon import Precon
from heca.environment.scenes.scene import Scene
from heca.misc.td import TDEntity, TDScene


class Cursor(Enum):
    IDLE = 1
    ERROR = 2
    ACTIVE = 3


@dataclass(kw_only=True)
class AgentFeedback:
    done: bool
    learn: bool
    state: Cursor
    reward: float
    terminal: bool


class Agent(Registerable):
    @dataclass(kw_only=True)
    class Config(Registerable.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def act(self, x: TDScene, y: TDScene) -> tuple[TDScene, AgentFeedback]:
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, x: TDScene, y: TDScene) -> AgentFeedback:
        raise NotImplementedError()

    @abstractmethod
    def sample(self) -> tuple[TDScene, TDScene]:
        raise NotImplementedError()

    @abstractmethod
    def gen_pre(self) -> list[tuple[Entity, TDEntity]]:
        raise NotImplementedError()

    @abstractmethod
    def gen_post(self) -> list[tuple[Entity, TDEntity]]:
        raise NotImplementedError()

    @cached_property
    def precons(self) -> list[tuple[Entity, TDEntity]]:
        return self.gen_pre()

    @cached_property
    def postcons(self) -> list[tuple[Entity, TDEntity]]:
        return self.gen_post()

    @abstractmethod
    def required_scenes(self) -> list[Scene.Query]:
        raise NotImplementedError()

    def make_options(self, x: TDScene, y: TDScene, con: Precon) -> list[HeteroData]:
        # take precon
        return [x]
