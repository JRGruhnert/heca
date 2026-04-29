from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import cached_property

from heca.classes.register import Registerable
from heca.entities.entity import Entity
from heca.environments.scene import Scene
from heca.misc.td import TDEntity, TDScene


class Cursor(Enum):
    IDLE = 1
    ERROR = 2
    RUNNING = 3


@dataclass(kw_only=True)
class AgentFeedback:
    reward: float
    done: bool
    terminal: bool
    cursor: Cursor = Cursor.IDLE
    can_learn: bool = False


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
    def sample(self) -> tuple[TDScene, TDScene]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def precons(self) -> list[tuple[Entity, TDEntity]]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def postcons(self) -> list[tuple[Entity, TDEntity]]:
        raise NotImplementedError()

    @abstractmethod
    def required_scenes(self) -> list[Scene.Query]:
        raise NotImplementedError()
