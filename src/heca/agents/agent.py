import abc
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from heca.conditions.pair import ConPair
from heca.misc.data import DCScene
from heca.misc.base import Persistable
from heca.misc.entity import Entity


class EEState(Enum):
    IDLE = 1
    ERROR = 2
    ACTIVE = 3


@dataclass(kw_only=True)
class AgentFeedback:
    done: bool
    reward: float
    terminal: bool


class Agent(Persistable, abc.ABC):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        n_samples: int = 1000
        folder: str = "agents"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abc.abstractmethod
    def act(self, x: DCScene, y: DCScene) -> DCScene:
        raise NotImplementedError

    @cached_property
    def entities(self) -> set[Entity]:
        raise NotImplementedError

    @cached_property
    def conditions(self) -> list[ConPair]:
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self):
        raise NotImplementedError
