import abc
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from heca.misc.td import TDAgentCon, TDScene
from heca.misc.base import Persistable


class Cursor(Enum):
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
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abc.abstractmethod
    def act(self, x: TDScene, y: TDScene) -> TDScene:
        raise NotImplementedError()

    @cached_property
    def precons(self) -> TDAgentCon:
        raise NotImplementedError

    @cached_property
    def postcons(self) -> TDAgentCon:
        raise NotImplementedError
