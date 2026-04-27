from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property

from heca.misc.classes import Persistable
from heca.entities.entity import Entity
from heca.misc.td import TDScene


@dataclass(kw_only=True)
class AgentFeedback:
    reward: float
    done: bool
    terminal: bool
    can_learn: bool = False


class Agent(Persistable):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def act(
        self, x: TDScene, y: TDScene, e: Entity | None = None
    ) -> tuple[TDScene, AgentFeedback]:
        raise NotImplementedError()

    @abstractmethod
    def sample(self) -> tuple[TDScene, TDScene]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def precons(self) -> list[Entity]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def postcons(self) -> list[Entity]:
        raise NotImplementedError()
