from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property

from hoopgn.misc.classes import StorageClass
from hoopgn.entities.entity import Entity
from hoopgn.misc.td import TDScene


@dataclass(kw_only=True)
class AgentFeedback:
    reward: float
    done: bool
    terminal: bool
    can_learn: bool = False


class Agent(StorageClass):

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
