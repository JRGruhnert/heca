from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property

from hoopgn.misc.classes import StoragableClass
from hoopgn.entities.entity import Entity
from hoopgn.evaluators.evaluator import Evaluator
from hoopgn.misc.td import TDEntity, TDScene


@dataclass(kw_only=True)
class AgentFeedback:
    reward: float
    done: bool


class Agent(StoragableClass):
    @dataclass(kw_only=True)
    class Query(StoragableClass.Query):
        label: str

    @dataclass(kw_only=True)
    class Config(StoragableClass.Config):
        evaluator: Evaluator.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.evaluator = Evaluator.from_config(cfg.evaluator)

    @abstractmethod
    def act(self, x: TDScene, y: TDScene) -> tuple[TDScene, AgentFeedback]:
        raise NotImplementedError()

    @abstractmethod
    def sample(self) -> tuple[TDScene, TDScene]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def precons(self) -> dict[Entity.Query, TDEntity]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def postcons(self) -> dict[Entity.Query, TDEntity]:
        raise NotImplementedError()
