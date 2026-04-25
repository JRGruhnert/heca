from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property

from hoopgn.misc.classes import StoragableClass
from hoopgn.entities.entity import Entity
from hoopgn.evaluators.evaluator import SceneEvaluator
from hoopgn.misc.td import TDEntity, TDScene


class Agent(StoragableClass):
    @dataclass(kw_only=True)
    class Query(StoragableClass.Query):
        label: str

    @dataclass(kw_only=True)
    class Config(StoragableClass.Config):
        evaluator: SceneEvaluator.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.evaluator = SceneEvaluator.from_config(cfg.evaluator)

    @abstractmethod
    def act(self, x: TDScene, y: TDScene) -> tuple[float, bool, bool]:
        raise NotImplementedError()

    @abstractmethod
    def sample_task(self) -> tuple[TDScene, TDScene]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def precons(self) -> dict[Entity.Query, TDEntity]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def postcons(self) -> dict[Entity.Query, TDEntity]:
        raise NotImplementedError()
