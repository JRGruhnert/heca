from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
import numpy as np
from torch_geometric.data import Batch
from hoopgn import logger
from hoopgn.base import ConfigurableClass, RegisterableClass
from hoopgn.evaluators.evaluator import Evaluator
from hoopgn.generators.hoopgn import Hoopgn
from hoopgn.observation.td_scene import TDScene
from hoopgn.environments.properties.features.conditions.condition import (
    PropertyCondition,
)


@dataclass(kw_only=True)
class SkillInfo:
    id: int
    label: str
    description: str


class Agent(RegisterableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        ident: SkillInfo
        hoopgn: Hoopgn.Config
        evaluator: Evaluator.Config
        train: bool = False

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.hoopgn = Hoopgn.from_config(cfg.hoopgn)
        self.evaluator = Evaluator.from_config(cfg.evaluator)

    def reset(self, goal: TDScene):
        self.hoopgn.reset(goal)
        self.evaluator.reset(goal)

    def graph(self, x: TDScene) -> Batch:
        return self.hoopgn(x)

    def predict(self, x: TDScene) -> np.ndarray | None:
        return self.hoopgn(x)

    @cached_property
    @abstractmethod
    def property_labels(self) -> set[str]:
        raise NotImplementedError()
        # return set(self.precons.keys()) | set(self.postcons.keys())

    @cached_property
    @abstractmethod
    def precons(self) -> dict[str, PropertyCondition]:
        raise NotImplementedError()
        # return self.policy.load_precons()

    @cached_property
    @abstractmethod
    def postcons(self) -> dict[str, PropertyCondition]:
        raise NotImplementedError()
        # return self.policy.load_postcons()
