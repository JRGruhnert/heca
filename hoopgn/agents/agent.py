from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
import numpy as np
from torch_geometric.data import Batch
from hoopgn.base import RegisterableClass
from hoopgn.evaluators.evaluator import Evaluator
from hoopgn.generators.hoopgn import Hoopgn
from hoopgn.observation.td_scene import TDScene
from hoopgn.properties.features.conditions.condition import (
    PropertyCondition,
)


class Agent(RegisterableClass):
    # @dataclass(kw_only=True)
    # class Signature(RegisterableClass.Signature):
    #    label: str # does currently not add to the default signature
    @dataclass(kw_only=True)
    class Config(RegisterableClass.Config):
        description: str
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
