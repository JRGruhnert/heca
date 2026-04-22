from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
import numpy as np
from torch_geometric.data import Batch
from hoopgn.base import RegisterableClass
from hoopgn.evaluators.evaluator import Evaluator
from hoopgn.generators.hoopgn import Hoopgn
from hoopgn.observation.td_entity import TDEntity
from hoopgn.observation.td_scene import TDScene


class Agent(RegisterableClass):
    # @dataclass(kw_only=True)
    class Signature(RegisterableClass.Signature):
        label: str  # does currently not add to the default signature
        description: str

    @dataclass(kw_only=True)
    class Config(RegisterableClass.Config):
        hoopgn: Hoopgn.Config
        evaluator: Evaluator.Config
        train: bool = False

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.hoopgn = Hoopgn.from_config(cfg.hoopgn)
        self.evaluator = Evaluator.from_config(cfg.evaluator)

    def graph(self, x: TDScene, y: TDScene) -> Batch:
        return self.hoopgn(x)

    @abstractmethod
    def act(self, x: TDScene, y: TDScene) -> tuple[float, bool, bool]:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x: TDScene) -> np.ndarray | None:
        raise NotImplementedError()

    @abstractmethod
    def sample_task(self) -> tuple[TDScene, TDScene]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def precons(self) -> dict[str, TDEntity]:
        raise NotImplementedError()
        # return self.policy.load_precons()

    @cached_property
    @abstractmethod
    def postcons(self) -> dict[str, TDEntity]:
        raise NotImplementedError()
        # return self.policy.load_postcons()
