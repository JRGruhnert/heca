from abc import abstractmethod
from functools import cached_property

from tensordict import TensorDict
import torch

from dataclasses import dataclass
from hoopgn.misc import logger
from hoopgn.agents.agent import Agent
from hoopgn.entities.entity import Entity
from hoopgn.properties.property import Property
from hoopgn.clusterers.custerer import Clusterer
from hoopgn.environments.environment import Environment
from hoopgn.converters.converter import Converter
from hoopgn.misc.td import TDScene, TDEntity


class LeafAgent(Agent):
    @dataclass(kw_only=True)
    class Query(Agent.Query):
        label: str

    @dataclass(kw_only=True)
    class Config(Agent.Config):
        environment: Environment.Query
        converter: Converter.Config
        clusterer: Clusterer.Config

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.step_counter = 0

    def sample_task(self) -> tuple[TDScene, TDScene]:
        env = Environment.search(self.cfg.environment)
        logger.info("Sampling new Task...")
        self.step_counter = 0
        x = env.sample()
        y = env.sample()
        attempts = 0
        while not self.evaluator.check_sample(x, y):
            attempts += 1
            if attempts % 5 == 0:
                x = env.sample()
            y = env.sample()
        return x, y

    def act(self, x: TDScene, y: TDScene) -> tuple[TDScene, float, bool]:
        z = self.execute(x, y)
        reward, done = self.evaluator.step(z)
        return z, reward, done

    def get_observation(self, x: TDScene) -> TensorDict:
        return x.get(self.cfg.environment.label)

    @abstractmethod
    def execute(self, x: TDScene, y: TDScene) -> TDScene:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def ppre(self) -> dict[Property.Query, torch.Tensor]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def ppost(self) -> dict[Property.Query, torch.Tensor]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def precons(self) -> dict[Entity.Query, TDEntity]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def postcons(self) -> dict[Entity.Query, TDEntity]:
        raise NotImplementedError()
