from abc import abstractmethod
from functools import cached_property

from tensordict import TensorDict
import torch

from dataclasses import dataclass
from heca.agents.agent import Agent, AgentFeedback
from heca.entities.entity import Entity
from heca.properties.property import Property
from heca.environments.environment import Environment
from heca.misc.td import TDScene, TDEntity


class LeafAgent(Agent):
    @dataclass(kw_only=True)
    class Query(Agent.Query):
        env: Environment.Query

    @dataclass(kw_only=True)
    class Config(Agent.Config):
        query: "LeafAgent.Query"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # def sample(self) -> tuple[TDScene, TDScene]:
    #    env = Environment.search(self.cfg.query.env)
    #    logger.info("Sampling new Task...")
    #    self.step_counter = 0
    #    x = env.sample()
    #    y = env.sample()
    #    attempts = 0
    #    while not self.evaluator.is_sample(x, y):
    #        attempts += 1
    #        if attempts % 5 == 0:
    #            x = env.sample()
    #        y = env.sample()
    #    return x, y

    def act(self, x: TDScene, y: TDScene) -> tuple[TDScene, AgentFeedback]:
        z = self.execute(x, y)
        return z, AgentFeedback(reward=0.0, done=True, terminal=True)

    def get_observation(self, x: TDScene) -> TensorDict:
        return x.leaf

    @abstractmethod
    def execute(self, x: TDScene, y: TDScene) -> TDScene:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def ppre(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def ppost(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def precons(self) -> dict[Entity, TDEntity]:
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def postcons(self) -> dict[Entity, TDEntity]:
        raise NotImplementedError()
