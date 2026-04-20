from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property

from hoopgn.agents.agent import Agent
from hoopgn.base import RegisterableClass
from hoopgn.environments.environment import Environment
from hoopgn.environments.properties.features.conditions.condition import (
    PropertyCondition,
)
from hoopgn.policies.policy import Policy


class LeafAgent(Agent):
    @dataclass(kw_only=True)
    class Identifier(RegisterableClass.Signature):
        label: str
        environment: Environment.Signature

    @dataclass(kw_only=True)
    class Config(Agent.Config):
        policies: Policy.Config

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    @cached_property
    @abstractmethod
    def property_labels(self) -> set[str]:
        # policies post and pres
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
