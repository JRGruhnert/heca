from abc import abstractmethod
from functools import cached_property

import torch

from dataclasses import dataclass
from heca.agents.agent import Agent
from heca.classes.register import Registerable
from heca.entities.entity import Entity
from heca.environments.scene import Scene
from heca.misc.td import TDScene, TDEntity


class SceneAgent(Agent):
    @dataclass(frozen=True, kw_only=True)
    class Query(Registerable.Query):
        type: str

    @dataclass(kw_only=True)
    class Config(Agent.Config):
        scene: Scene.Query

    def __init__(self, cfg: Config):
        self.cfg = cfg

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

    def required_scenes(self) -> list[Scene.Query]:
        return [self.cfg.scene]
