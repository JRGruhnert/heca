from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from heca.agents.agent import Agent


class GraphNode(ABC):
    x: torch.Tensor

    @property
    @abstractmethod
    def src(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def key(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dst(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class PosCompNode(GraphNode):
    comp: str
    x: torch.Tensor

    @property
    def src(self) -> str:
        raise NotImplementedError

    @property
    def key(self) -> str:
        raise NotImplementedError

    @property
    def dst(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class RotCompNode(GraphNode):
    comp: str
    x: torch.Tensor

    @property
    def src(self) -> str:
        raise NotImplementedError

    @property
    def key(self) -> str:
        raise NotImplementedError

    @property
    def dst(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class StateCompNode(GraphNode):
    comp: str
    x: torch.Tensor

    @property
    def src(self) -> str:
        raise NotImplementedError

    @property
    def key(self) -> str:
        raise NotImplementedError

    @property
    def dst(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class CompNode(GraphNode):
    comp: str
    entity: str
    x: Optional[torch.Tensor] = None

    @property
    def src(self) -> str:
        raise NotImplementedError

    @property
    def key(self) -> str:
        raise NotImplementedError

    @property
    def dst(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class StepMixNode(GraphNode):
    entity: str
    condition: str
    x: Optional[torch.Tensor] = None

    @property
    def src(self) -> str:
        raise NotImplementedError

    @property
    def key(self) -> str:
        raise NotImplementedError

    @property
    def dst(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class PosNode(GraphNode):
    entity: str
    x: torch.Tensor

    @property
    def src(self) -> str:
        raise NotImplementedError

    @property
    def key(self) -> str:
        raise NotImplementedError

    @property
    def dst(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class RotNode(GraphNode):
    entity: str
    x: torch.Tensor

    @property
    def src(self) -> str:
        raise NotImplementedError

    @property
    def key(self) -> str:
        raise NotImplementedError

    @property
    def dst(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class StateNode(GraphNode):
    entity: str
    x: torch.Tensor

    @property
    def src(self) -> str:
        raise NotImplementedError

    @property
    def key(self) -> str:
        raise NotImplementedError

    @property
    def dst(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class EntityNode(GraphNode):
    entity: str
    tag: str
    x: Optional[torch.Tensor] = None

    @property
    def src(self) -> str:
        raise NotImplementedError

    @property
    def key(self) -> str:
        raise NotImplementedError

    @property
    def dst(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class ConditionNode(GraphNode):
    condition: str
    x: Optional[torch.Tensor] = None

    @property
    def src(self) -> str:
        raise NotImplementedError

    @property
    def key(self) -> str:
        raise NotImplementedError

    @property
    def dst(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class OptionNode(GraphNode):
    condition: str
    agent: Agent.Config
    x: Optional[torch.Tensor] = None

    @property
    def src(self) -> str:
        raise NotImplementedError

    @property
    def key(self) -> str:
        raise NotImplementedError

    @property
    def dst(self) -> str:
        raise NotImplementedError
