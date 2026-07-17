from abc import ABC
from dataclasses import dataclass

from heca.agents.agent import Agent
from heca.conditions.condition import Condition
from heca.misc.data import DCEntity, DCScene


@dataclass(slots=True)
class GraphNode(ABC):
    changed: bool
    data: DCEntity
    sources: set[tuple[str, str]]


@dataclass(slots=True)
class EntityNode(GraphNode):
    entity: str
    data: DCEntity
    changed: bool = True
    static: bool = False
    weight: float = 1.0
    con: Condition | None = None


@dataclass(slots=True)
class OptionNode(GraphNode):
    agent: Agent.Config
    changed: bool = False
    data: DCScene = DCScene.empty()
