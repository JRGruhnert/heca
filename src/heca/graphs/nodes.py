from abc import ABC
from dataclasses import dataclass

import numpy as np

from heca.agents.agent import Agent
from heca.conditions.condition import Condition


@dataclass(slots=True)
class NodeData:
    gnn: np.ndarray = np.zeros(0)
    env: np.ndarray = np.zeros(0)


@dataclass(slots=True)
class GraphNode(ABC):
    changed: bool
    data: NodeData
    sources: set[tuple[str, str]]


@dataclass(slots=True)
class EntityNode(GraphNode):
    entity: str
    data: NodeData
    changed: bool = True
    static: bool = False
    weight: float = 1.0
    con: Condition | None = None


@dataclass(slots=True)
class OptionNode(GraphNode):
    agent: Agent.Config
    changed: bool = False
    data: NodeData = NodeData()
