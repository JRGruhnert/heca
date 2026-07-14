from abc import ABC
from dataclasses import dataclass

import numpy as np

from heca.agents.agent import Agent
from heca.conditions.condition import Condition


@dataclass
class NodeData:
    gnn: np.ndarray = np.zeros(0)
    env: np.ndarray = np.zeros(0)


@dataclass
class GraphNode(ABC):
    changed: bool
    data: NodeData
    sources: set[tuple[str, str]]


@dataclass
class EntityNode(GraphNode):
    entity: str
    data: NodeData  # = GoalEntry()
    changed: bool = True
    static: bool = False
    weight: float = 1.0
    con: Condition | None = None


@dataclass
class OptionNode(GraphNode):
    agent: Agent.Config
    changed: bool = False
    data: NodeData = NodeData()
