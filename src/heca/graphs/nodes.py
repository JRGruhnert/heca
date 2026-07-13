from abc import ABC
from dataclasses import dataclass

import numpy as np

from heca.agents.agent import Agent


@dataclass
class GraphNode(ABC):
    tag: str
    changed: bool
    data: np.ndarray
    sources: set[tuple[str, str]]


@dataclass
class EntityNode(GraphNode):
    entity: str
    data: np.ndarray = np.zeros(0)
    changed: bool = True
    static: bool = False
    weight: float = 1.0


@dataclass
class OptionNode(GraphNode):
    tag: str
    agent: Agent.Config
    changed: bool = False
    data: np.ndarray = np.zeros(0)
