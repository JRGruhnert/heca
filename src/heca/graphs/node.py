from abc import ABC
from dataclasses import dataclass

from heca.agents.agent import Agent
from heca.conditions.condition import Condition
from heca.misc.data import DCEntity, DCScene


@dataclass(slots=True, kw_only=True)
class GraphNode(ABC):
    changed: bool
    data: DCEntity
    sources: set[tuple[str, str]]

    def __str__(self) -> str:
        src_str = (
            ", ".join(f"({e},{k})" for e, k in self.sources) if self.sources else "∅"
        )
        return f"changed={self.changed} data={self.data} sources=[{src_str}]"


@dataclass(slots=True, kw_only=True)
class EntityNode(GraphNode):
    entity: str
    data: DCEntity
    n_states: int
    changed: bool = True
    static: bool = False
    weight: float = 1.0
    con: Condition | None = None

    # EntityNode __str__:
    def __str__(self) -> str:
        src_str = (
            ", ".join(f"({e},{k})" for e, k in self.sources) if self.sources else "∅"
        )
        return (
            f"EntityNode\n"
            f"  entity:     {self.entity}\n"
            f"  sources:    [{src_str}]\n"
            f"  changed:    {self.changed}\n"
            f"  data:       {self.data}\n"
            f"  n_states:   {self.n_states}\n"
            f"  static:     {int(self.static)}\n"
            f"  weight:     {self.weight:.2f}"
        )


@dataclass(slots=True, kw_only=True)
class OptionNode(GraphNode):
    agent: Agent.Config
    changed: bool = False
    data: DCScene = DCScene.empty()

    # OptionNode __str__:
    def __str__(self) -> str:
        src_str = (
            ", ".join(f"({e},{k})" for e, k in self.sources) if self.sources else "∅"
        )
        return (
            f"OptionNode"
            f"  agent:      {self.agent.tag}"
            f"  sources:    [{src_str}]"
            f"  changed:    {self.changed}"
        )
