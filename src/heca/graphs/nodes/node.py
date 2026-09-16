from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from heca.experts.expert import ExpertModel
from heca.data.condition import Condition
from heca.data.data import DCEntity
from heca.graphs.roles import ENMode, ENRole, OPGate


@dataclass(slots=True, kw_only=True)
class GraphNode(ABC):
    sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set[str]))

    def __str__(self) -> str:
        src_str = ", ".join(f"{self.sources}" if self.sources else "∅")
        return f"sources=[{src_str}]"


@dataclass(slots=True, kw_only=True)
class EntityNode(GraphNode):
    entity: str
    type_id: int
    data: DCEntity
    n_states: int
    mode: ENMode
    role: ENRole
    con: Condition | None = None

    def __str__(self) -> str:
        src_str = ", ".join(f"{self.sources}" if self.sources else "∅")
        return (
            f"EntityNode\n"
            f"  entity:     {self.entity}\n"
            f"  sources:    [{src_str}]\n"
            f"  data:       {self.data}\n"
            f"  n_states:   {self.n_states}\n"
        )


@dataclass(slots=True, kw_only=True)
class CompNode(GraphNode):
    entity: str
    type_id: int
    data: DCEntity
    n_states: int
    weight: float


@dataclass(slots=True, kw_only=True)
class StateNode(GraphNode):
    role: OPGate = OPGate.NONE
    budget: float = 0.0


@dataclass(slots=True, kw_only=True)
class OptionNode(GraphNode):
    model: ExpertModel.Config
    gated: bool = False
    exec_count: int = 0
    recency: float = 0.0

    # OptionNode __str__:
    def __str__(self) -> str:
        src_str = ", ".join(f"{self.sources}" if self.sources else "∅")
        return (
            f"OptionNode" f"  model:      {self.model.tag}" f"  sources:    [{src_str}]"
        )
