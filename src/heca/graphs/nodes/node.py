from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from heca.experts.expert import ExpertModel
from heca.data.condition import Condition
from heca.data.data import DCEntity
from heca.graphs.roles import ROLE_OTHER


class ValueMode(Enum):
    GOAL = "Goal"
    START = "Start"
    SAMPLE = "Sample"
    SUBGOAL = "Subgoal"


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
    vmode: ValueMode
    role: int = ROLE_OTHER
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
    role: int
    data: float = 0.0


@dataclass(slots=True, kw_only=True)
class OptionNode(GraphNode):
    model: ExpertModel.Config
    effect: np.ndarray
    gated: bool = False

    # OptionNode __str__:
    def __str__(self) -> str:
        src_str = ", ".join(f"{self.sources}" if self.sources else "∅")
        return (
            f"OptionNode" f"  model:      {self.model.tag}" f"  sources:    [{src_str}]"
        )
