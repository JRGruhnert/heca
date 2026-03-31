from dataclasses import dataclass, field

from src.skills.tree.leafs.tapas.tapas_networker import (
    TapasNetworkerConfig,
    TapasNetworkerConfig,
)
from src.skills.tree.leafs.tapas.tapas_operator import TapasOperatorConfig
from src.skills.tree.networker import NodeNetworkerConfig
from src.skills.tree.node import TreeNodeConfig
from src.states.logic.distances.distance import ValueDistanceConfig
from src.states.state import ObjectConfig


@dataclass
class TapasConfig(TreeNodeConfig):
    states: list[ObjectConfig] = field(default_factory=list)
    overrides: set[str] = field(default_factory=set)
    childs: set[int] = field(default_factory=set)
    distance: ValueDistanceConfig = ValueDistanceConfig()
    networker: NodeNetworkerConfig = TapasNetworkerConfig()
    operator: TapasOperatorConfig = field(init=False)
    reversed: bool = field(init=False)

    def __post_init__(self):
        self.operator = TapasOperatorConfig(
            conditions={state.label: state.condition for state in self.states},
            label=self.label,
            reversed=self.reversed,
            overrides=self.overrides,
        )
        self.reversed = True if len(self.overrides) != 0 else False
