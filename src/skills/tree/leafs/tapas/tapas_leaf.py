from dataclasses import dataclass, field

from src.skills.tree.leafs.tapas.tapas_networker import (
    TapasNetworkerConfig,
    TapasNetworkerConfig,
)
from src.skills.tree.leafs.tapas.tapas_operator import TapasOperatorConfig
from src.skills.tree.networker import NodeNetworkerConfig
from src.skills.tree.node import TreeNodeConfig
from src.states.logic.distance import DistanceConfig
from src.states.state import StateConfig


@dataclass
class TapasConfig(TreeNodeConfig):
    states: list[StateConfig] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)
    childs: list[int] = field(default_factory=list)
    distance: DistanceConfig = DistanceConfig()
    networker: NodeNetworkerConfig = TapasNetworkerConfig()
    operator: TapasOperatorConfig = field(init=False)
    reversed: bool = field(init=False)

    def __post_init__(self):
        self.operator = TapasOperatorConfig(
            label=self.label,
            reversed=self.reversed,
            overrides=self.overrides,
        )
        self.reversed = True if len(self.overrides) != 0 else False
