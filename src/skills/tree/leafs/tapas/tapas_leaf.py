from dataclasses import dataclass, field

from src.skills.tree.leafs.tapas.tapas_networker import (
    TapasNetworkerConfig,
    TapasNetworkerConfig,
)
from src.skills.tree.leafs.tapas.tapas_operator import TapasOperatorConfig
from src.skills.networker import NodeNetworkerConfig
from src.skills.node import TreeNodeConfig
from src.objects.properties.property import StateConfig


@dataclass
class TapasConfig(TreeNodeConfig):
    states: list[StateConfig] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)
    childs: set[int] = field(default_factory=set)
    networker: NodeNetworkerConfig = TapasNetworkerConfig()
    operator: TapasOperatorConfig = field(init=False)
    reversed: bool = field(init=False)

    def __post_init__(self):
        self.operator = TapasOperatorConfig(
            conditions={state.label: state.condition for state in self.states},
            label=self.label,
            reversed=self.reversed,
            overrides=set(self.overrides),
        )
        self.reversed = True if len(self.overrides) != 0 else False
