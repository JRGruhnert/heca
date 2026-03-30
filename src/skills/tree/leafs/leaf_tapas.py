from dataclasses import dataclass, field
from src.skills.tree.leafs.loader import OperatorLoaderConfig
from src.skills.tree.leafs.operators.operator import OperatorConfig
from src.skills.tree.leafs.leaf import LeafConfig
from src.skills.tree.leafs.operators.tapas import TapasOperatorConfig
from src.states.logic.distance import DistanceConfig


@dataclass
class TapasLeafConfig(LeafConfig):
    reversed: bool
    overrides: list[str]
    distance: DistanceConfig = DistanceConfig()
    loader: OperatorLoaderConfig = OperatorLoaderConfig(
        precons=None,
        postcons=None,
    )
    operator: OperatorConfig = field(init=False)

    def __post_init__(self):
        self.operator = TapasOperatorConfig(
            label=self.label,
            reversed=self.reversed,
            overrides=self.overrides,
            loader=self.loader,
        )
