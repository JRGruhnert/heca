from dataclasses import dataclass

from src.skills.tree.leafs.tapas.tapas_operator import (
    TapasLeafOperator,
    TapasOperatorConfig,
)
from src.skills.parameter import NodeParameter, NodeParameterConfig
from src.objects.properties.condition import Condition, ConditionConfig


@dataclass
class TapasParameterConfig(NodeParameterConfig):
    precons: dict[str, ConditionConfig] | None
    postcons: dict[str, ConditionConfig] | None


class TapasParameter(NodeParameter):
    def __init__(self, config: TapasParameterConfig):
        super().__init__(config)
        self.config = config

    def __call__(self, start):
        "TODO: implement"
        raise NotImplementedError()

    def reset(self, goal):
        "TODO: implement"
        raise NotImplementedError()

    def load_operator(self, config: TapasOperatorConfig) -> TapasLeafOperator:
        raise NotImplementedError("Subclasses must implement method.")

    def load_demos(self):
        raise NotImplementedError("Subclasses must implement method.")

    def load_parameter_precons(self) -> dict[str, Condition]:
        if self.config.precons is None:
            return {}
        return {k: Condition(v) for k, v in self.config.precons.items()}
