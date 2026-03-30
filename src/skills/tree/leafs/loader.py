from dataclasses import dataclass

from src.skills.tree.leafs.operators.operator import Operator, OperatorConfig
from src.states.logic.condition import Condition, ConditionConfig


@dataclass
class OperatorLoaderConfig:
    precons: dict[str, ConditionConfig] | None
    postcons: dict[str, ConditionConfig] | None


class OperatorLoader:
    def __init__(self, config: OperatorLoaderConfig):
        self.config = config

    def load_operator(self, config: OperatorConfig) -> Operator:
        """Load the demonstration preconditions for the leaf."""
        # Load Demos

        raise NotImplementedError("Subclasses must implement method.")

    def load_demos(self):
        """Load the demonstration preconditions for the leaf."""
        raise NotImplementedError("Subclasses must implement method.")

    def load_parameter_precons(self) -> dict[str, Condition]:
        """Load the preconditions for the operator parameters."""
        if self.config.precons is None:
            return {}
        return {k: Condition(v) for k, v in self.config.precons.items()}
