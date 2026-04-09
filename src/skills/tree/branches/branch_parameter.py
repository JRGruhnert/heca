from dataclasses import dataclass

from src.skills.tree.branches.branch_operator import (
    BranchOperator,
    BranchOperatorConfig,
)
from src.objects.properties.condition import Condition, ConditionConfig


@dataclass
class BranchParameterConfig:
    precons: dict[str, ConditionConfig] | None
    postcons: dict[str, ConditionConfig] | None


class BranchParameter:
    def __init__(self, config: BranchParameterConfig):
        self.config = config

    def load_operator(self, config: BranchOperatorConfig) -> BranchOperator:
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
