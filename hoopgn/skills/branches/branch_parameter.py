from dataclasses import dataclass

from hoopgn.skills.branches.branch_operator import (
    BranchOperator,
    BranchOperatorConfig,
)
from hoopgn.objects.properties.property_condition import (
    PropertyCondition,
    PropertyConditionConfig,
)


@dataclass(kw_only=True)
class BranchParameterConfig:
    precons: dict[str, PropertyConditionConfig] | None
    postcons: dict[str, PropertyConditionConfig] | None


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

    def load_parameter_precons(self) -> dict[str, PropertyCondition]:
        """Load the preconditions for the operator parameters."""
        if self.config.precons is None:
            return {}
        return {k: PropertyCondition(v) for k, v in self.config.precons.items()}
