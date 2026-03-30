from dataclasses import dataclass
import torch

from src.observation.observation import StateValueDict
from src.skills.tree.leafs.loader import OperatorLoader, OperatorLoaderConfig
from src.states.logic.condition import Condition


@dataclass
class OperatorConfig:
    loader: OperatorLoaderConfig


class Operator:
    def __init__(self, config: OperatorConfig):
        self.config = config
        self.loader = OperatorLoader(config.loader)

    def reset(self, goal: StateValueDict):
        """Prepare the operator for execution. Before each use."""
        raise NotImplementedError("Subclasses must implement method.")

    def act(self, current: StateValueDict) -> torch.Tensor:
        """Get the next action for the operator."""
        raise NotImplementedError("Subclasses must implement method.")

    def load_demo_precons(self) -> dict[str, torch.Tensor]:
        """Load the preconditions from the demonstration."""
        raise NotImplementedError("Subclasses must implement method.")

    def load_demo_postcons(self) -> dict[str, torch.Tensor]:
        """Load the postconditions from the demonstration."""
        raise NotImplementedError("Subclasses must implement method.")

    def load_parameter_precons(self) -> dict[str, Condition]:
        """Load the preconditions for the operator parameters."""
        raise NotImplementedError("Subclasses must implement method.")

    def load_parameter_postcons(self) -> dict[str, Condition]:
        """Load the postconditions for the operator parameters."""
        raise NotImplementedError("Subclasses must implement method.")
