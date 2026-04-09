from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from src.observation.observation import StateValueDict
from src.skills.tree.operator import NodeOperator, NodeOperatorConfig
from src.objects.properties.condition import Condition


@dataclass
class TreeOperatorConfig(NodeOperatorConfig):
    pass


class TreeOperator(NodeOperator):
    def __init__(self, config: TreeOperatorConfig):
        self.config = config

    @abstractmethod
    def __call__(self, start: StateValueDict) -> np.ndarray | None:
        """Predict the next action given the current observation."""
        raise NotImplementedError("Subclasses must implement method.")

    def reset(self, goal: StateValueDict):
        """Prepare the operator for execution. Before each use."""
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
