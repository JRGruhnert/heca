from dataclasses import dataclass

import torch


@dataclass
class OperatorConfig:
    pass


class Operator:
    def __init__(self, config: OperatorConfig):
        self.config = config

    def reset(self, *args, **kwargs):
        """Prepare the operator for execution. Before each use."""
        raise NotImplementedError("Subclasses must implement method.")

    def act(self, current, goal) -> torch.Tensor:
        """Get the next action for the operator."""
        raise NotImplementedError("Subclasses must implement method.")


@dataclass
class TapasOperatorConfig(OperatorConfig):
    pass


class TapasOperator(Operator):
    def __init__(self, config: TapasOperatorConfig):
        super().__init__(config)
        self.config = config

    def reset(self, *args, **kwargs):
        """Prepare the operator for execution. Before each use."""
        raise NotImplementedError("Subclasses must implement method.")

    def act(self, current, goal) -> torch.Tensor:
        """Get the next action for the operator."""
        raise NotImplementedError("Subclasses must implement method.")
