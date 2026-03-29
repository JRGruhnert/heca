from dataclasses import dataclass

import torch

from src.skills.tree.leafs.operator import Operator, OperatorConfig
from src.states.logic.condition import Condition, ConditionConfig


@dataclass
class LeafLoaderConfig:
    precons: dict[str, ConditionConfig] | None
    postcons: dict[str, ConditionConfig] | None


class LeafLoader:
    def __init__(self, config: LeafLoaderConfig):
        self.config = config

    def load_operator(self, config: OperatorConfig) -> Operator:
        """Load the demonstration preconditions for the leaf."""
        # Load Demos

        raise NotImplementedError("Subclasses must implement method.")

    def load_demo_precons(self) -> dict[str, torch.Tensor]:
        """Load the demonstration preconditions for the leaf."""
        raise NotImplementedError("Subclasses must implement method.")

    def load_demo_postcons(self) -> dict[str, torch.Tensor]:
        """Load the demonstration postconditions for the leaf."""
        raise NotImplementedError("Subclasses must implement method.")

    def load_precons(self) -> dict[str, Condition]:
        """Load the demonstration preconditions for the leaf."""
        raise NotImplementedError("Subclasses must implement method.")

    def load_postcons(self) -> dict[str, Condition]:
        """Load the demonstration postconditions for the leaf."""
        # TODO: How to get postocons just from state label?
        for key, cfg in self.config.precons.items():
            self.precons[key] = Condition(config=cfg)
        if self.config.postcons:
            for key, cfg in self.config.postcons.items():
                self.postcons[key] = Condition(config=cfg)
