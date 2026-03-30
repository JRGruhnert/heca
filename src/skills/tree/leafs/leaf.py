from dataclasses import dataclass
from functools import cached_property
import torch
from src.factory import select_operator
from src.observation.observation import StateValueDict
from src.operators.operator import OperatorConfig
from src.states.logic.condition import Condition


@dataclass
class LeafConfig:
    label: str
    id: int
    operator: OperatorConfig


class Leaf:
    def __init__(
        self,
        config: LeafConfig,
    ):
        self.config = config
        self.operator = select_operator(config.operator)

    def prepare(self, goal: StateValueDict):
        self.operator.reset(goal)

    def act(self, current: StateValueDict) -> torch.Tensor:
        return self.operator.act(current)

    @cached_property
    def parameter(self) -> set[str]:
        return set(self.precons.keys()) | set(self.postcons.keys())

    @cached_property
    def precons(self) -> dict[str, Condition]:
        return self.operator.load_parameter_precons()

    @cached_property
    def postcons(self) -> dict[str, Condition]:
        return self.operator.load_parameter_postcons()
