from dataclasses import dataclass

import torch

from src.factory import (
    select_distance,
    select_eval_condition,
    select_value_condition,
)

from src.states.logic.condition import ConditionConfig
from src.states.logic.distances.distance import DistanceConfig
from src.states.logic.evaluations.evaluation import EvaluationConfig
from src.states.logic.values.value import ValueConfig


@dataclass
class StateConfig:
    id: int
    size: int
    label: str
    type_str: str
    distance_cnd_skill: DistanceConfig
    distance_cnd_goal: DistanceConfig
    eval_cnd: EvaluationConfig
    value_cnd: ValueConfig
    value_cnd_eval: ValueConfig | None
    condition: ConditionConfig


class State:
    def __init__(
        self,
        config: StateConfig,
    ):
        self.config = config
        self.distance_cnd_skill = select_distance(config.distance_cnd_skill)
        self.distance_cnd_goal = select_distance(config.distance_cnd_goal)
        self.eval_cnd = select_eval_condition(config.eval_cnd)
        self.value_cnd = select_value_condition(config.value_cnd)
        self.value_cnd_eval = (
            select_value_condition(config.value_cnd_eval)
            if config.value_cnd_eval is not None
            else self.value_cnd
        )

    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the value of the state as a tensor."""
        assert isinstance(x, torch.Tensor), "Input must be torch.Tensor"
        return self.value_cnd.make_input(x)

    def distance_to_skill(
        self,
        current: torch.Tensor,
        precon: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        assert isinstance(current, torch.Tensor) and isinstance(
            precon, torch.Tensor
        ), "Inputs must be torch.Tensor"
        current_norm = self.value_cnd(current)
        precon_norm = self.value_cnd(precon)
        value = self.distance_cnd_skill.distance(current_norm, precon_norm)
        assert isinstance(value, float), "Distance must be a float"
        assert 0.0 <= value <= 1.0, "Distance must be in [0.0, 1.0]"
        return value

    def distance_to_goal(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        assert isinstance(current, torch.Tensor) and isinstance(
            goal, torch.Tensor
        ), "Inputs must be torch.Tensor"
        current_norm = self.value_cnd(current)
        goal_norm = self.value_cnd(goal)
        value = self.distance_cnd_goal.distance(current_norm, goal_norm)
        assert isinstance(value, float), "Distance must be a float"
        assert 0.0 <= value <= 1.0, "Distance must be in [0.0, 1.0]"
        return value

    def evaluate(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> bool:
        """Evaluate success using injected strategy."""
        assert isinstance(current, torch.Tensor) and isinstance(
            goal, torch.Tensor
        ), "Inputs must be torch.Tensor"
        current_norm = self.value_cnd_eval(current)
        goal_norm = self.value_cnd_eval(goal)
        return self.eval_cnd(current_norm, goal_norm)
