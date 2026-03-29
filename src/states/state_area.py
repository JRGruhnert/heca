from dataclasses import dataclass
import torch

from src.states.logic.addon import AddonConfig
from src.states.logic.area.area_eval_cnd import AreaEvalCondition
from src.states.logic.area.area_value_cnd import AreaValueNormalizerConfig
from src.states.logic.eval_cnd import EvalConditionConfig
from src.states.logic.location.euclidean_distance_cnd import (
    EuclideanDistanceConditionConfig,
)
from src.states.logic.value_cnd import ValueConditionConfig
from src.states.state import State, StateConfig


@dataclass
class AreaStateConfig(StateConfig):
    value_cnd: AreaValueNormalizerConfig
    distance_cnd_skill: EuclideanDistanceConditionConfig
    distance_cnd_goal: EuclideanDistanceConditionConfig
    eval_cnd: EvalConditionConfig
    value_cnd_eval: ValueConditionConfig | None = None
    addons: dict[str, AddonConfig] | None = None
    label: str = "AreaEuler"
    size: int = 6


class AreaState(State):
    def __init__(self, config: AreaStateConfig):
        super().__init__(config)

    def is_in_an_existing_area(self, value: torch.Tensor) -> bool:
        """Checks if the given euler value is within the defined areas."""
        assert isinstance(
            self.value_cnd, AreaEvalCondition
        ), "Eval condition is not of type AreaEvalCondition."
        return self.value_cnd.is_in_area(value)
