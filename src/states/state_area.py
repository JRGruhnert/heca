from dataclasses import dataclass
from src.states.logic.addon import AddonConfig
from src.states.logic.addons.addon_euler import EulerTapasAddonConfig
from src.states.logic.area.area_value_cnd import AreaValueConditionConfig
from src.states.logic.eval_cnd import EvalConditionConfig
from src.states.logic.location.euclidean_distance_cnd import (
    EuclideanDistanceConditionConfig,
)
from src.states.logic.value_cnd import ValueConditionConfig
from src.states.state import StateConfig


@dataclass
class AreaStateConfig(StateConfig):
    value_cnd: AreaValueConditionConfig
    distance_cnd_skill: EuclideanDistanceConditionConfig
    distance_cnd_goal: EuclideanDistanceConditionConfig
    eval_cnd: EvalConditionConfig
    label: str = "AreaEuler"
    size: int = 6
    value_cnd_eval: ValueConditionConfig | None = None
    addons = {
        "tapas": EulerTapasAddonConfig(),
    }
