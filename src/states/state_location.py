from dataclasses import dataclass

from src.states.logic.addons.addon_euler import EulerTapasAddonConfig
from src.states.logic.boundary import AreaBoundaryConfig, BoundaryConfig
from src.states.logic.linear.linear_value_cnd import LinearValueNormalizerConfig
from src.states.logic.location.euclidean_distance_cnd import (
    EuclideanDistanceConditionConfig,
)

from src.states.logic.thresholds.threshold_eval_cnd import ThresholdEvalConditionConfig
from src.states.logic.value_cnd import ValueCondition
from src.states.state import StateConfig


@dataclass
class LocationStateConfig(StateConfig):
    size: int = 3
    type_str: str = "EulerPrecise"
    distance_cnd_goal: EuclideanDistanceConditionConfig = (
        EuclideanDistanceConditionConfig()
    )
    distance_cnd_skill: EuclideanDistanceConditionConfig = (
        EuclideanDistanceConditionConfig()
    )
    eval_cnd: ThresholdEvalConditionConfig = ThresholdEvalConditionConfig(
        distance=EuclideanDistanceConditionConfig(),
    )
    value_cnd: LinearValueNormalizerConfig = LinearValueNormalizerConfig(
        boundary=AreaBoundaryConfig(),
    )
    value_cnd_eval: ValueCondition | None = None
    addons: dict[str, EulerTapasAddonConfig] = {
        "tapas": EulerTapasAddonConfig(),
    }
