from dataclasses import dataclass

from src.states.logic.boundary import BoundaryConfig
from src.states.logic.linear.linear_value_cnd import LinearValueNormalizerConfig
from src.states.logic.location.euclidean_distance_cnd import (
    EuclideanDistanceConditionConfig,
)

from src.states.logic.thresholds.threshold_eval_cnd import ThresholdEvalConditionConfig
from src.states.logic.value_cnd import ValueCondition
from src.states.state import StateConfig


@dataclass
class LocationStateConfig(StateConfig):
    label = "EulerPrecise"
    size = 3
    ignore = False
    distance_cnd_goal = EuclideanDistanceConditionConfig()
    distance_cnd_skill = EuclideanDistanceConditionConfig()
    eval_cnd = ThresholdEvalConditionConfig(
        distance=EuclideanDistanceConditionConfig(),
    )
    value_cnd = LinearValueNormalizerConfig(
        boundary=BoundaryConfig(
            lower_bound=[-1.0, -1.0, -1.0],
            upper_bound=[1.0, 1.0, 1.0],
        ),
    )
    value_cnd_eval: ValueCondition | None = None
