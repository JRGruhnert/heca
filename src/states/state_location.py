from dataclasses import dataclass

from src.states.euler import EulerState
from src.states.logic.boundary import BoundaryConfig
from src.states.logic.eval_cnd import EvalCondition
from src.states.logic.linear.linear_value_cnd import LinearValueNormalizerConfig
from src.states.logic.location.euclidean_distance_cnd import (
    EuclideanDistanceConditionConfig,
)
from src.states.logic.precise.precise_eval_cnd import (
    PreciseEvalCondition,
    PreciseEvalConditionConfig,
)
from src.states.state import StateConfig


@dataclass
class LocationStateConfig(StateConfig):
    type_str = "EulerPrecise"
    size = 3
    ignore = False
    value_cnd = LinearValueNormalizerConfig(
        boundary=BoundaryConfig(
            lower_bound=[-1.0, -1.0, -1.0],
            upper_bound=[1.0, 1.0, 1.0],
        ),
    )
    eval_cnd = PreciseEvalConditionConfig(
        distance=EuclideanDistanceConditionConfig(),
    )

    eval_normalizer: ValueCondition | None = None


class LocationState(EulerState):
    def __init__(
        self,
        name,
        id,
        ignore: bool = False,
    ):
        super().__init__(
            name=name,
            id=id,
            type_str="EulerPrecise",
            size=3,
            ignore=ignore,
            eval_condition=PreciseEvalCondition(
                condition=EulerDistanceCondition(),
            ),
        )
