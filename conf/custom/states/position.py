from dataclasses import dataclass

from src.states.logic.addons.prepro_euclidean import EuclideanStatePreprocessorConfig
from src.states.logic.addons.state_preprocessor import StatePreprocessorConfig
from src.states.logic.boundary import AreaBoundaryConfig
from src.states.logic.values.value_linear import LinearValueConfig
from src.states.logic.distances.distance_euclidean import (
    EuclideanDistanceConfig,
)

from src.states.logic.evaluations.evaluation_threshold import ThresholdEvaluationConfig
from src.states.logic.values.value import Value
from src.states.state import StateConfig


@dataclass
class LocationStateConfig(StateConfig):
    size: int = 3
    type_str: str = "EulerPrecise"
    distance_cnd_goal: EuclideanDistanceConfig = EuclideanDistanceConfig()
    distance_cnd_skill: EuclideanDistanceConfig = EuclideanDistanceConfig()
    eval_cnd: ThresholdEvaluationConfig = ThresholdEvaluationConfig(
        distance=EuclideanDistanceConfig(),
    )
    value_cnd: LinearValueConfig = LinearValueConfig(
        boundary=AreaBoundaryConfig(),
    )
    value_cnd_eval: Value | None = None
    preprocessor_old: StatePreprocessorConfig = EuclideanStatePreprocessorConfig()
