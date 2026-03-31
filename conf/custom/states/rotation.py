from dataclasses import dataclass

from src.states.logic.addons.prepro_euclidean import EuclideanStatePreprocessorConfig
from src.states.logic.addons.state_preprocessor import StatePreprocessorConfig
from src.states.logic.distances.distance_angular import (
    AngularDistanceConfig,
)
from src.states.logic.values.value_quaternion import (
    QuaternionValueConfig,
)
from src.states.logic.evaluations.evaluation_threshold import ThresholdEvaluationConfig
from src.states.logic.values.value import Value
from src.states.state import StateConfig


@dataclass
class RotationStateConfig(StateConfig):
    size: int = 4
    type_str: str = "Quat"
    value_cnd: QuaternionValueConfig = QuaternionValueConfig()
    distance_cnd_skill: AngularDistanceConfig = AngularDistanceConfig()
    distance_cnd_goal: AngularDistanceConfig = AngularDistanceConfig()
    eval_cnd: ThresholdEvaluationConfig = ThresholdEvaluationConfig(
        distance=AngularDistanceConfig(),
    )
    value_cnd_eval: Value | None = None
    preprocessor_old: StatePreprocessorConfig = EuclideanStatePreprocessorConfig()
