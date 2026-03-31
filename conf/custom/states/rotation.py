from dataclasses import dataclass

from src.states.logic.addons.prepro_euclidean import EuclideanStatePreprocessorConfig
from src.states.logic.addons.state_preprocessor import StatePreprocessorConfig
from src.states.logic.rotation.quaternion_distance_cnd import (
    QuaternionDistanceConditionConfig,
)
from src.states.logic.rotation.quaternion_value_cnd import (
    QuaternionValueConditionConfig,
)
from src.states.logic.thresholds.threshold_eval_cnd import ThresholdEvalConditionConfig
from src.states.logic.value_cnd import ValueCondition
from src.states.state import StateConfig


@dataclass
class RotationStateConfig(StateConfig):
    size: int = 4
    type_str: str = "Quat"
    value_cnd: QuaternionValueConditionConfig = QuaternionValueConditionConfig()
    distance_cnd_skill: QuaternionDistanceConditionConfig = (
        QuaternionDistanceConditionConfig()
    )
    distance_cnd_goal: QuaternionDistanceConditionConfig = (
        QuaternionDistanceConditionConfig()
    )
    eval_cnd: ThresholdEvalConditionConfig = ThresholdEvalConditionConfig(
        distance=QuaternionDistanceConditionConfig(),
    )
    value_cnd_eval: ValueCondition | None = None
    preprocessor_old: StatePreprocessorConfig = EuclideanStatePreprocessorConfig()
