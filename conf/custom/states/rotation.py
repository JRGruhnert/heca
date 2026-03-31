from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.states.logic.addons.prepro_euclidean import EuclideanStatePreprocessorConfig
from src.states.logic.addons.prepro_rotation import RotationStatePreprocessorConfig
from src.states.logic.addons.state_preprocessor import StatePreprocessorConfig
from src.states.logic.condition import ConditionConfig
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
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="Quat",
        dim_input=4,
    )
    value_handler: QuaternionValueConfig = QuaternionValueConfig()
    distance_skill: AngularDistanceConfig = AngularDistanceConfig()
    distance_goal: AngularDistanceConfig = AngularDistanceConfig()
    eval_handler: ThresholdEvaluationConfig = ThresholdEvaluationConfig(
        distance=AngularDistanceConfig(),
    )
    condition: ConditionConfig = ConditionConfig(
        distance=AngularDistanceConfig(),
        preprocessor=RotationStatePreprocessorConfig(),
    )
    value_handler_eval: Value | None = None
