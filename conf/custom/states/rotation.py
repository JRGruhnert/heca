from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.states.logic.addons.prepro_rotation import QuaternionPreprocessorConfig
from src.states.logic.condition import ConditionConfig
from src.states.logic.distances.distance_angular import (
    AngularDistanceConfig,
)

from src.states.logic.evaluations.evaluation_threshold import ThresholdEvaluationConfig
from src.states.logic.value_handler.normalizers.rotation_normalizer import (
    QuaternionNormalizerConfig,
)
from src.states.state import StateConfig


@dataclass
class RotationStateConfig(StateConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="Quat",
        dim_input=4,
    )
    normalizer: QuaternionNormalizerConfig = QuaternionNormalizerConfig()
    distance: AngularDistanceConfig = AngularDistanceConfig()
    evaluator: ThresholdEvaluationConfig = ThresholdEvaluationConfig(
        distance=AngularDistanceConfig(),
    )
    condition: ConditionConfig = ConditionConfig(
        distance=AngularDistanceConfig(),
        preprocessor=QuaternionPreprocessorConfig(),
    )
