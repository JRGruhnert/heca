from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.objects.properties.value_handler.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from src.objects.properties.value_handler.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from src.objects.properties.condition import ConditionConfig
from src.objects.properties.value_handler.parameters.quaternion_parameter import (
    QuaternionParameterConfig,
)
from src.objects.properties.value_handler.rulers.angular_ruler import AngularRulerConfig
from src.objects.properties.value_handler.normalizers.quaternion_normalizer import (
    QuaternionNormalizerConfig,
)
from src.objects.properties.property import StateConfig


@dataclass
class QuaternionStateConfig(StateConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="Quat",
        dim_input=4,
    )
    normalizer: QuaternionNormalizerConfig = QuaternionNormalizerConfig()
    ruler: AngularRulerConfig = AngularRulerConfig()
    evaluator: StateEvaluatorConfig = ThresholdEvaluatorConfig(
        ruler=AngularRulerConfig(),
    )
    condition: ConditionConfig = ConditionConfig(
        ruler=AngularRulerConfig(),
        parameter=QuaternionParameterConfig(),
    )
