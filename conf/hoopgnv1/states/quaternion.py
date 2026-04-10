from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.objects.properties.handlers.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from src.objects.properties.handlers.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from src.objects.properties.property_condition import PropertyConditionConfig
from src.objects.properties.handlers.parameters.quaternion_parameter import (
    QuaternionParameterConfig,
)
from src.objects.properties.handlers.rulers.angular_ruler import AngularRulerConfig
from src.objects.properties.handlers.normalizers.quaternion_normalizer import (
    QuaternionNormalizerConfig,
)
from src.objects.properties.property import PropertyConfig


@dataclass
class QuaternionStateConfig(PropertyConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="Quat",
        dim_input=4,
    )
    normalizer: QuaternionNormalizerConfig = QuaternionNormalizerConfig()
    ruler: AngularRulerConfig = AngularRulerConfig()
    evaluator: StateEvaluatorConfig = ThresholdEvaluatorConfig(
        ruler=AngularRulerConfig(),
    )
    condition: PropertyConditionConfig = PropertyConditionConfig(
        ruler=AngularRulerConfig(),
        parameter=QuaternionParameterConfig(),
    )
