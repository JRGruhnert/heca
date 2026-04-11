from dataclasses import dataclass

from hoopgn.networks.layers.encoder import StateEncoderConfig
from hoopgn.objects.properties.handlers.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from hoopgn.objects.properties.handlers.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from hoopgn.objects.properties.property_condition import PropertyConditionConfig
from hoopgn.objects.properties.handlers.parameters.quaternion_parameter import (
    QuaternionParameterConfig,
)
from hoopgn.objects.properties.handlers.rulers.angular_ruler import AngularRulerConfig
from hoopgn.objects.properties.handlers.normalizers.quaternion_normalizer import (
    QuaternionNormalizerConfig,
)
from hoopgn.objects.properties.property import PropertyConfig


@dataclass
class QuaternionPropertyConfig(PropertyConfig):
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
