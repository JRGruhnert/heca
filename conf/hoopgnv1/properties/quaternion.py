from dataclasses import dataclass

from hoopgn.networks.layers.encoder import StateEncoderConfig
from hoopgn.objects.properties.features.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from hoopgn.objects.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from hoopgn.objects.properties.features.conditions.condition import (
    PropertyConditionConfig,
)
from hoopgn.objects.properties.features.parameters.quaternion_parameter import (
    QuaternionParameterConfig,
)
from hoopgn.objects.properties.features.rulers.angular_ruler import AngularRulerConfig
from hoopgn.objects.properties.features.normalizers.quaternion_normalizer import (
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
