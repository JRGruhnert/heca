from dataclasses import dataclass

from hoopgn.networks.layers.encoder import StateEncoderConfig
from hoopgn.objects.properties.features.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from hoopgn.objects.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from hoopgn.objects.properties.features.parameters.binary_parameter import (
    BinaryParameterConfig,
)
from hoopgn.objects.properties.features.rulers.binary_ruler import BinaryRulerConfig
from hoopgn.objects.properties.features.rulers.ruler import RulerConfig

from hoopgn.objects.properties.features.normalizers.ignore_normalizer import (
    IgnoreNormalizerConfig,
)
from hoopgn.objects.properties.features.normalizers.normalizer import NormalizerConfig
from hoopgn.objects.properties.property import PropertyConfig
from hoopgn.objects.properties.features.conditions.condition import (
    PropertyConditionConfig,
)


@dataclass
class BoolPropertyConfig(PropertyConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="Bool",
        dim_input=1,
        middle_dim=8,
    )
    normalizer: NormalizerConfig = IgnoreNormalizerConfig()
    ruler: RulerConfig = BinaryRulerConfig()
    evaluator: StateEvaluatorConfig = ThresholdEvaluatorConfig(
        ruler=BinaryRulerConfig(),
    )
    condition: PropertyConditionConfig = PropertyConditionConfig(
        ruler=BinaryRulerConfig(),
        parameter=BinaryParameterConfig(),
    )
