from dataclasses import dataclass

from hoopgn.networks.layers.encoder import StateEncoderConfig
from hoopgn.objects.properties.handlers.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from hoopgn.objects.properties.handlers.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from hoopgn.objects.properties.handlers.parameters.binary_parameter import (
    BinaryParameterConfig,
)
from hoopgn.objects.properties.handlers.rulers.binary_ruler import BinaryRulerConfig
from hoopgn.objects.properties.handlers.rulers.ruler import RulerConfig

from hoopgn.objects.properties.handlers.normalizers.ignore_normalizer import (
    IgnoreNormalizerConfig,
)
from hoopgn.objects.properties.handlers.normalizers.normalizer import NormalizerConfig
from hoopgn.objects.properties.property import PropertyConfig
from hoopgn.objects.properties.property_condition import PropertyConditionConfig


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
