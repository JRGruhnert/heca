from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.objects.properties.handlers.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from src.objects.properties.handlers.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from src.objects.properties.handlers.parameters.binary_parameter import (
    BinaryParameterConfig,
)
from src.objects.properties.handlers.rulers.binary_ruler import BinaryRulerConfig
from src.objects.properties.handlers.rulers.ruler import RulerConfig

from src.objects.properties.handlers.normalizers.ignore_normalizer import (
    IgnoreNormalizerConfig,
)
from src.objects.properties.handlers.normalizers.normalizer import NormalizerConfig
from src.objects.properties.property import PropertyConfig
from src.objects.properties.property_condition import PropertyConditionConfig


@dataclass
class BoolStateConfig(PropertyConfig):
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
