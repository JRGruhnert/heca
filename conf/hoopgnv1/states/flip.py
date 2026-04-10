from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.objects.properties.handlers.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from src.objects.properties.handlers.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from src.objects.properties.handlers.parameters.flip_parameter import (
    FlipParameterConfig,
)
from src.objects.properties.handlers.rulers.binary_ruler import BinaryRulerConfig
from src.objects.properties.handlers.rulers.flip_ruler import FlipRulerConfig
from src.objects.properties.handlers.rulers.ruler import RulerConfig
from src.objects.properties.handlers.normalizers.boundary_normalizer import (
    BoolNormalizerConfig,
)
from src.objects.properties.property_condition import PropertyConditionConfig
from src.objects.properties.handlers.normalizers.normalizer import NormalizerConfig
from src.objects.properties.property import PropertyConfig


@dataclass
class FlipStateConfig(PropertyConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="Flip",
        dim_input=1,
        middle_dim=8,
    )
    normalizer: NormalizerConfig = BoolNormalizerConfig()
    ruler: RulerConfig = BinaryRulerConfig()
    evaluator: StateEvaluatorConfig = ThresholdEvaluatorConfig(
        ruler=BinaryRulerConfig(),
    )
    condition: PropertyConditionConfig = PropertyConditionConfig(
        ruler=FlipRulerConfig(),
        parameter=FlipParameterConfig(),
    )
