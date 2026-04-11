from dataclasses import dataclass

from hoopgn.networks.layers.encoder import StateEncoderConfig
from hoopgn.objects.properties.handlers.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from hoopgn.objects.properties.handlers.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from hoopgn.objects.properties.handlers.parameters.flip_parameter import (
    FlipParameterConfig,
)
from hoopgn.objects.properties.handlers.rulers.binary_ruler import BinaryRulerConfig
from hoopgn.objects.properties.handlers.rulers.flip_ruler import FlipRulerConfig
from hoopgn.objects.properties.handlers.rulers.ruler import RulerConfig
from hoopgn.objects.properties.handlers.normalizers.boundary_normalizer import (
    BoolNormalizerConfig,
)
from hoopgn.objects.properties.property_condition import PropertyConditionConfig
from hoopgn.objects.properties.handlers.normalizers.normalizer import NormalizerConfig
from hoopgn.objects.properties.property import PropertyConfig


@dataclass
class FlipPropertyConfig(PropertyConfig):
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
