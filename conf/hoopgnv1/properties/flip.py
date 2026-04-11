from dataclasses import dataclass

from hoopgn.networks.layers.encoder import StateEncoderConfig
from hoopgn.objects.properties.features.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from hoopgn.objects.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from hoopgn.objects.properties.features.parameters.flip_parameter import (
    FlipParameterConfig,
)
from hoopgn.objects.properties.features.rulers.binary_ruler import BinaryRulerConfig
from hoopgn.objects.properties.features.rulers.flip_ruler import FlipRulerConfig
from hoopgn.objects.properties.features.rulers.ruler import RulerConfig
from hoopgn.objects.properties.features.normalizers.boundary_normalizer import (
    BoolNormalizerConfig,
)
from hoopgn.objects.properties.features.conditions.condition import (
    PropertyConditionConfig,
)
from hoopgn.objects.properties.features.normalizers.normalizer import NormalizerConfig
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
