from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.objects.properties.handlers.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from src.objects.properties.handlers.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from src.objects.properties.handlers.parameters.euclidean_parameter import (
    EuclideanParameterConfig,
)
from src.objects.properties.handlers.rulers.euclidean_ruler import (
    EuclideanRulerConfig,
)
from src.objects.properties.handlers.normalizers.boundary_normalizer import (
    AreaNormalizerConfig,
)
from src.objects.properties.property_condition import PropertyConditionConfig
from src.objects.properties.handlers.normalizers.normalizer import NormalizerConfig
from src.objects.properties.property import PropertyConfig


@dataclass
class PositionStateConfig(PropertyConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="EulerPrecise",
        dim_input=3,
    )
    ruler: EuclideanRulerConfig = EuclideanRulerConfig()
    evaluator: StateEvaluatorConfig = ThresholdEvaluatorConfig(
        ruler=EuclideanRulerConfig(),
    )
    normalizer: NormalizerConfig = AreaNormalizerConfig()
    condition: PropertyConditionConfig = PropertyConditionConfig(
        ruler=EuclideanRulerConfig(),
        parameter=EuclideanParameterConfig(),
    )
