from dataclasses import dataclass

from hoopgn.networks.layers.encoder import StateEncoderConfig
from hoopgn.objects.properties.features.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from hoopgn.objects.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from hoopgn.objects.properties.features.parameters.euclidean_parameter import (
    EuclideanParameterConfig,
)
from hoopgn.objects.properties.features.rulers.euclidean_ruler import (
    EuclideanRulerConfig,
)
from hoopgn.objects.properties.features.normalizers.boundary_normalizer import (
    AreaNormalizerConfig,
)
from hoopgn.objects.properties.features.conditions.condition import (
    PropertyConditionConfig,
)
from hoopgn.objects.properties.features.normalizers.normalizer import NormalizerConfig
from hoopgn.objects.properties.property import PropertyConfig


@dataclass
class PositionPropertyConfig(PropertyConfig):
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
