from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.objects.properties.value_handler.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from src.objects.properties.value_handler.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from src.objects.properties.value_handler.parameters.euclidean_parameter import (
    EuclideanParameterConfig,
)
from src.objects.properties.value_handler.rulers.euclidean_ruler import (
    EuclideanRulerConfig,
)
from src.objects.properties.value_handler.normalizers.boundary_normalizer import (
    AreaBoundaryConfig,
)
from src.objects.properties.condition import ConditionConfig
from src.objects.properties.value_handler.normalizers.normalizer import NormalizerConfig
from src.objects.properties.property import StateConfig


@dataclass
class PositionStateConfig(StateConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="EulerPrecise",
        dim_input=3,
    )
    ruler: EuclideanRulerConfig = EuclideanRulerConfig()
    evaluator: StateEvaluatorConfig = ThresholdEvaluatorConfig(
        ruler=EuclideanRulerConfig(),
    )
    normalizer: NormalizerConfig = AreaBoundaryConfig()
    condition: ConditionConfig = ConditionConfig(
        ruler=EuclideanRulerConfig(),
        parameter=EuclideanParameterConfig(),
    )
