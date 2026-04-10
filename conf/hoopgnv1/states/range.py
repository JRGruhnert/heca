from dataclasses import dataclass, field
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
from src.objects.properties.handlers.rulers.binary_ruler import BinaryRulerConfig
from src.objects.properties.handlers.rulers.euclidean_ruler import (
    EuclideanRulerConfig,
)
from src.objects.properties.handlers.rulers.ruler import RulerConfig
from src.objects.properties.handlers.normalizers.boundary_normalizer import (
    BoundaryNormalizerConfig,
)
from src.objects.properties.property_condition import PropertyConditionConfig
from src.objects.properties.handlers.normalizers.normalizer import NormalizerConfig
from src.objects.properties.property import PropertyConfig


@dataclass
class RangeStateConfig(PropertyConfig):
    low: float = 0.0
    high: float = 1.0
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="Range",
        dim_input=1,
        middle_dim=8,
    )
    ruler: RulerConfig = BinaryRulerConfig()
    evaluator: StateEvaluatorConfig = field(init=False)
    normalizer: NormalizerConfig = field(init=False)
    condition: PropertyConditionConfig = field(init=False)

    def __post_init__(self):
        self.evaluator = ThresholdEvaluatorConfig(
            ruler=BinaryRulerConfig(),
        )
        self.normalizer = BoundaryNormalizerConfig(
            lower=[self.low],
            upper=[self.high],
        )
        self.condition = PropertyConditionConfig(
            ruler=EuclideanRulerConfig(),
            parameter=EuclideanParameterConfig(),
        )
