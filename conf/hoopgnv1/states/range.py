from dataclasses import dataclass, field
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
from src.objects.properties.value_handler.rulers.binary_ruler import BinaryRulerConfig
from src.objects.properties.value_handler.rulers.euclidean_ruler import (
    EuclideanRulerConfig,
)
from src.objects.properties.value_handler.rulers.ruler import RulerConfig
from src.objects.properties.value_handler.normalizers.boundary_normalizer import (
    BoundaryNormalizerConfig,
)
from src.objects.properties.condition import ConditionConfig
from src.objects.properties.value_handler.normalizers.normalizer import NormalizerConfig
from src.objects.properties.property import StateConfig


@dataclass
class RangeStateConfig(StateConfig):
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
    condition: ConditionConfig = field(init=False)

    def __post_init__(self):
        self.evaluator = ThresholdEvaluatorConfig(
            ruler=BinaryRulerConfig(),
        )
        self.normalizer = BoundaryNormalizerConfig(
            lower=[self.low],
            upper=[self.high],
        )
        self.condition = ConditionConfig(
            ruler=EuclideanRulerConfig(),
            parameter=EuclideanParameterConfig(),
        )
