from dataclasses import dataclass, field
from src.modules.evaluators.evaluator import EvaluatorConfig
from src.networks.layers.encoder import StateEncoderConfig
from src.states.addons.prepro_scalar import ScalarStatePreprocessorConfig
from src.states.logic.distances.distance import RulerConfig
from src.states.logic.evaluations.evaluation import EvaluationConfig
from src.states.value_handler.normalizers.boundary_normalizer import (
    BoundaryNormalizerConfig,
)
from src.states.logic.condition import ConditionConfig
from src.states.logic.distances.distance_euclidean import EuclideanRulerConfig
from src.states.logic.threshold_boundary import BoundaryThresholdConfig
from src.states.logic.threshold_boundary import BoundaryThresholdConfig
from src.states.logic.evaluations.evaluation_threshold import ThresholdEvaluationConfig
from src.states.logic.distances.distance_binary import BinaryRulerConfig
from src.states.value_handler.normalizers.normalizer import NormalizerConfig
from src.states.state import StateConfig


@dataclass
class RangeStateConfig(StateConfig):
    low: float = 0.0
    high: float = 1.0
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="Range",
        dim_input=1,
        middle_dim=8,
    )
    distance: RulerConfig = BinaryRulerConfig()
    evaluator: EvaluationConfig = field(init=False)
    normalizer: NormalizerConfig = field(init=False)
    condition: ConditionConfig = field(init=False)

    def __post_init__(self):
        self.evaluator = ThresholdEvaluationConfig(
            distance=BinaryRulerConfig(),
        )
        self.normalizer = BoundaryNormalizerConfig(
            lower=[self.low],
            upper=[self.high],
        )
        self.condition = ConditionConfig(
            ruler=EuclideanRulerConfig(),
            preprocessor=ScalarStatePreprocessorConfig(
                threshold=BoundaryThresholdConfig(boundary=self.normalizer),
            ),
        )
