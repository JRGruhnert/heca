from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.states.logic.addons.prepro_euclidean import EuclideanStatePreprocessorConfig
from src.states.logic.value_handler.normalizers.boundary_normalizer import (
    AreaBoundaryConfig,
    BoundaryNormalizerConfig,
)
from src.states.logic.condition import ConditionConfig
from src.states.logic.distances.distance_euclidean import (
    EuclideanRulerConfig,
)

from src.states.logic.evaluations.evaluation_threshold import ThresholdEvaluationConfig
from src.states.logic.value_handler.normalizers.normalizer import NormalizerConfig
from src.states.state import StateConfig


@dataclass
class PositionStateConfig(StateConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="EulerPrecise",
        dim_input=3,
    )
    distance: EuclideanRulerConfig = EuclideanRulerConfig()
    evaluator: ThresholdEvaluationConfig = ThresholdEvaluationConfig(
        distance=EuclideanRulerConfig(),
    )
    normalizer: NormalizerConfig = AreaBoundaryConfig()
    condition: ConditionConfig = ConditionConfig(
        distance=EuclideanRulerConfig(),
        preprocessor=EuclideanStatePreprocessorConfig(),
    )
