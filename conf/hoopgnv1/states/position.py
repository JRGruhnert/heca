from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.states.logic.addons.prepro_euclidean import EuclideanStatePreprocessorConfig
from src.states.logic.value_handler.normalizers.boundary_normalizer import (
    AreaBoundaryConfig,
    BoundaryNormalizerConfig,
)
from src.states.logic.condition import ConditionConfig
from src.states.logic.distances.distance_euclidean import (
    EuclideanDistanceConfig,
)

from src.states.logic.evaluations.evaluation_threshold import ThresholdEvaluationConfig
from src.states.logic.value_handler.normalizers.normalizer import NormalizerConfig
from src.states.state import ObjectConfig


@dataclass
class PositionStateConfig(ObjectConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="EulerPrecise",
        dim_input=3,
    )
    distance: EuclideanDistanceConfig = EuclideanDistanceConfig()
    evaluator: ThresholdEvaluationConfig = ThresholdEvaluationConfig(
        distance=EuclideanDistanceConfig(),
    )
    normalizer: NormalizerConfig = AreaBoundaryConfig()
    condition: ConditionConfig = ConditionConfig(
        distance=EuclideanDistanceConfig(),
        preprocessor=EuclideanStatePreprocessorConfig(),
    )
