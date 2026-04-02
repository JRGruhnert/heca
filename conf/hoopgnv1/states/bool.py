from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.states.addons.prepro_scalar import ScalarStatePreprocessorConfig
from src.states.evaluators.evaluation import StateEvaluatorConfig
from src.states.evaluators.evaluation_threshold import ThresholdEvaluationConfig
from src.states.rulers.binary_ruler import BinaryRulerConfig
from src.states.rulers.ruler import RulerConfig
from src.states.value_handler.normalizers.boundary_normalizer import (
    BoolBoundaryConfig,
)
from src.states.logic.condition import ConditionConfig
from src.states.value_handler.normalizers.ignore_normalizer import (
    IgnoreValueConfig,
)
from src.states.value_handler.normalizers.normalizer import NormalizerConfig
from src.states.logic.threshold_boundary import BoundaryThresholdConfig
from src.states.state import StateConfig


@dataclass
class BoolStateConfig(StateConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="Bool",
        dim_input=1,
        middle_dim=8,
    )
    normalizer: NormalizerConfig = IgnoreValueConfig()
    ruler: RulerConfig = BinaryRulerConfig()
    evaluator: StateEvaluatorConfig = ThresholdEvaluationConfig(
        ruler=BinaryRulerConfig(),
    )
    condition: ConditionConfig = ConditionConfig(
        ruler=BinaryRulerConfig(),
        preprocessor=ScalarStatePreprocessorConfig(
            threshold=BoundaryThresholdConfig(
                boundary=BoolBoundaryConfig(),
            )
        ),
    )
