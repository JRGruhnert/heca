from dataclasses import dataclass

from src.states.logic.addons.prepro_scalar import ScalarStatePreprocessorConfig
from src.states.logic.addons.state_preprocessor import StatePreprocessorConfig
from src.states.logic.boundary import (
    BoolBoundaryConfig,
    BoundaryConfig,
    FlipBoundaryConfig,
)
from src.states.logic.values.value_identity import IdentityValueConfig
from src.states.logic.threshold_boundary import BoundaryThresholdConfig
from src.states.logic.evaluations.evaluation_threshold import ThresholdEvaluationConfig
from src.states.logic.distances.distance_binary import ScalarDistanceConfig
from src.states.logic.values.value import ValueConfig
from src.states.state import StateConfig


@dataclass
class BoolStateConfig(StateConfig):
    type_str: str = "Bool"
    size: int = 1
    value_cnd: IdentityValueConfig = IdentityValueConfig()
    distance_cnd_skill: ScalarDistanceConfig = ScalarDistanceConfig()
    distance_cnd_goal: ScalarDistanceConfig = ScalarDistanceConfig()
    eval_cnd: ThresholdEvaluationConfig = ThresholdEvaluationConfig(
        distance=ScalarDistanceConfig(),
    )
    value_cnd_eval: ValueConfig | None = None

    preprocessor_old: StatePreprocessorConfig = ScalarStatePreprocessorConfig(
        threshold=BoundaryThresholdConfig(
            boundary=BoolBoundaryConfig(),
        )
    )
