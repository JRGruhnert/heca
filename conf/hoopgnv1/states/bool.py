from dataclasses import dataclass

from src.networks.layers.encoder import StateEncoderConfig
from src.objects.properties.value_handler.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from src.objects.properties.value_handler.evaluators.threshold_evaluator import (
    ThresholdEvaluatorConfig,
)
from src.objects.properties.value_handler.parameters.binary_parameter import (
    BinaryParameterConfig,
)
from src.objects.properties.value_handler.rulers.binary_ruler import BinaryRulerConfig
from src.objects.properties.value_handler.rulers.ruler import RulerConfig

from src.objects.properties.condition import ConditionConfig
from src.objects.properties.value_handler.normalizers.ignore_normalizer import (
    IgnoreValueConfig,
)
from src.objects.properties.value_handler.normalizers.normalizer import NormalizerConfig
from src.objects.properties.property import StateConfig


@dataclass
class BoolStateConfig(StateConfig):
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="Bool",
        dim_input=1,
        middle_dim=8,
    )
    normalizer: NormalizerConfig = IgnoreValueConfig()
    ruler: RulerConfig = BinaryRulerConfig()
    evaluator: StateEvaluatorConfig = ThresholdEvaluatorConfig(
        ruler=BinaryRulerConfig(),
    )
    condition: ConditionConfig = ConditionConfig(
        ruler=BinaryRulerConfig(),
        parameter=BinaryParameterConfig(),
    )
