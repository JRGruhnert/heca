from dataclasses import dataclass

import torch

from src.factory import (
    select_state_evaluator,
    select_state_ruler,
    select_eval_condition,
    select_state_validator,
    select_value_handler,
)

from src.networks.layers.encoder import StateEncoderConfig
from src.objects.properties.condition import ConditionConfig
from src.objects.properties.value_handler.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from src.objects.properties.value_handler.rulers.ruler import RulerConfig
from src.objects.properties.value_handler.normalizers.ignore_normalizer import (
    IgnoreValueConfig,
)
from src.objects.properties.value_handler.normalizers.normalizer import NormalizerConfig
from src.objects.properties.value_handler.validators.ignore_validator import (
    IgnoreValidatorConfig,
)
from src.objects.properties.value_handler.validators.validator import ValidatorConfig
from src.objects.properties.value_handler.value_handler import ValueHandlerConfig


@dataclass
class StateConfig:
    id: int
    label: str
    ruler: RulerConfig
    condition: ConditionConfig
    evaluator: StateEvaluatorConfig
    encoder: StateEncoderConfig
    normalizer: NormalizerConfig
    validator: ValidatorConfig = IgnoreValidatorConfig()
    preencoder: ValueHandlerConfig = IgnoreValueConfig()


class State:
    def __init__(
        self,
        config: StateConfig,
    ):
        self.config = config
        self.ruler = select_state_ruler(config.ruler)
        self.validator = select_state_validator(config.validator)
        self.evaluator = select_state_evaluator(config.evaluator)
        self.normalizer = select_state_normalizer(config.normalizer)
        self.preencoder = select_value_handler(config.preencoder)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> float:
        xn = self.normalizer(x)
        yn = self.normalizer(y)
        return self.ruler(xn, yn)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        xn = self.normalizer(x)
        yn = self.normalizer(y)
        return self.evaluator(xn, yn)

    def validate(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """Checks if the given value is a valid sample."""
        nx = self.normalizer(x)
        ny = self.normalizer(y)
        return self.validator(nx, ny)

    def pre_encode(self, x: torch.Tensor) -> torch.Tensor:
        nx = self.normalizer(x)
        return self.preencoder(nx)
