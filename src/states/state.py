from dataclasses import dataclass

import torch

from src.factory import (
    select_distance,
    select_eval_condition,
    select_value_handler,
)

from src.networks.layers.encoder import StateEncoderConfig
from src.states.logic.condition import ConditionConfig
from src.states.logic.evaluators.evaluation import StateEvaluatorConfig
from src.states.logic.rulers.ruler import RulerConfig
from src.states.logic.value_handler.normalizers.ignore_normalizer import (
    IgnoreValueConfig,
)
from src.states.logic.value_handler.normalizers.normalizer import NormalizerConfig
from src.states.logic.value_handler.value_handler import ValueHandlerConfig


@dataclass
class StateConfig:
    id: int
    label: str
    ruler: RulerConfig
    condition: ConditionConfig
    evaluator: StateEvaluatorConfig
    encoder: StateEncoderConfig
    normalizer: NormalizerConfig
    preencoder: ValueHandlerConfig = IgnoreValueConfig()
    validator: ValueHandlerConfig = IgnoreValueConfig()


class State:
    def __init__(
        self,
        config: StateConfig,
    ):
        self.config = config
        self.ruler = select_distance(config.ruler)
        self.validator = select_value_handler(config.validator)
        self.evaluator = select_eval_condition(config.evaluator)
        self.normalizer = select_value_handler(config.normalizer)
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
