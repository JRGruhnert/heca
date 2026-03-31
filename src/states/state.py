from dataclasses import dataclass

import torch

from src.factory import (
    select_distance,
    select_eval_condition,
    select_value_handler,
)

from src.networks.layers.encoder import StateEncoderConfig
from src.states.logic.condition import ConditionConfig
from src.states.logic.distances.distance import Distance, ValueDistanceConfig
from src.states.logic.evaluations.evaluation import ValueEvaluationConfig
from src.states.logic.value_handler.normalizers.ignore_normalizer import (
    IgnoreValueConfig,
)
from src.states.logic.value_handler.normalizers.normalizer import NormalizerConfig
from src.states.logic.value_handler.value_handler import ValueHandlerConfig


@dataclass
class StateConfig:
    id: int
    label: str
    condition: ConditionConfig
    dst_skill: ValueDistanceConfig
    dst_goal: ValueDistanceConfig
    evaluator: ValueEvaluationConfig
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
        self.dst_skill = select_distance(config.dst_skill)
        self.dst_goal = select_distance(config.dst_goal)
        self.evaluator = select_eval_condition(config.evaluator)
        self.normalizer = select_value_handler(config.normalizer)
        self.preencoder = select_value_handler(config.preencoder)
        self.validator = select_value_handler(config.validator)

    def pre_encode(self, x: torch.Tensor) -> torch.Tensor:
        nx = self.normalizer(x)
        return self.preencoder(nx)

    def distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        dst: Distance,
    ) -> float:
        xn = self.normalizer(x)
        yn = self.normalizer(y)
        return dst.distance(xn, yn)

    def distance_to_skill(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        return self.distance(x, y, self.dst_skill)

    def distance_to_goal(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        return self.distance(x, y, self.dst_goal)

    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> bool:
        xn = self.validator(x)
        yn = self.validator(y)
        return self.evaluator(xn, yn)
