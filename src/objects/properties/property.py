from dataclasses import dataclass
import torch
from src.networks.layers.encoder import StateEncoderConfig
from src.objects.properties.property_condition import PropertyConditionConfig
from src.objects.properties.handlers import select_state_handler
from src.objects.properties.handlers.evaluators import select_state_evaluator
from src.objects.properties.handlers.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from src.objects.properties.handlers.normalizers import select_state_normalizer
from src.objects.properties.handlers.rulers import select_state_ruler
from src.objects.properties.handlers.rulers.ruler import RulerConfig

from src.objects.properties.handlers.normalizers.normalizer import NormalizerConfig
from src.objects.properties.handlers.validators import select_state_validator
from src.objects.properties.handlers.validators.ignore_validator import (
    IgnoreValidatorConfig,
)
from src.objects.properties.handlers.validators.validator import StateValidatorConfig


@dataclass(kw_only=True)
class PropertyConfig:
    id: int
    label: str
    ruler: RulerConfig
    condition: PropertyConditionConfig
    evaluator: StateEvaluatorConfig
    encoder: StateEncoderConfig
    normalizer: NormalizerConfig
    validator: StateValidatorConfig = IgnoreValidatorConfig()
    # encoder: ValueHandlerConfig = IgnoreValueConfig()


class Property:
    def __init__(
        self,
        config: PropertyConfig,
    ):
        self.config = config
        self.ruler = select_state_ruler(config.ruler)
        self.validator = select_state_validator(config.validator)
        self.evaluator = select_state_evaluator(config.evaluator)
        self.normalizer = select_state_normalizer(config.normalizer)
        self.encoder = select_state_handler(config.encoder)

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
        return self.encoder(nx)
