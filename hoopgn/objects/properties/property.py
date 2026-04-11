from dataclasses import dataclass
import torch
from hoopgn.networks.layers.encoder import StateEncoderConfig

from hoopgn.objects.properties.features.extractors import select_property_extractor
from hoopgn.objects.properties.features.extractors.extractor import (
    PropertyExtractorConfig,
)
from hoopgn.objects.properties.features.modifiers import select_property_modifier

from hoopgn.objects.properties.features.conditions.condition import (
    PropertyConditionConfig,
)
from hoopgn.objects.properties.features.evaluators import select_property_evaluator
from hoopgn.objects.properties.features.evaluators.evaluator import (
    PropertyEvaluatorConfig,
)
from hoopgn.objects.properties.features.modifiers.modifier import PropertyModifierConfig
from hoopgn.objects.properties.features.modifiers.skip_modifier import (
    SkipModifierConfig,
)
from hoopgn.objects.properties.features.normalizers import select_property_normalizer
from hoopgn.objects.properties.features.rulers import select_property_ruler
from hoopgn.objects.properties.features.rulers.ruler import PropertyRulerConfig

from hoopgn.objects.properties.features.normalizers.normalizer import (
    PropertyNormalizerConfig,
)
from hoopgn.objects.properties.features.validators import select_property_validator

from hoopgn.objects.properties.features.validators.skip_validator import (
    SkipValidatorConfig,
)
from hoopgn.objects.properties.features.validators.validator import (
    PropertyValidatorConfig,
)


@dataclass(kw_only=True)
class PropertyConfig:
    id: int
    label: str
    ruler: PropertyRulerConfig
    condition: PropertyConditionConfig
    evaluator: PropertyEvaluatorConfig
    encoder: StateEncoderConfig
    normalizer: PropertyNormalizerConfig
    extractor: PropertyExtractorConfig
    modifier: PropertyModifierConfig = SkipModifierConfig()
    validator: PropertyValidatorConfig = SkipValidatorConfig()


class Property:
    def __init__(
        self,
        config: PropertyConfig,
    ):
        self.config = config
        self.ruler = select_property_ruler(config.ruler)
        self.validator = select_property_validator(config.validator)
        self.evaluator = select_property_evaluator(config.evaluator)
        self.normalizer = select_property_normalizer(config.normalizer)
        self.extractor = select_property_extractor(config.extractor)
        self.modifier = select_property_modifier(config.modifier)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Measures the distance between two values."""
        xn = self.normalizer(x)
        yn = self.normalizer(y)
        return self.ruler(xn, yn)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """Evaluates whether given values are similar."""
        xn = self.normalizer(x)
        yn = self.normalizer(y)
        return self.evaluator(xn, yn)

    def validate(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """Checks if the given value is a valid sample."""
        nx = self.normalizer(x)
        ny = self.normalizer(y)
        return self.validator(nx, ny)

    def modify(self, x: torch.Tensor) -> torch.Tensor:
        """Applies additional modifications to the given value."""
        nx = self.normalizer(x)
        return self.modifier(nx)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the property value from the given modality."""
        ex = self.extractor(x)
        return self.normalizer(ex)
