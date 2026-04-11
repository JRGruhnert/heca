from dataclasses import dataclass, field
import torch
from hoopgn.networks.layers.encoder import PropertyEncoderConfig

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
    encoder: PropertyEncoderConfig
    condition: PropertyConditionConfig
    evaluator: PropertyEvaluatorConfig
    normalizer: PropertyNormalizerConfig
    extractor: PropertyExtractorConfig = field(init=False)
    modifier: PropertyModifierConfig = SkipModifierConfig()
    validator: PropertyValidatorConfig = SkipValidatorConfig()


class Property:
    def __init__(
        self,
        config: PropertyConfig,
    ):
        self.config = config
        self.ruler = select_property_ruler(config.ruler)
        self.modifier = select_property_modifier(config.modifier)
        self.evaluator = select_property_evaluator(config.evaluator)
        self.validator = select_property_validator(config.validator)
        self.extractor = select_property_extractor(config.extractor)
        self.normalizer = select_property_normalizer(config.normalizer)

    def measure(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Measures the distance between two values."""
        return self.ruler(x, y)

    def modify(self, x: torch.Tensor) -> torch.Tensor:
        """Applies additional modifications to the given value."""
        return self.modifier(x)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """Evaluates whether given values are similar."""
        return self.evaluator(x, y)

    def validate(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """Checks if the given value is a valid sample."""
        return self.validator(x, y)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the property value from the given modality."""
        ex = self.extractor(x)
        return self.normalizer(ex)
