from dataclasses import dataclass, field
import torch
from hoopgn.properties.property import PropertyConfig
from hoopgn.networks.layers.encoder import PropertyEncoderConfig

from hoopgn.properties.features.extractors import select_property_extractor
from hoopgn.properties.features.extractors.extractor import (
    PropertyExtractorConfig,
)
from hoopgn.properties.features.modifiers import select_property_modifier

from hoopgn.properties.features.conditions.condition import (
    PropertyConditionConfig,
)
from hoopgn.properties.features.evaluators import select_property_evaluator
from hoopgn.properties.features.evaluators.evaluator import (
    PropertyEvaluatorConfig,
)
from hoopgn.properties.features.modifiers.modifier import (
    PropertyModifierConfig,
)
from hoopgn.properties.features.modifiers.skip_modifier import (
    SkipModifierConfig,
)
from hoopgn.properties.features.normalizers import select_property_normalizer
from hoopgn.properties.features.rulers import select_property_ruler
from hoopgn.properties.features.rulers.ruler import PropertyRulerConfig

from hoopgn.properties.features.normalizers.normalizer import (
    PropertyNormalizerConfig,
)
from hoopgn.properties.features.validators import select_property_validator

from hoopgn.properties.features.validators.skip_validator import (
    SkipValidatorConfig,
)
from hoopgn.properties.features.validators.validator import (
    PropertyValidatorConfig,
)


@dataclass(kw_only=True)
class EntityConfig:
    id: int
    label: str
    # Properties
    domain: PropertyConfig
    position: PropertyConfig
    rotation: PropertyConfig
    state: PropertyConfig
    # Features
    ruler: EntityRulerConfig
    encoder: EntityEncoderConfig
    condition: EntityConditionConfig
    evaluator: EntityEvaluatorConfig
    normalizer: EntityNormalizerConfig
    extractor: EntityExtractorConfig
    modifier: EntityModifierConfig
    validator: EntityValidatorConfig


class Entity:
    def __init__(
        self,
        config: EntityConfig,
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
