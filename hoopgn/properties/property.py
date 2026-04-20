from dataclasses import dataclass, field
import torch
from hoopgn.base import ConfigurableClass
from hoopgn.properties.features.conditions.condition import (
    PropertyCondition,
)
from hoopgn.networks.layers.property_encoder import PropertyEncoder
from hoopgn.properties.features.extractors.extractor import (
    PropertyExtractor,
)

from hoopgn.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
)
from hoopgn.properties.features.modifiers.modifier import PropertyModifier
from hoopgn.properties.features.modifiers.skip_modifier import (
    DefaultModifier,
)

from hoopgn.properties.features.rulers.ruler import PropertyRuler

from hoopgn.properties.features.normalizers.normalizer import (
    PropertyNormalizer,
)

from hoopgn.properties.features.validators.skip_validator import (
    DefaultValidator,
)
from hoopgn.properties.features.validators.validator import (
    PropertyValidator,
)


class Property(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):

        ruler: PropertyRuler.Config
        condition: PropertyCondition.Config
        encoder: PropertyEncoder.Config
        evaluator: PropertyEvaluator.Config
        normalizer: PropertyNormalizer.Config
        modifier: PropertyModifier.Config = DefaultModifier.Config()
        validator: PropertyValidator.Config = DefaultValidator.Config()
        extractor: PropertyExtractor.Config = field(init=False)

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ruler = PropertyRuler.from_config(cfg.ruler)
        self.modifier = PropertyModifier.from_config(cfg.modifier)
        self.evaluator = PropertyEvaluator.from_config(cfg.evaluator)
        self.validator = PropertyValidator.from_config(cfg.validator)
        self.extractor = PropertyExtractor.from_config(cfg.extractor)
        self.normalizer = PropertyNormalizer.from_config(cfg.normalizer)

    def measure(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Measures the distance between two values."""
        return self.ruler(x, y)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """Evaluates whether given values are similar."""
        return self.evaluator(x, y)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the property value from the given modality."""
        ex = self.extractor(x)
        return self.normalizer(ex)

    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Applies additional modifications to the given value."""
        return self.modifier(x)

    def validate(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """Checks if the given value is a valid sample."""
        return self.validator(x, y)
