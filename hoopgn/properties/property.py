from dataclasses import dataclass, field
import torch
from hoopgn.base import RegisterableClass
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


class Property(RegisterableClass):
    @dataclass(kw_only=True)
    class Config(RegisterableClass.Config):
        ruler: PropertyRuler.Config
        condition: PropertyCondition.Config
        encoder: PropertyEncoder.Config
        evaluator: PropertyEvaluator.Config
        normalizer: PropertyNormalizer.Config
        extractor: PropertyExtractor.Config = field(init=False)

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ruler = PropertyRuler.from_config(cfg.ruler)
        self.evaluator = PropertyEvaluator.from_config(cfg.evaluator)
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
