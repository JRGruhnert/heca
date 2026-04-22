from dataclasses import dataclass
import torch
from hoopgn.base import RegisterableClass

from hoopgn.networks.layers.property_encoder import PropertyEncoder
from hoopgn.properties.features.extractors.extractor import (
    PropertyExtractor,
)

from hoopgn.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
)

from hoopgn.properties.features.parameters.parameter import PropertyParameter
from hoopgn.properties.features.rulers.ruler import PropertyRuler

from hoopgn.properties.features.normalizers.normalizer import (
    PropertyNormalizer,
)


class Property(RegisterableClass):
    @dataclass(kw_only=True)
    class Signature(RegisterableClass.Signature):
        id: int

    @dataclass(kw_only=True)
    class Config(RegisterableClass.Config):
        ruler: PropertyRuler.Config
        encoder: PropertyEncoder.Config
        parameter: PropertyParameter.Config
        evaluator: PropertyEvaluator.Config
        extractor: PropertyExtractor.Config
        normalizer: PropertyNormalizer.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ruler = PropertyRuler.from_config(cfg.ruler)
        self.encoder = PropertyEncoder.from_config(cfg.encoder)
        self.parameter = PropertyParameter.from_config(cfg.parameter)
        self.evaluator = PropertyEvaluator.from_config(cfg.evaluator)
        self.extractor = PropertyExtractor.from_config(cfg.extractor)
        self.normalizer = PropertyNormalizer.from_config(cfg.normalizer)

    def read(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the property value from the given modality."""
        ex = self.extractor(x)
        return self.normalizer(ex)

    def measure_distance(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Measures the distance between two values."""
        return self.ruler(x, y)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """Evaluates whether given values are similar."""
        m = self.ruler(x, y)
        return self.evaluator(x, y, m)

    def extract_condition(
        self, x: torch.Tensor, y: torch.Tensor, by_policy: bool
    ) -> torch.Tensor | None:
        """Extracts the precondition value from the given modality."""
        return self.parameter.hoopgnv1(x, y, by_policy)
