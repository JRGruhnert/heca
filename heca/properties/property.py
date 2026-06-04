import torch
from dataclasses import dataclass
from heca.misc.base import Configurable

from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.extractors.extractor import PropertyExtractor
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.normalizers.normalizer import PropertyNormalizer


class PropertyV1(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        label: str
        ruler: PropertyRuler.Config
        encoder: PropertyEncoder.Config
        evaluator: PropertyEvaluator.Config
        extractor: PropertyExtractor.Config
        normalizer: PropertyNormalizer.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ruler = PropertyRuler.create(cfg.ruler)
        self.encoder = PropertyEncoder.create(cfg.encoder)
        self.evaluator = PropertyEvaluator.create(cfg.evaluator)
        self.extractor = PropertyExtractor.create(cfg.extractor)
        self.normalizer = PropertyNormalizer.create(cfg.normalizer)

    def read(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the property value from the given modality."""
        ex = self.extractor(x)
        return self.normalizer(ex)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Measures the distance between two values."""
        return self.ruler(x, y)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """Evaluates whether given values are similar."""
        m = self.ruler(x, y)
        return self.evaluator(x, y, m)
