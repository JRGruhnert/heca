import torch
from dataclasses import dataclass
from heca.misc.base import Configurable

from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.evaluators.evaluator import PropertyEvaluator


class Property(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        ruler: PropertyRuler.Config
        encoder: PropertyEncoder.Config
        evaluator: PropertyEvaluator.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ruler = PropertyRuler.create(cfg.ruler)
        self.encoder = PropertyEncoder.get(cfg.encoder)
        self.evaluator = PropertyEvaluator.create(cfg.evaluator)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Measures the distance between two values."""
        return self.ruler(x, y)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """Evaluates whether given values are similar."""
        m = self.ruler(x, y)
        return self.evaluator(x, y, m)
