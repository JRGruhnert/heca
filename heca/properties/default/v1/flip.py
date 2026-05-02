from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.encoders.v1.flip import FlipEncoder
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.evaluators.threshold import ThresholdEvaluator
from heca.properties.normalizers.ignore import IgnoreNormalizer
from heca.properties.normalizers.normalizer import PropertyNormalizer
from heca.properties.rulers.flip import FlipRuler
from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.default.v1.property import PropertyV1


class FlipProperty(PropertyV1):
    @dataclass(kw_only=True)
    class Config(PropertyV1.Config):
        ruler: PropertyRuler.Config = FlipRuler.Config()
        encoder: PropertyEncoder.Config = FlipEncoder.Config()
        normalizer: PropertyNormalizer.Config = IgnoreNormalizer.Config()
        evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
