from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.encoders.v1.bool import BoolEncoder
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.evaluators.threshold import ThresholdEvaluator
from heca.properties.normalizers.ignore import IgnoreNormalizer
from heca.properties.normalizers.normalizer import PropertyNormalizer
from heca.properties.rulers.binary import BinaryRuler
from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.v1 import PropertyV1


class BoolProperty(PropertyV1):
    @dataclass(kw_only=True)
    class Config(PropertyV1.Config):
        ruler: PropertyRuler.Config = BinaryRuler.Config()
        encoder: PropertyEncoder.Config = BoolEncoder.Config(
            query=BoolEncoder.Query(),
        )
        normalizer: PropertyNormalizer.Config = IgnoreNormalizer.Config()
        evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
