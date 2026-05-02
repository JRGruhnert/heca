from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.encoders.v2.rotation import RotationEncoder
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.evaluators.threshold import ThresholdEvaluator
from heca.properties.normalizers.normalizer import PropertyNormalizer
from heca.properties.normalizers.quaternion import QuaternionNormalizer
from heca.properties.rulers.angular import AngularRuler
from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.property import Property


class RotationProperty(Property):
    @dataclass(kw_only=True)
    class Config(Property.Config):
        ruler: PropertyRuler.Config = AngularRuler.Config()
        encoder: PropertyEncoder.Query = RotationEncoder.Query()
        evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
