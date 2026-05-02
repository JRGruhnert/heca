from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.encoders.v2.position import PositionEncoder
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.evaluators.threshold import ThresholdEvaluator
from heca.properties.rulers.euclidean import EuclideanRuler
from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.property import Property


class PositionProperty(Property):
    @dataclass(kw_only=True)
    class Config(Property.Config):
        ruler: PropertyRuler.Config = EuclideanRuler.Config()
        encoder: PropertyEncoder.Query = PositionEncoder.Query()
        evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
