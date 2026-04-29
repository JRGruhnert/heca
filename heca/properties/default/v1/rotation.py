from dataclasses import dataclass

from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.encoders.v1.rotation import QuaternionEncoder
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.evaluators.threshold import ThresholdEvaluator
from heca.properties.extractors.gt import CGTExtractor
from heca.properties.extractors.extractor import PropertyExtractor
from heca.properties.normalizers.normalizer import PropertyNormalizer
from heca.properties.normalizers.quaternion import QuaternionNormalizer
from heca.properties.rulers.angular import AngularRuler
from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.v1 import PropertyV1


class RotationProperty(PropertyV1):
    @dataclass(kw_only=True)
    class Config(PropertyV1.Config):
        ruler: PropertyRuler.Config = AngularRuler.Config()
        encoder: PropertyEncoder.Config = QuaternionEncoder.Config(
            query=QuaternionEncoder.Query(),
        )
        evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
        normalizer: PropertyNormalizer.Config = QuaternionNormalizer.Config()
        extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="Quat")
