from dataclasses import dataclass

from hoopgn.properties.encoders.encoder import PropertyEncoder
from hoopgn.properties.encoders.v2.rotation import QuaternionEncoderConfig
from hoopgn.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.properties.evaluators.threshold import ThresholdEvaluator
from hoopgn.properties.extractors.c_gt import CGTExtractor
from hoopgn.properties.extractors.extractor import PropertyExtractor
from hoopgn.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.properties.normalizers.quaternion import QuaternionNormalizer
from hoopgn.properties.parameters.parameter import PropertyParameter
from hoopgn.properties.parameters.quaternion import QuaternionParameter
from hoopgn.properties.rulers.angular import AngularRuler
from hoopgn.properties.rulers.ruler import PropertyRuler
from hoopgn.properties.property import Property


@dataclass(kw_only=True)
class RotationConfig(Property.Config):
    ruler: PropertyRuler.Config = AngularRuler.Config()
    encoder: PropertyEncoder.Config = QuaternionEncoderConfig()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
    parameter: PropertyParameter.Config = QuaternionParameter.Config()
    normalizer: PropertyNormalizer.Config = QuaternionNormalizer.Config()
    extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="Quat")
