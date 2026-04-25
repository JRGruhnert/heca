from dataclasses import dataclass

from hoopgn.entities.properties.encoders.encoder import PropertyEncoder
from hoopgn.entities.properties.encoders.v2.rotation import QuaternionEncoderConfig
from hoopgn.entities.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.entities.properties.evaluators.threshold import ThresholdEvaluator
from hoopgn.entities.properties.extractors.c_gt import CGTExtractor
from hoopgn.entities.properties.extractors.extractor import PropertyExtractor
from hoopgn.entities.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.entities.properties.normalizers.quaternion import QuaternionNormalizer
from hoopgn.entities.properties.parameters.parameter import PropertyParameter
from hoopgn.entities.properties.parameters.quaternion import QuaternionParameter
from hoopgn.entities.properties.rulers.angular import AngularRuler
from hoopgn.entities.properties.rulers.ruler import PropertyRuler
from hoopgn.entities.properties.property import Property


@dataclass(kw_only=True)
class RotationConfig(Property.Config):
    ruler: PropertyRuler.Config = AngularRuler.Config()
    encoder: PropertyEncoder.Config = QuaternionEncoderConfig()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
    parameter: PropertyParameter.Config = QuaternionParameter.Config()
    normalizer: PropertyNormalizer.Config = QuaternionNormalizer.Config()
    extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="Quat")
