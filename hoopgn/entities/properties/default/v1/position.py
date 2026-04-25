from dataclasses import dataclass

from hoopgn.entities.properties.encoders.encoder import PropertyEncoder

from hoopgn.entities.properties.encoders.v1.position import PositionEncoderConfig
from hoopgn.entities.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.entities.properties.evaluators.threshold import ThresholdEvaluator

from hoopgn.entities.properties.extractors.c_gt import CGTExtractor
from hoopgn.entities.properties.extractors.extractor import PropertyExtractor
from hoopgn.entities.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.entities.properties.parameters.euclidean import EuclideanParameter
from hoopgn.entities.properties.parameters.parameter import PropertyParameter
from hoopgn.entities.properties.rulers.euclidean import EuclideanRuler
from hoopgn.entities.properties.normalizers.area import AreaNormalizer

from hoopgn.entities.properties.rulers.ruler import PropertyRuler
from hoopgn.entities.properties.property import Property


@dataclass(kw_only=True)
class PositionConfig(Property.Config):
    ruler: PropertyRuler.Config = EuclideanRuler.Config()
    encoder: PropertyEncoder.Config = PositionEncoderConfig()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
    normalizer: PropertyNormalizer.Config = AreaNormalizer.Config()
    parameter: PropertyParameter.Config = EuclideanParameter.Config()
    extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="EulerPrecise")
