from dataclasses import dataclass

from hoopgn.entities.properties.encoders.encoder import PropertyEncoder
from hoopgn.entities.properties.encoders.v1.flip import FlipEncoderConfig
from hoopgn.entities.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.entities.properties.evaluators.threshold import ThresholdEvaluator

from hoopgn.entities.properties.extractors.c_gt import CGTExtractor
from hoopgn.entities.properties.extractors.extractor import PropertyExtractor
from hoopgn.entities.properties.normalizers.ignore import IgnoreNormalizer
from hoopgn.entities.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.entities.properties.parameters.flip import FlipParameter
from hoopgn.entities.properties.parameters.parameter import PropertyParameter
from hoopgn.entities.properties.rulers.flip import FlipRuler


from hoopgn.entities.properties.rulers.ruler import PropertyRuler
from hoopgn.entities.properties.property import Property


@dataclass(kw_only=True)
class FlipPropertyConfig(Property.Config):
    ruler: PropertyRuler.Config = FlipRuler.Config()
    encoder: PropertyEncoder.Config = FlipEncoderConfig()
    normalizer: PropertyNormalizer.Config = IgnoreNormalizer.Config()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
    parameter: PropertyParameter.Config = FlipParameter.Config()
    extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="Flip")
