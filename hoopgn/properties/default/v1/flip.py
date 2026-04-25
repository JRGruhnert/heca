from dataclasses import dataclass

from hoopgn.properties.encoders.encoder import PropertyEncoder
from hoopgn.properties.encoders.v1.flip import FlipEncoderConfig
from hoopgn.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.properties.evaluators.threshold import ThresholdEvaluator

from hoopgn.properties.extractors.c_gt import CGTExtractor
from hoopgn.properties.extractors.extractor import PropertyExtractor
from hoopgn.properties.normalizers.ignore import IgnoreNormalizer
from hoopgn.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.properties.parameters.flip import FlipParameter
from hoopgn.properties.parameters.parameter import PropertyParameter
from hoopgn.properties.rulers.flip import FlipRuler


from hoopgn.properties.rulers.ruler import PropertyRuler
from hoopgn.properties.property import Property


@dataclass(kw_only=True)
class FlipPropertyConfig(Property.Config):
    ruler: PropertyRuler.Config = FlipRuler.Config()
    encoder: PropertyEncoder.Config = FlipEncoderConfig()
    normalizer: PropertyNormalizer.Config = IgnoreNormalizer.Config()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
    parameter: PropertyParameter.Config = FlipParameter.Config()
    extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="Flip")
