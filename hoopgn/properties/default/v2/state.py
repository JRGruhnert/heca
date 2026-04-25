from dataclasses import dataclass

from hoopgn.properties.encoders.encoder import PropertyEncoder
from hoopgn.properties.encoders.v2.state import StateEncoderConfig
from hoopgn.properties.evaluators.domain import DomainEvaluator
from hoopgn.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.properties.extractors.c_gt import CGTExtractor
from hoopgn.properties.extractors.extractor import PropertyExtractor
from hoopgn.properties.normalizers.domain import DomainNormalizer
from hoopgn.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.properties.parameters.domain import DomainParameter
from hoopgn.properties.parameters.parameter import PropertyParameter
from hoopgn.properties.rulers.domain import DomainRuler
from hoopgn.properties.rulers.ruler import PropertyRuler
from hoopgn.properties.property import Property


@dataclass(kw_only=True)
class StateConfig(Property.Config):
    ruler: PropertyRuler.Config = DomainRuler.Config()
    encoder: PropertyEncoder.Config = StateEncoderConfig()
    normalizer: PropertyNormalizer.Config = DomainNormalizer.Config()
    evaluator: PropertyEvaluator.Config = DomainEvaluator.Config()
    parameter: PropertyParameter.Config = DomainParameter.Config()
    extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="State")
