from dataclasses import dataclass

from hoopgn.entities.properties.encoders.encoder import PropertyEncoder
from hoopgn.entities.properties.encoders.v2.state import StateEncoderConfig
from hoopgn.entities.properties.evaluators.domain import DomainEvaluator
from hoopgn.entities.properties.evaluators.evaluator import PropertyEvaluator
from hoopgn.entities.properties.extractors.c_gt import CGTExtractor
from hoopgn.entities.properties.extractors.extractor import PropertyExtractor
from hoopgn.entities.properties.normalizers.domain import DomainNormalizer
from hoopgn.entities.properties.normalizers.normalizer import PropertyNormalizer
from hoopgn.entities.properties.parameters.domain import DomainParameter
from hoopgn.entities.properties.parameters.parameter import PropertyParameter
from hoopgn.entities.properties.rulers.domain import DomainRuler
from hoopgn.entities.properties.rulers.ruler import PropertyRuler
from hoopgn.entities.properties.property import Property


@dataclass(kw_only=True)
class StateConfig(Property.Config):
    ruler: PropertyRuler.Config = DomainRuler.Config()
    encoder: PropertyEncoder.Config = StateEncoderConfig()
    normalizer: PropertyNormalizer.Config = DomainNormalizer.Config()
    evaluator: PropertyEvaluator.Config = DomainEvaluator.Config()
    parameter: PropertyParameter.Config = DomainParameter.Config()
    extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="State")
