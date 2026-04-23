from dataclasses import dataclass

from hoopgn.networks.layers.property_encoder import PropertyEncoder
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
class DomainEncoderConfig(PropertyEncoder.Config):
    sig: PropertyEncoder.Signature = PropertyEncoder.Signature(
        label="Domain",
    )
    dim_input: int = 1
    middle_dim: int = 8


@dataclass(kw_only=True)
class DomainConfig(Property.Config):
    ruler: PropertyRuler.Config = DomainRuler.Config()
    encoder: PropertyEncoder.Config = DomainEncoderConfig()
    normalizer: PropertyNormalizer.Config = DomainNormalizer.Config()
    evaluator: PropertyEvaluator.Config = DomainEvaluator.Config()
    parameter: PropertyParameter.Config = DomainParameter.Config()
    extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="Domain")
