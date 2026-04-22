from dataclasses import dataclass

from hoopgn.networks.layers.property_encoder import PropertyEncoder
from hoopgn.properties.features.evaluators.domain_evaluator import DomainEvaluator
from hoopgn.properties.features.evaluators.evaluator import PropertyEvaluator
from hoopgn.properties.features.extractors.c_gt_extractor import CGTExtractor
from hoopgn.properties.features.extractors.extractor import PropertyExtractor
from hoopgn.properties.features.normalizers.domain_normalizer import DomainNormalizer
from hoopgn.properties.features.normalizers.normalizer import PropertyNormalizer
from hoopgn.properties.features.parameters.domain_parameter import DomainParameter
from hoopgn.properties.features.parameters.parameter import PropertyParameter
from hoopgn.properties.features.rulers.domain_ruler import DomainRuler
from hoopgn.properties.features.rulers.ruler import PropertyRuler
from hoopgn.properties.property import Property


@dataclass(kw_only=True)
class DomainEncoderConfig(PropertyEncoder.Config):
    sig: PropertyEncoder.Signature = PropertyEncoder.Signature(
        label="Domain",
    )
    dim_input: int = 1
    middle_dim: int = 8


@dataclass(kw_only=True)
class DomainPropertyConfig(Property.Config):
    ruler: PropertyRuler.Config = DomainRuler.Config()
    encoder: PropertyEncoder.Config = DomainEncoderConfig()
    normalizer: PropertyNormalizer.Config = DomainNormalizer.Config()
    evaluator: PropertyEvaluator.Config = DomainEvaluator.Config()
    parameter: PropertyParameter.Config = DomainParameter.Config()
    extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="Domain")
