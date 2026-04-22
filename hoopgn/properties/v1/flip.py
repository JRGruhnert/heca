from dataclasses import dataclass

from hoopgn.networks.layers.property_encoder import PropertyEncoder
from hoopgn.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
)
from hoopgn.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluator,
)

from hoopgn.properties.features.extractors.c_gt_extractor import (
    CGTExtractor,
)
from hoopgn.properties.features.extractors.extractor import PropertyExtractor
from hoopgn.properties.features.normalizers.ignore_normalizer import IgnoreNormalizer
from hoopgn.properties.features.normalizers.normalizer import PropertyNormalizer
from hoopgn.properties.features.parameters.flip_parameter import (
    FlipParameter,
)
from hoopgn.properties.features.parameters.parameter import PropertyParameter
from hoopgn.properties.features.rulers.flip_ruler import FlipRuler


from hoopgn.properties.features.rulers.ruler import PropertyRuler
from hoopgn.properties.property import Property


@dataclass(kw_only=True)
class FlipEncoderConfig(PropertyEncoder.Config):
    sig: PropertyEncoder.Signature = PropertyEncoder.Signature(
        label="Flip",
    )
    dim_input: int = 1
    middle_dim: int = 8


@dataclass(kw_only=True)
class FlipPropertyConfig(Property.Config):
    ruler: PropertyRuler.Config = FlipRuler.Config()
    encoder: PropertyEncoder.Config = FlipEncoderConfig()
    normalizer: PropertyNormalizer.Config = IgnoreNormalizer.Config()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
    parameter: PropertyParameter.Config = FlipParameter.Config()
    extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="Flip")
