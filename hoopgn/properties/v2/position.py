from dataclasses import dataclass

from hoopgn.networks.layers.property_encoder import PropertyEncoder

from hoopgn.properties.features.evaluators.evaluator import PropertyEvaluator
from hoopgn.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluator,
)

from hoopgn.properties.features.extractors.c_gt_extractor import (
    CGTExtractor,
)
from hoopgn.properties.features.extractors.extractor import PropertyExtractor
from hoopgn.properties.features.normalizers.normalizer import PropertyNormalizer
from hoopgn.properties.features.parameters.euclidean_parameter import (
    EuclideanParameter,
)
from hoopgn.properties.features.parameters.parameter import PropertyParameter
from hoopgn.properties.features.rulers.euclidean_ruler import (
    EuclideanRuler,
)
from hoopgn.properties.features.normalizers.area_normalizer import (
    AreaNormalizer,
)

from hoopgn.properties.features.rulers.ruler import PropertyRuler
from hoopgn.properties.property import Property


@dataclass(kw_only=True)
class PositionEncoderConfig(PropertyEncoder.Config):
    sig: PropertyEncoder.Signature = PropertyEncoder.Signature(
        label="EulerPrecise",
    )
    dim_input: int = 3


@dataclass(kw_only=True)
class PositionConfig(Property.Config):
    ruler: PropertyRuler.Config = EuclideanRuler.Config()
    encoder: PropertyEncoder.Config = PositionEncoderConfig()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
    normalizer: PropertyNormalizer.Config = AreaNormalizer.Config()
    parameter: PropertyParameter.Config = EuclideanParameter.Config()
    extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="EulerPrecise")
