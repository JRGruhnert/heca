from dataclasses import dataclass

from hoopgn.networks.layers.property_encoder import PropertyEncoder
from hoopgn.properties.features.evaluators.evaluator import PropertyEvaluator
from hoopgn.properties.features.evaluators.threshold_evaluator import ThresholdEvaluator
from hoopgn.properties.features.extractors.c_gt_extractor import (
    CGTExtractor,
)

from hoopgn.properties.features.extractors.extractor import PropertyExtractor
from hoopgn.properties.features.normalizers.normalizer import PropertyNormalizer
from hoopgn.properties.features.normalizers.quaternion_normalizer import (
    QuaternionNormalizer,
)
from hoopgn.properties.features.parameters.parameter import PropertyParameter
from hoopgn.properties.features.parameters.quaternion_parameter import (
    QuaternionParameter,
)
from hoopgn.properties.features.rulers.angular_ruler import AngularRuler
from hoopgn.properties.features.rulers.ruler import PropertyRuler
from hoopgn.properties.property import Property


@dataclass(kw_only=True)
class QuaternionEncoderConfig(PropertyEncoder.Config):
    sig: PropertyEncoder.Signature = PropertyEncoder.Signature(
        label="Quat",
    )
    dim_input: int = 4


@dataclass(kw_only=True)
class RotationConfig(Property.Config):
    ruler: PropertyRuler.Config = AngularRuler.Config()
    encoder: PropertyEncoder.Config = QuaternionEncoderConfig()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config()
    parameter: PropertyParameter.Config = QuaternionParameter.Config()
    normalizer: PropertyNormalizer.Config = QuaternionNormalizer.Config()
    extractor: PropertyExtractor.Config = CGTExtractor.Config(field_name="Quat")
