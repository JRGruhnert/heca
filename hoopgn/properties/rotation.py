from dataclasses import dataclass

from hoopgn.networks.layers.property_encoder import PropertyEncoder
from hoopgn.properties.features.evaluators.threshold_evaluator import ThresholdEvaluator
from hoopgn.properties.features.extractors.c_gt_extractor import (
    CalvinGTExtractor,
)

from hoopgn.properties.features.normalizers.quaternion_normalizer import (
    QuaternionNormalizer,
)
from hoopgn.properties.features.parameters.quaternion_parameter import (
    QuaternionParameter,
)
from hoopgn.properties.features.rulers.angular_ruler import AngularRuler
from hoopgn.properties.property import Property


@dataclass(kw_only=True)
class QuaternionEncoderConfig(PropertyEncoder.Config):
    sig: PropertyEncoder.Signature = PropertyEncoder.Signature(
        label="Quat",
    )
    dim_input: int = 4


@dataclass
class RotationConfig(Property.Config):
    ruler = AngularRuler.Config()
    encoder = QuaternionEncoderConfig()
    evaluator = ThresholdEvaluator.Config()
    parameter = QuaternionParameter.Config()
    normalizer = QuaternionNormalizer.Config()
    extractor = CalvinGTExtractor.Config(field_name="Quat")
