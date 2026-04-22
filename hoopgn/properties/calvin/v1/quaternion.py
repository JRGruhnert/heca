from dataclasses import dataclass

from hoopgn.networks.layers.property_encoder import PropertyEncoder
from hoopgn.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
)
from hoopgn.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluator,
)
from hoopgn.properties.features.conditions.condition import (
    PropertyCondition,
)

from hoopgn.properties.features.extractors.c_gt_extractor import (
    CalvinGTExtractor,
)
from hoopgn.properties.features.parameters.quaternion_parameter import (
    QuaternionParameter,
)
from hoopgn.properties.features.rulers.angular_ruler import (
    AngularRuler,
)
from hoopgn.properties.features.normalizers.quaternion_normalizer import (
    QuaternionNormalizer,
)
from hoopgn.properties.property import Property


@dataclass
class QuaternionPropertyConfig(Property.Config):
    encoder: PropertyEncoder.Config = PropertyEncoder.Config(
        label="Quat",
        dim_input=4,
    )
    normalizer: QuaternionNormalizer.Config = QuaternionNormalizer.Config()
    ruler: AngularRuler.Config = AngularRuler.Config()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config(
        ruler=AngularRuler.Config(),
    )
    condition: PropertyCondition.Config = PropertyCondition.Config(
        ruler=AngularRuler.Config(),
        parameter=QuaternionParameter.Config(),
    )

    def __post_init__(self):
        self.extractor = CalvinGTExtractor.Config(label=self.label)
