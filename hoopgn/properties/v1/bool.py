from dataclasses import dataclass

from hoopgn.networks.layers.property_encoder import PropertyEncoder
from hoopgn.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
)
from hoopgn.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluator,
)

from hoopgn.properties.features.extractors.calvin_gt_extractor import (
    CalvinGTExtractor,
)
from hoopgn.properties.features.normalizers.boundary_normalizer import (
    BoolNormalizerConfig,
)
from hoopgn.properties.features.parameters.binary_parameter import (
    BinaryParameter,
)
from hoopgn.properties.features.rulers.binary_ruler import (
    BinaryRuler,
)
from hoopgn.properties.features.rulers.ruler import PropertyRuler

from hoopgn.properties.features.normalizers.ignore_normalizer import (
    IgnoreNormalizer,
)
from hoopgn.properties.features.normalizers.normalizer import (
    PropertyNormalizer,
)
from hoopgn.properties.property import Property
from hoopgn.properties.features.conditions.condition import (
    PropertyCondition,
)


@dataclass
class BoolPropertyConfig(Property.Config):
    encoder: PropertyEncoder.Config = PropertyEncoder.Config(
        label="Bool",
        dim_input=1,
        middle_dim=8,
    )
    normalizer: PropertyNormalizer.Config = BoolNormalizerConfig()
    ruler: PropertyRuler.Config = BinaryRuler.Config()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config(
        ruler=BinaryRuler.Config(),
    )
    condition: PropertyCondition.Config = PropertyCondition.Config(
        ruler=BinaryRuler.Config(),
        parameter=BinaryParameter.Config(),
    )

    def __post_init__(self):
        self.extractor = CalvinGTExtractor.Config(field_name=self.label)
