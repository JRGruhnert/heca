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
from hoopgn.properties.features.parameters.flip_parameter import (
    FlipParameter,
)
from hoopgn.properties.features.rulers.binary_ruler import (
    BinaryRuler,
)
from hoopgn.properties.features.rulers.flip_ruler import FlipRuler
from hoopgn.properties.features.rulers.ruler import PropertyRuler
from hoopgn.properties.features.normalizers.boundary_normalizer import (
    BoolNormalizerConfig,
)
from hoopgn.properties.features.conditions.condition import (
    PropertyCondition,
)
from hoopgn.properties.features.normalizers.normalizer import (
    PropertyNormalizer,
)
from hoopgn.properties.property import Property


@dataclass
class FlipPropertyConfig(Property.Config):
    encoder: PropertyEncoder.Config = PropertyEncoder.Config(
        label="Flip",
        dim_input=1,
        middle_dim=8,
    )
    normalizer: PropertyNormalizer.Config = BoolNormalizerConfig()
    ruler: PropertyRuler.Config = BinaryRuler.Config()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config(
        ruler=BinaryRuler.Config(),
    )
    condition: PropertyCondition.Config = PropertyCondition.Config(
        ruler=FlipRuler.Config(),
        parameter=FlipParameter.Config(),
    )

    def __post_init__(self):
        self.extractor = CalvinGTExtractor.Config(field_name=self.label)
