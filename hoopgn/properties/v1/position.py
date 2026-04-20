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
from hoopgn.properties.features.parameters.euclidean_parameter import (
    EuclideanParameter,
)
from hoopgn.properties.features.rulers.euclidean_ruler import (
    EuclideanRuler,
)
from hoopgn.properties.features.normalizers.boundary_normalizer import (
    AreaNormalizerConfig,
)
from hoopgn.properties.features.conditions.condition import (
    PropertyCondition,
)
from hoopgn.properties.features.normalizers.normalizer import (
    PropertyNormalizer,
)
from hoopgn.properties.property import Property


@dataclass
class PositionConfig(Property.Config):
    encoder: PropertyEncoder.Config = PropertyEncoder.Config(
        label="EulerPrecise",
        dim_input=3,
    )
    ruler: EuclideanRuler.Config = EuclideanRuler.Config()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config(
        ruler=EuclideanRuler.Config(),
    )
    normalizer: PropertyNormalizer.Config = AreaNormalizerConfig()
    condition: PropertyCondition.Config = PropertyCondition.Config(
        ruler=EuclideanRuler.Config(),
        parameter=EuclideanParameter.Config(),
    )

    def __post_init__(self):
        self.extractor = CalvinGTExtractor.Config(label=self.label)
