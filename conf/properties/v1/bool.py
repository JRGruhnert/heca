from dataclasses import dataclass

from hoopgn.networks.layers.encoder import PropertyEncoder
from hoopgn.environments.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
)
from hoopgn.environments.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluator,
)

from hoopgn.environments.properties.features.extractors.calvin_gt_extractor import (
    CalvinGTExtractor,
)
from hoopgn.environments.properties.features.parameters.binary_parameter import (
    BinaryParameter,
)
from hoopgn.environments.properties.features.rulers.binary_ruler import (
    BinaryRuler,
)
from hoopgn.environments.properties.features.rulers.ruler import PropertyRuler

from hoopgn.environments.properties.features.normalizers.ignore_normalizer import (
    IgnoreNormalizer,
)
from hoopgn.environments.properties.features.normalizers.normalizer import (
    PropertyNormalizer,
)
from hoopgn.environments.properties.property import Property
from hoopgn.environments.properties.features.conditions.condition import (
    PropertyCondition,
)


@dataclass
class BoolPropertyConfig(Property.Config):
    encoder: PropertyEncoder.Config = PropertyEncoder.Config(
        label="Bool",
        dim_input=1,
        middle_dim=8,
    )
    normalizer: PropertyNormalizer.Config = IgnoreNormalizer.Config()
    ruler: PropertyRuler.Config = BinaryRuler.Config()
    evaluator: PropertyEvaluator.Config = ThresholdEvaluator.Config(
        ruler=BinaryRuler.Config(),
    )
    condition: PropertyCondition.Config = PropertyCondition.Config(
        ruler=BinaryRuler.Config(),
        parameter=BinaryParameter.Config(),
    )

    def __post_init__(self):
        self.extractor = CalvinGTExtractor.Config(label=self.label)
