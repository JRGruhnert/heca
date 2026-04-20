from dataclasses import dataclass, field

from hoopgn.networks.layers.encoder import PropertyEncoder

from hoopgn.environments.properties.features.evaluators.area_evaluator import (
    AreaEvaluator,
)
from hoopgn.environments.properties.features.extractors.calvin_gt_extractor import (
    CalvinGTExtractor,
)

from hoopgn.environments.properties.features.modifiers.modifier import (
    PropertyModifier,
)
from hoopgn.environments.properties.features.modifiers.one_hot_modifier import (
    OneHotModifier,
)
from hoopgn.environments.properties.features.validators.validator import (
    PropertyValidator,
)
from hoopgn.environments.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
)
from hoopgn.environments.properties.features.parameters.euclidean_parameter import (
    EuclideanParameter,
)
from hoopgn.environments.properties.features.rulers.euclidean_ruler import (
    EuclideanRuler,
)
from hoopgn.environments.properties.features.normalizers.boundary_normalizer import (
    AreaNormalizerConfig,
)

from hoopgn.environments.properties.features.normalizers.normalizer import (
    PropertyNormalizer,
)
from hoopgn.environments.properties.features.validators.area_validator import (
    AreaValidator,
    AreaValidatorConfig,
)

from hoopgn.environments.properties.property import Property
from hoopgn.environments.properties.features.conditions.condition import (
    PropertyCondition,
)
from hoopgn.environments.properties.v1.area_state import AreaState


@dataclass
class CalvinAreaConfig(AreaState.Config):
    label: str = "Area"
    values: set[str] = field(
        default_factory=lambda: {"table", "drawer_open", "drawer_closed"}
    )
    spawn_surfaces: dict = field(
        default_factory=lambda: {
            "table": [[0.0, -0.15, 0.46], [0.30, -0.03, 0.46]],
            "drawer_open": [[0.04, -0.35, 0.38], [0.30, -0.21, 0.38]],
            "drawer_closed": [[0.04, -0.16, 0.38], [0.30, -0.03, 0.38]],
        }
    )
    eval_surfaces: dict = field(
        default_factory=lambda: {
            "table": [[-0.02, -0.17, 0.44], [0.32, -0.01, 0.54]],
            "drawer_open": [[0.02, -0.37, 0.34], [0.32, -0.23, 0.44]],
            "drawer_closed": [[0.02, -0.18, 0.34], [0.32, -0.00, 0.44]],
        }
    )


@dataclass
class AreaPropertyConfig(Property.Config):
    encoder: PropertyEncoder.Config = PropertyEncoder.Config(
        label="AreaEuler",
        dim_input=6,
    )
    normalizer: PropertyNormalizer.Config = AreaNormalizer.Config()
    ruler: EuclideanRuler.Config = EuclideanRuler.Config()
    evaluator: PropertyEvaluator.Config = AreaEvaluator.Config(
        label="AreaEvaluator",
        area=CalvinAreaConfig(),
    )
    condition: PropertyCondition.Config = PropertyCondition.Config(
        ruler=EuclideanRuler.Config(),
        parameter=EuclideanParameter.Config(),
    )
    validator: PropertyValidator.Config = AreaValidator.Config(
        area=CalvinArea.Config(),
    )
    modifier: PropertyModifier.Config = OneHotModifier.Config(
        state=CalvinArea.Config(),
    )

    def __post_init__(self):
        self.extractor = CalvinGTExtractor.Config(label=self.label)
