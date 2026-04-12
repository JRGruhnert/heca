from dataclasses import dataclass, field

from hoopgn.networks.layers.encoder import PropertyEncoderConfig

from hoopgn.entities.properties.features.evaluators.area_evaluator import (
    AreaEvaluatorConfig,
)
from hoopgn.entities.properties.features.extractors.calvin_gt_extractor import (
    CalvinGTExtractorConfig,
)

from hoopgn.entities.properties.features.modifiers.modifier import (
    PropertyModifierConfig,
)
from hoopgn.entities.properties.features.modifiers.one_hot_modifier import (
    OneHotModifierConfig,
)
from hoopgn.entities.properties.features.validators.validator import (
    PropertyValidatorConfig,
)
from hoopgn.entities.properties.features.evaluators.evaluator import (
    PropertyEvaluatorConfig,
)
from hoopgn.entities.properties.features.parameters.euclidean_parameter import (
    EuclideanParameterConfig,
)
from hoopgn.entities.properties.features.rulers.euclidean_ruler import (
    EuclideanRulerConfig,
)
from hoopgn.entities.properties.features.normalizers.boundary_normalizer import (
    AreaNormalizerConfig,
)

from hoopgn.entities.properties.features.normalizers.normalizer import (
    PropertyNormalizerConfig,
)
from hoopgn.entities.properties.features.validators.area_validator import (
    AreaValidatorConfig,
)

from hoopgn.entities.properties.property import PropertyConfig
from hoopgn.entities.properties.features.conditions.condition import (
    PropertyConditionConfig,
)
from hoopgn.entities.properties.states.area_state import AreaStateConfig


@dataclass
class CalvinAreaConfig(AreaStateConfig):
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
class AreaPropertyConfig(PropertyConfig):
    encoder: PropertyEncoderConfig = PropertyEncoderConfig(
        label="AreaEuler",
        dim_input=6,
    )
    normalizer: PropertyNormalizerConfig = AreaNormalizerConfig()
    ruler: EuclideanRulerConfig = EuclideanRulerConfig()
    evaluator: PropertyEvaluatorConfig = AreaEvaluatorConfig(
        area=CalvinAreaConfig(),
    )
    condition: PropertyConditionConfig = PropertyConditionConfig(
        ruler=EuclideanRulerConfig(),
        parameter=EuclideanParameterConfig(),
    )
    validator: PropertyValidatorConfig = AreaValidatorConfig(
        area=CalvinAreaConfig(),
    )
    modifier: PropertyModifierConfig = OneHotModifierConfig(
        state=CalvinAreaConfig(),
    )

    def __post_init__(self):
        self.extractor = CalvinGTExtractorConfig(label=self.label)
