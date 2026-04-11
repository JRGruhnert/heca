from dataclasses import dataclass, field

from hoopgn.networks.layers.classifiers.state_classifier import StateClassifierConfig
from hoopgn.networks.layers.encoder import StateEncoderConfig
from hoopgn.objects.properties.area import AreaConfig

from hoopgn.objects.properties.features.modifiers.modifier import ModifierConfig
from hoopgn.objects.properties.features.modifiers.one_hot_modifier import (
    OneHotModifierConfig,
)
from hoopgn.objects.properties.features.validators.validator import StateValidatorConfig
from hoopgn.objects.properties.features.evaluators.evaluator import (
    StateEvaluatorConfig,
)
from hoopgn.objects.properties.features.parameters.euclidean_parameter import (
    EuclideanParameterConfig,
)
from hoopgn.objects.properties.features.rulers.euclidean_ruler import (
    EuclideanRulerConfig,
)
from hoopgn.objects.properties.features.normalizers.boundary_normalizer import (
    AreaNormalizerConfig,
)

from hoopgn.objects.properties.features.normalizers.normalizer import NormalizerConfig
from hoopgn.objects.properties.features.validators.area_validator import (
    AreaValidatorConfig,
)

from hoopgn.objects.properties.property import PropertyConfig
from hoopgn.objects.properties.features.conditions.condition import (
    PropertyConditionConfig,
)


@dataclass
class CalvinAreaConfig(AreaConfig):
    label: str = "Area"
    values: list[str] = field(
        default_factory=lambda: ["table", "drawer_open", "drawer_closed", "drawer"]
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
    encoder: StateEncoderConfig = StateEncoderConfig(
        label="AreaEuler",
        dim_input=6,
    )
    normalizer: NormalizerConfig = AreaNormalizerConfig()
    ruler: EuclideanRulerConfig = EuclideanRulerConfig()
    evaluator: StateEvaluatorConfig = StateEvaluatorConfig()
    condition: PropertyConditionConfig = PropertyConditionConfig(
        ruler=EuclideanRulerConfig(),
        parameter=EuclideanParameterConfig(),
    )
    validator: StateValidatorConfig = AreaValidatorConfig(
        area=CalvinAreaConfig(
            classifier=StateClassifierConfig(),
        )
    )
    modifier: ModifierConfig = OneHotModifierConfig(
        state=CalvinAreaConfig(
            classifier=StateClassifierConfig(),
        ),
    )
