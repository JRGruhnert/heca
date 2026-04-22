from dataclasses import dataclass, field


from hoopgn.networks.layers.property_encoder import PropertyEncoder

from hoopgn.properties.features.evaluators.area_evaluator import (
    AreaEvaluator,
)
from hoopgn.properties.features.extractors.c_gt_area_extractor import (
    CGTAreaExtractor,
)

from hoopgn.properties.features.extractors.extractor import PropertyExtractor
from hoopgn.properties.features.parameters.parameter import PropertyParameter
from hoopgn.properties.features.rulers.ruler import PropertyRuler
from hoopgn.properties.features.validators.validator import (
    PropertyValidator,
)
from hoopgn.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
)
from hoopgn.properties.features.parameters.euclidean_parameter import (
    EuclideanParameter,
)
from hoopgn.properties.features.rulers.euclidean_ruler import (
    EuclideanRuler,
)
from hoopgn.properties.features.normalizers.area_normalizer import (
    AreaNormalizer,
)

from hoopgn.properties.features.normalizers.normalizer import (
    PropertyNormalizer,
)
from hoopgn.properties.features.validators.area_validator import (
    AreaValidator,
)

from hoopgn.properties.property import Property
from hoopgn.properties.logic.area import Area


@dataclass(kw_only=True)
class CalvinAreaConfig(Area.Config):
    labels: set[str] = field(
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


@dataclass(kw_only=True)
class AreaEncoderConfig(PropertyEncoder.Config):
    sig: PropertyEncoder.Signature = PropertyEncoder.Signature(
        label="AreaEuler",
    )
    dim_input: int = 6


@dataclass(kw_only=True)
class CalvinAreaPropertyConfig(Property.Config):
    ruler: PropertyRuler.Config = EuclideanRuler.Config()
    encoder: PropertyEncoder.Config = AreaEncoderConfig()
    normalizer: PropertyNormalizer.Config = AreaNormalizer.Config()
    evaluator: PropertyEvaluator.Config = AreaEvaluator.Config(
        area=CalvinAreaConfig(),
    )
    parameter: PropertyParameter.Config = EuclideanParameter.Config()

    validator: PropertyValidator.Config = AreaValidator.Config(
        area=CalvinAreaConfig(),
    )
    extractor: PropertyExtractor.Config = CGTAreaExtractor.Config(
        field_name="AreaEuler"
    )
