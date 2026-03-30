from dataclasses import dataclass, field

from src.states.logic.addons.prepro_euler import EulerTapasAddonConfig
from src.states.logic.area.area import AreaConfig
from src.states.logic.area.area_value_cnd import AreaValueConditionConfig
from src.states.logic.boundary import AreaBoundaryConfig
from src.states.logic.eval_cnd import EvalConditionConfig
from src.states.logic.location.euclidean_distance_cnd import (
    EuclideanDistanceConditionConfig,
)
from src.states.logic.value_cnd import ValueConditionConfig
from src.states.state import StateConfig


@dataclass
class CalvinAreaConfig(AreaConfig):
    # ORIGINAl
    # surfaces = {
    #    "table": [[0.0, -0.15, 0.46], [0.30, -0.03, 0.46]],
    #    "drawer_open": [[0.04, -0.35, 0.38], [0.30, -0.21, 0.38]],
    #    "drawer_closed": [[0.04, -0.16, 0.38], [0.30, -0.03, 0.38]],
    # }
    label: str = "AreaEuler"
    surfaces: list[str] = field(
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
class CalvinAreaStateConfig(StateConfig):
    type_str: str = "AreaEuler"
    size: int = 6
    value_cnd: AreaValueConditionConfig = AreaValueConditionConfig(
        area=CalvinAreaConfig(),
        boundary=AreaBoundaryConfig(),
    )
    distance_cnd_skill: EuclideanDistanceConditionConfig = (
        EuclideanDistanceConditionConfig()
    )
    distance_cnd_goal: EuclideanDistanceConditionConfig = (
        EuclideanDistanceConditionConfig()
    )
    eval_cnd: EvalConditionConfig = EvalConditionConfig()
    value_cnd_eval: ValueConditionConfig | None = None
    addons: dict[str, EulerTapasAddonConfig] = {
        "tapas": EulerTapasAddonConfig(),
    }
