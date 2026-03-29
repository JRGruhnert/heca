from dataclasses import dataclass

from src.states.logic.addons.addon_euler import EulerTapasAddonConfig
from src.states.logic.area.area import AreaConfig

from src.states.logic.addon import AddonConfig
from src.states.logic.area.area_value_cnd import AreaValueConditionConfig
from src.states.logic.boundary import BoundaryConfig
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
    surfaces: list[str] = ["table", "drawer_open", "drawer_closed", "drawer"]
    spawn_surfaces = {
        "table": [[0.0, -0.15, 0.46], [0.30, -0.03, 0.46]],
        "drawer_open": [[0.04, -0.35, 0.38], [0.30, -0.21, 0.38]],
        "drawer_closed": [[0.04, -0.16, 0.38], [0.30, -0.03, 0.38]],
    }

    eval_surfaces = {
        "table": [[-0.02, -0.17, 0.44], [0.32, -0.01, 0.54]],
        "drawer_open": [[0.02, -0.37, 0.34], [0.32, -0.23, 0.44]],
        "drawer_closed": [[0.02, -0.18, 0.34], [0.32, -0.00, 0.44]],
    }


@dataclass
class CalvinAreaStateConfig(StateConfig):
    type_str: str = "AreaEuler"
    size: int = 6
    value_cnd = AreaValueConditionConfig(
        area=CalvinAreaConfig(),
        boundary=BoundaryConfig(),
    )
    distance_cnd_skill = EuclideanDistanceConditionConfig()
    distance_cnd_goal = EuclideanDistanceConditionConfig()
    eval_cnd = EvalConditionConfig()
    value_cnd_eval: ValueConditionConfig | None = None
    addons = {
        "tapas": EulerTapasAddonConfig(),
    }
