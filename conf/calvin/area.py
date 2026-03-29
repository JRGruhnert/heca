from dataclasses import dataclass

from src.states.logic.area.area import AreaConfig


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
