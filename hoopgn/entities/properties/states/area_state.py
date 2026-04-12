from dataclasses import dataclass

import numpy as np
import torch

from hoopgn import logger
from hoopgn.entities.properties.states.state import StateConfig, State


@dataclass(kw_only=True)
class AreaStateConfig(StateConfig):
    spawn_surfaces: dict[str, list[list[float]]]
    eval_surfaces: dict[str, list[list[float]]]


class AreaState(State):
    def __init__(
        self,
        config: AreaStateConfig,
    ):
        super().__init__(config)
        self.config = config
        self.spawn_surfaces = self._make_surfaces(config.spawn_surfaces)
        self.eval_surfaces = self._make_surfaces(config.eval_surfaces)

    def label(self, x: torch.Tensor) -> str | None:
        return self.check_eval_area(x)

    def check_eval_area(self, x: torch.Tensor) -> str | None:
        """Check if the point x is in any of the defined areas."""
        for name, (min_corner, max_corner) in self.eval_surfaces.items():
            if torch.all(x >= min_corner) and torch.all(x <= max_corner):
                return name
        return None

    def check_spawn_area(self, x: torch.Tensor) -> str | None:
        """Check if the point x is in any of the defined spawn areas."""
        for name, (min_corner, max_corner) in self.spawn_surfaces.items():
            if torch.all(x >= min_corner) and torch.all(x <= max_corner):
                return name
        return None

    def check_area_type_discrepancy(self, value: torch.Tensor) -> bool:
        """Check if the given value is in the same eval and spawn area."""
        eval_area = self.check_eval_area(value)
        spawn_area = self.check_spawn_area(value)
        return eval_area == spawn_area

    def check_area_similarity(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        """Check if both obs and goal are in the same defined area."""
        current_area = self.check_eval_area(current)
        goal_area = self.check_eval_area(goal)
        return current_area == goal_area

    def _make_surfaces(
        self,
        surfaces: dict[str, list[list[float]]],
    ):
        return {k: torch.from_numpy(np.array(v)) for k, v in surfaces.items()}

    def add_surface_padding(
        self,
        surface: list[list[float]],
        padding_percent: float,
    ):
        """Add padding to surface bounds in x and y directions"""
        # surface = np.array(surface)

        # Get bounds
        x_min, y_min, z_min = surface[0]
        x_max, y_max, z_max = surface[1]

        # Calculate padding amounts
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_padding = x_range * padding_percent / 2  # Divide by 2 for each side
        y_padding = y_range * padding_percent / 2

        # Apply padding (keep z unchanged)
        padded_surface = [
            [x_min - x_padding, y_min - y_padding, z_min],
            [x_max + x_padding, y_max + y_padding, z_max],
        ]

        return padded_surface
