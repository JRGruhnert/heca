from dataclasses import dataclass
from typing import ClassVar, Any

import numpy as np

from heca.data.data import DCScene
from heca.data.entity import Entity


class PrismaticEntity(Entity):
    BLOCKS: ClassVar[tuple[str, ...]] = ("state", "pos", "rot", "extra")
    REFERENCE_NAMES: ClassVar[tuple[str, ...]] = ()

    @dataclass(kw_only=True)
    class Config(Entity.Config):
        type_id: int = 2

    @property
    def measurement(self) -> dict:
        return {
            "pose": {
                "model": "gaussian_diag",
                "n_columns": 3 + self.rot_dim + 1,
                "reg_covar": Entity.REG_COVAR,
            },
            "state": {
                "model": "categorical",
                "n_columns": 1,
            },
        }

    def extra_values(self, label: str, demos: Any) -> tuple[float, float]:
        sca = np.asarray(demos[f"heca_{label}_sca"])
        return float(sca.min()), float(sca.max())

    def reference_positions(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError(
                f"expected a pool of (N, 3) encoded positions, got "
                f"{positions.shape}"
            )
        centred = positions - positions.mean(axis=0)
        # the travel's own axis: the direction the pool spreads out along most
        _, vectors = np.linalg.eigh(centred.T @ centred)
        axis = vectors[:, -1]
        if axis[int(np.argmax(np.abs(axis)))] < 0:
            axis = -axis
        projection = centred @ axis
        if float(projection.max()) - float(projection.min()) <= 0.0:
            raise ValueError(
                "every encoded position is the same point, so the slide has no "
                "travel to derive ends from"
            )
        low_band = positions[projection <= np.quantile(projection, 0.10)]
        high_band = positions[projection >= np.quantile(projection, 0.90)]
        return {
            "min": np.median(low_band, axis=0),
            "max": np.median(high_band, axis=0),
        }

    def extra_part(
        self,
        label: str,
        obs: dict,
        extra_range: tuple[float, float] | None = None,
    ) -> np.ndarray:
        if extra_range is None:
            raise ValueError(
                f"{label}: a slide needs the travel the demos recorded, and the "
                "scene has none for it"
            )
        current = float(np.asarray(obs[f"heca_{label}_sca"]).ravel()[0])
        return np.array([np.clip(self.fraction(current, extra_range), -1.0, 1.0)])

    def fraction(self, current: float, extra_range: tuple[float, float]) -> float:
        lo, hi = extra_range
        return 2.0 * (current - lo) / (hi - lo) - 1.0

    @property
    def extra_sigma(self) -> np.ndarray:
        return np.full(1, self.cfg.sca_sigma)

    def extra_from_references(
        self, label: str, positions: dict[str, np.ndarray]
    ) -> np.ndarray:
        axis = np.asarray(positions["max"], dtype=np.float64) - np.asarray(
            positions["min"], dtype=np.float64
        )
        span = float(np.linalg.norm(axis))
        if span <= 0.0:
            raise ValueError
        current = np.asarray(positions["current"], dtype=np.float64) - np.asarray(
            positions["min"], dtype=np.float64
        )
        fraction = float(np.dot(current, axis)) / span**2
        return np.clip(np.array([2.0 * fraction - 1.0]), -1.0, 1.0)

    def env_state_value(
        self, label: str, x: DCScene, unnormalize_pos=None
    ) -> dict[str, Any]:
        dc = x.get(label)
        pos = unnormalize_pos(dc.pos) if unnormalize_pos is not None else dc.pos
        return {
            f"heca_{label}_pos": pos,
            f"heca_{label}_rot": dc.rot,
            f"heca_{label}_ste": dc.ste,
        }

    def make_agent_key(self, label: str, obs: Any, start: int, end: int) -> str:
        start_val = obs[f"heca_{label}_sca"][start][0]
        target_val = obs[f"heca_{label}_sca"][end][0]
        direction = "a_b" if target_val > start_val else "b_a"
        return f"{label}_{direction}"
