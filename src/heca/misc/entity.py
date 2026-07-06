from dataclasses import dataclass
from enum import Enum
from textwrap import dedent
from functools import total_ordering

import numpy as np

from heca.misc.base import Configurable
from heca.misc.dc import DCEntity
from heca.properties.default.v2.position import PositionProperty
from heca.properties.default.v2.state import StateProperty


class Mobility(Enum):
    FREE = "free"  # Can move freely in the scene
    STATIC = "static"  # Has a fixed position and rotation in the scene
    ARTICULATED = "articulated"  # Has a fixed position but can have a variable rotation in the scene


@total_ordering
class Entity(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        label: str
        scene: str
        states: set[str]
        question: str
        answers: set[str]
        mobility: Mobility
        position: PositionProperty.Config = PositionProperty.Config()
        rotation: PositionProperty.Config = PositionProperty.Config()

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.position = PositionProperty.get(cfg.position)
        self.rotation = PositionProperty.get(cfg.rotation)
        self.state = StateProperty.get(
            StateProperty.Config(
                values=cfg.states,
            )
        )

    def stepmix_fmt(self, dce: DCEntity) -> np.ndarray:
        return np.concatenate(
            (dce.pos, dce.rot, dce.ste[:, None]),
            axis=1,
        )

    def dc_format(self, value: np.ndarray) -> DCEntity:
        return DCEntity(
            value[:3],
            value[3:7],
            value[-1],
            self.state.one_hot_from_idx_dc(
                value[-1],
            ),
        )

    def evaluate(self, a: DCEntity, b: DCEntity) -> bool:
        raise NotImplementedError

    def distance(self, a: DCEntity, b: DCEntity) -> float:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.cfg.label == other.cfg.label

    def __hash__(self) -> int:
        return hash(self.cfg.label)

    def __lt__(self, other: "Entity") -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.cfg.label < other.cfg.label
