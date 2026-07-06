from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class TDEntity:
    pos: np.ndarray
    rot: np.ndarray
    ste: np.ndarray


@dataclass
class TDScene:
    entities: Dict[str, TDEntity]
    extras: Dict[str, np.ndarray] | None = None

    def __post_init__(self):
        if self.extras is None:
            self.extras = {}

    def __getitem__(self, key):
        return self.entities[key]
