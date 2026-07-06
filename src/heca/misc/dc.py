import numpy as np
from typing import Iterator, Tuple, Dict

import torch


class DCEntity:
    """Holds position, rotation, and state for one entity."""

    def __init__(
        self, pos: np.ndarray, rot: np.ndarray, ste: np.ndarray, soh: np.ndarray
    ):
        self.pos = pos
        self.rot = rot
        self.ste = ste
        self.soh = soh

    @property
    def tpos(self) -> torch.Tensor:
        return torch.Tensor(self.pos)

    @property
    def trot(self) -> torch.Tensor:
        return torch.Tensor(self.rot)

    @property
    def tste(self) -> torch.Tensor:
        return torch.Tensor(self.ste)

    @property
    def tsoh(self) -> torch.Tensor:
        return torch.Tensor(self.tsoh)

    @property
    def stacked(self) -> np.ndarray:
        return np.concatenate(
            (self.pos, self.rot, self.ste[:, None]),
            axis=1,
        )

    @classmethod
    def empty(cls) -> "DCEntity":
        return cls(
            np.zeros(3),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.zeros(1),
            np.zeros(1),
        )


class DCScene:
    """Holds a collection of entities and extra tensors."""

    def __init__(
        self,
        ee: DCEntity,
        entities: Dict[str, DCEntity],
        extras: Dict[str, np.ndarray] = {},
    ):
        self._ee = ee
        self._entities = entities
        self._extras = extras if extras is not None else {}

    def __getitem__(self, key: str) -> DCEntity:
        if key not in self._entities:
            raise KeyError(f"Entity {key} not found in TDScene")
        return self._entities[key]

    @property
    def extras(self) -> Dict[str, np.ndarray]:
        return self._extras

    @property
    def ee(self) -> DCEntity:
        return self._ee

    def entities(self) -> Iterator[Tuple[str, DCEntity]]:
        for key, value in self._entities.items():
            yield key, value

    @classmethod
    def empty(cls) -> "DCScene":
        return cls(DCEntity.empty(), {})
