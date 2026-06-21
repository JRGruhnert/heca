from dataclasses import dataclass
from typing import Callable

import torch
from heca.classes.merge import Mergeable
from heca.properties.default.v2.position import PositionProperty
from heca.properties.default.v2.state import StateProperty
from heca.properties.default.v2.property import Property
from heca.misc.td import TDEntity, TDWorld


class Entity(Mergeable):
    @dataclass(frozen=True, kw_only=True)
    class Query(Mergeable.Query):
        env: str

    @dataclass(kw_only=True)
    class Config(Mergeable.Config):
        position: PositionProperty.Config = PositionProperty.Config()
        rotation: PositionProperty.Config = PositionProperty.Config()
        state: StateProperty.Config
        weights: list[float]

        def __post_init__(self):
            assert len(self.weights) == len(self.props)
            assert len(self.props) > 0
            assert sum(self.weights) == 1.0

    def __init__(self, cfg: Config):
        self.cfg = cfg

        weights = self.cfg.weights or [1.0 / len(self.cfg.props)] * len(self.cfg.props)
        self.properties: dict[str, tuple[Property, float]] = {
            v.label: (Property.get(v), w) for v, w in zip(self.cfg.props, weights)
        }

    @staticmethod
    def ensure_same_meta(items: list["Entity.Query"]) -> str:
        meta = set(i.meta for i in items)
        if len(meta) > 1:
            raise ValueError("All entities must have the same meta")
        return meta.pop()

    # @classmethod
    # def merge(cls, items: list["Entity.Query"]) -> "Entity":
    #    assert len(items) > 0
    #    cls.ensure_same_meta(items)
    #    props = cls.make_properties(items)
    #    meta = f"{heca}.{old}"
    #    # TODO how to mege? with query?
    #    return Entity(cfg=Entity.Config(props=props))

    def _weighted(self, a: TDWorld, b: TDWorld, op: Callable) -> list[torch.Tensor]:
        values = []
        for k, v in self.properties.items():
            assert k in a and k in b
            d = op(v[0], a[k], b[k])
            values.append(d * v[1])
        return values

    def _distance_op(self, prop: Property, x, y) -> float:
        return prop.distance(x, y)

    def _evaluate_op(self, prop: Property, x, y) -> bool:
        return prop.evaluate(x, y)

    # def _extract_op(self, prop: Property, x, y, values) -> torch.Tensor:
    #    return prop.extract(x, y, values)

    # def distance(self, a: TDEntity, b: TDEntity) -> float:
    #    values = self._weighted(a, b, self._distance_op)
    #    return sum(values) / len(values) if values else 0.0

    def evaluate(self, a: TDEntity, b: TDEntity) -> bool:
        values = self._weighted(a, b, self._evaluate_op)
        return all(values) if values else False

    def extract(
        self, x: torch.Tensor, y: torch.Tensor, values: list[torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def state(self) -> StateProperty:
        for prop, _ in self.properties.values():
            if isinstance(prop, StateProperty):
                return prop
        raise ValueError("Entity does not have a state property.")

    def make_state_one_hot(self, idx: int) -> torch.Tensor:
        return self.state.state.one_hot_from_idx(idx)
