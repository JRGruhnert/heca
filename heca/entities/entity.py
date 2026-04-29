from dataclasses import dataclass
from typing import Callable

import torch
from heca.classes.merge import Mergeable
from heca.properties.property import Property
from heca.environments.scene import Environment
from heca.misc.td import TDEntity, TDWorld


class Entity(Mergeable):
    @dataclass(kw_only=True)
    class Config(Mergeable.Config):
        props: set[Property.Config]
        weights: list[float] | None = None
        env: Environment.Query | None = None

        def __post_init__(self):
            if self.weights is not None and len(self.weights) != len(self.props):
                raise ValueError("Weights must be provided for all properties")
            assert len(self.props) > 0, "At least one property must be provided"
            assert sum(self.weights) == 1.0 if self.weights is not None else True

    def __init__(self, cfg: Config):
        self.cfg = cfg

        weights = self.cfg.weights or [1.0 / len(self.cfg.props)] * len(self.cfg.props)
        self.properties: dict[str, tuple[Property, float]] = {
            v.label: (Property.create(v), w) for v, w in zip(self.cfg.props, weights)
        }

    @staticmethod
    def ensure_same_meta(items: list["Entity.Query"]) -> str:
        meta = set(i.meta for i in items)
        if len(meta) > 1:
            raise ValueError("All entities must have the same meta")
        return meta.pop()

    @classmethod
    def merge(cls, query: "Entity.Query", items: list["Entity.Query"]) -> "Entity":
        assert len(items) > 0
        cls.ensure_same_meta(items)
        props = cls.make_properties(items)
        meta = f"{heca}.{old}"
        # TODO how to mege? with query?
        return Entity(cfg=Entity.Config(props=props))

    @staticmethod
    def make_properties(items: list["Entity.Query"]) -> set[Property.Config]:
        prop_dict: dict[str, list[Property.Config]] = {}
        # TODO this is a bit hacky, we should have a better way to cluster properties
        for item in items:
            for prop in item.props:
                if prop.label not in prop_dict:
                    prop_dict[prop.label] = []
                prop_dict[prop.label].append(prop)
        merged_props = set()
        for label, configs in prop_dict.items():
            merged_prop = Property.extract(label, configs)
            merged_props.add(merged_prop)
        return merged_props

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

    def _extract_op(self, prop: Property, x, y, values) -> torch.Tensor:
        return prop.extract(x, y, values)

    def _read_op(self, prop: Property, x) -> torch.Tensor:
        ex = prop.extractor(x)
        return prop.normalizer(ex)

    def read(self, x: torch.Tensor) -> torch.Tensor:
        values = self._weighted(x, x, self._read_op)
        return all(values) if values else False

    def distance(self, a: TDEntity, b: TDEntity) -> float:
        values = self._weighted(a, b, self._distance_op)
        return sum(values) / len(values) if values else 0.0

    def evaluate(self, a: TDEntity, b: TDEntity) -> bool:
        values = self._weighted(a, b, self._evaluate_op)
        return all(values) if values else False

    def extract(
        self, x: torch.Tensor, y: torch.Tensor, values: list[torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError()
