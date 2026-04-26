from dataclasses import dataclass

from hoopgn.entities.entity import Entity
from hoopgn.environments.environment import Environment
from hoopgn.properties.property import Property


class RealEntity(Entity):
    @dataclass(kw_only=True)
    class Config(Entity.Config):
        env: Environment.Query
        properties: list[Property.Config]
        weights: list[float] | None = None

        def __post_init__(self):
            if self.weights is not None and len(self.weights) != len(self.properties):
                raise ValueError("Weights must be provided for all properties")

            assert (
                sum(self.weights) == 1.0 if self.weights is not None else True
            ), "Weights must sum to 1.0"

            self.meta = f"{self.meta}{self.env}.{self.label}"

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Ensure weights are set correctly
        weights = self.cfg.weights or [1.0 / len(self.cfg.properties)] * len(
            self.cfg.properties
        )
        self.properties: dict[str, tuple[Property, float]] = {
            v.query.label: (Property.from_config(v), w)
            for v, w in zip(self.cfg.properties, weights)
        }
