from dataclasses import dataclass

from heca.misc.classes import ConfigClass
from heca.properties.property import Property


class Entity(ConfigClass):
    @dataclass(kw_only=True)
    class Config(ConfigClass.Config):
        label: str
        meta: str = "root"
        version: str
        properties: set[Property.Config]
        weights: list[float] | None = None

        def __post_init__(self):
            if self.weights is not None and len(self.weights) != len(self.properties):
                raise ValueError("Weights must be provided for all properties")

            assert (
                sum(self.weights) == 1.0 if self.weights is not None else True
            ), "Weights must sum to 1.0"

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Ensure weights are set correctly
        weights = self.cfg.weights or [1.0 / len(self.cfg.properties)] * len(
            self.cfg.properties
        )
        self.properties: dict[str, tuple[Property, float]] = {
            v.label: (Property.from_config(v), w)
            for v, w in zip(self.cfg.properties, weights)
        }
