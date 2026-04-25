from dataclasses import dataclass

from hoopgn.agents.leafs.leaf import LeafAgent
from hoopgn.classes import SearchableClass
from hoopgn.entities.entity import Entity
from hoopgn.properties.property import Property
from hoopgn.environments.environment import Environment


class Domain(SearchableClass):
    @dataclass(kw_only=True)
    class Signature(SearchableClass.Query):
        label: str

    @dataclass(kw_only=True)
    class Config(SearchableClass.Config):
        environment: Environment.Config
        agents: list[LeafAgent.Config]
        entities: list[Entity.Config]
        properties: list[Property.Config]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        Environment.load(self.cfg.environment)
        LeafAgent.load(self.cfg.agents)
        Entity.load(self.cfg.entities)
        Property.load(self.cfg.properties)
