from dataclasses import dataclass

from hoopgn.agents.leaf_agent import LeafAgent
from hoopgn.base import RegisterableClass
from hoopgn.entities.entity import Entity
from hoopgn.environments.environment import Environment
from hoopgn.properties.property import Property


class Domain(RegisterableClass):
    @dataclass(kw_only=True)
    class Signature(RegisterableClass.Signature):
        label: str

    @dataclass(kw_only=True)
    class Config(RegisterableClass.Config):
        environment: Environment.Config
        agents: set[LeafAgent.Signature]
        entities: set[Entity.Config]
        properties: set[Property.Config]

    def __init__(self, cfg: Config):
        self.cfg = cfg
