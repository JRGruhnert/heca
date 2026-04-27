from dataclasses import dataclass

from heca.entities.entity import Entity
from heca.environments.environment import Environment


class RealEntity(Entity):
    @dataclass(kw_only=True)
    class Config(Entity.Config):
        env: Environment.Query
