from dataclasses import dataclass

from hoopgn.entities.entity import Entity
from hoopgn.environments.calvin import CalvinEnvironment
from hoopgn.environments.environment import Environment


class EntityV1(Entity):
    @dataclass(kw_only=True)
    class Query(Entity.Query):
        env: Environment.Query = CalvinEnvironment.Query()
        label: str = "v1"

    @dataclass(kw_only=True)
    class Config(Entity.Config):
        pass
        # TODO: but how to parse tensordict now
