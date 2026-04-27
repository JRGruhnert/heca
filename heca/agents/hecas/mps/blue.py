from heca.entities.real import RealEntity
from heca.environments.calvin import CalvinEnvironment
from heca.properties import base, slide_base, red_base, pink_base, blue_base

RealEntity.Config(
    env=CalvinEnvironment.Query(),
    label="blue",
    version="v1",
    properties=set(base + blue_base),
)
