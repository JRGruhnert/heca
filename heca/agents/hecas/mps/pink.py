from heca.entities.real import RealEntity
from heca.environments.scenes.calvin import CalvinEnvironment
from heca.properties import base, slide_base, red_base, pink_base, blue_base

RealEntity.Config(
    env=CalvinEnvironment.Query(),
    label="pink",
    version="v1",
    props=set(base + pink_base),
)
