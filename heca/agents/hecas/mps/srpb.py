from heca.entities.real import RealEntity
from heca.environment.scenes.calvin import CalvinEnvironment
from heca.properties import base, slide_base, red_base, pink_base, blue_base

RealEntity.Config(
    env=CalvinEnvironment.Query(),
    label="srpb",
    version="v1",
    props=set(base + slide_base + red_base + pink_base + blue_base),
)
