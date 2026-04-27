from dataclasses import dataclass

from heca.agents.agent import Agent
from heca.agents.hecas.mps.mp import MPHeca
from heca.entities.real import RealEntity
from heca.environments.calvin import CalvinEnvironment
from heca.properties import base, slide_base, red_base, pink_base, blue_base

RealEntity.Config(
    env=CalvinEnvironment.Query(),
    label="red",
    version="v1",
    props=set(base + red_base),
)


class RedMPHeca(MPHeca):
    @dataclass(kw_only=True)
    class Query(MPHeca.Query):
        label: str = "red"

    @dataclass(kw_only=True)
    class Config(MPHeca.Config):
        agents: set[Agent.Query] = set()
