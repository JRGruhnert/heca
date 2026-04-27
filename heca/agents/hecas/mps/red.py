from dataclasses import dataclass

from heca.agents.agent import Agent
from heca.agents.hecas.mps.mp import MPAgent


class RedMPAgent(MPAgent):
    @dataclass(kw_only=True)
    class Config(MPAgent.Config):
        agents: set[Agent.Query] = set()
