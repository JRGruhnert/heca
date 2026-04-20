from dataclasses import dataclass

from hoopgn.agents.agent import Agent
from hoopgn.base import RegisterableClass
from hoopgn.environments.environment import Environment

from hoopgn.policies.policy import Policy


class LeafAgent(Agent):
    @dataclass(kw_only=True)
    class Signature(RegisterableClass.Signature):
        label: str
        environment: Environment.Signature

    @dataclass(kw_only=True)
    class Config(Agent.Config):
        signature: "LeafAgent.Signature"
        policy: Policy.Config

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.policy = Policy.from_config(self.cfg.policy)
