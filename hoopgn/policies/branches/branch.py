from dataclasses import dataclass

from hoopgn.agents.agent import Agent
from hoopgn.observation.td_scene import TDScene
from hoopgn.policies.policy import Policy


class BranchPolicy(Policy):
    @dataclass(kw_only=True)
    class Config(Policy.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def __call__(
        self, x: TDScene, y: TDScene
    ) -> tuple[Agent.Signature, TDScene, TDScene]:
        raise NotImplementedError()
