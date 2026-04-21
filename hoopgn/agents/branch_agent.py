from dataclasses import dataclass

from hoopgn.evaluators.evaluator import Evaluator
from hoopgn.observation.td_scene import TDScene
from hoopgn.agents.agent import Agent


class BranchAgent(Agent):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        agents: set[Agent.Signature]
        evaluator: Evaluator.Config

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def sample_task(self) -> tuple[TDScene, TDScene]:
        raise NotImplementedError("TODO: implement branch sampling logic.")
