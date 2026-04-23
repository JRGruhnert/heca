from dataclasses import dataclass
from hoopgn.agents.evaluators.evaluator import Evaluator
from hoopgn.observation.td_properties import TDProperties


class SparseEvaluator(Evaluator):
    @dataclass(kw_only=True)
    class Config(Evaluator.Config):
        step_reward: float

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def step(self, x: TDProperties) -> tuple[float, bool]:
        if self.is_equal(x):
            return self.cfg.success_reward, True
        else:
            return self.cfg.step_reward, False
