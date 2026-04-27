from dataclasses import dataclass
from heca.evaluators.evaluator import SceneEvaluator
from heca.misc.td import TDProperties


class SparseEvaluator(SceneEvaluator):
    @dataclass(kw_only=True)
    class Config(SceneEvaluator.Config):
        step_reward: float

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def step(self, x: TDProperties) -> tuple[float, bool]:
        if self.is_equal(x):
            return self.cfg.success_reward, True
        else:
            return self.cfg.step_reward, False
