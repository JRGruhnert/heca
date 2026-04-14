from dataclasses import dataclass
from hoopgn.evaluators.evaluator import EvaluatorConfig, Evaluator
from hoopgn.observation.td_properties import TDProperties


@dataclass(kw_only=True)
class SparseEvaluatorConfig(EvaluatorConfig):
    step_reward: float


class SparseEvaluator(Evaluator):
    def __init__(self, config: SparseEvaluatorConfig):
        super().__init__(config)
        self.config = config

    def step(
        self,
        current: TDProperties,
        goal: TDProperties,
    ) -> tuple[float, bool]:
        if self.is_equal(current, goal):
            # Success reached
            return self.config.success_reward, True
        else:
            # Success not reached
            return self.config.step_reward, False
