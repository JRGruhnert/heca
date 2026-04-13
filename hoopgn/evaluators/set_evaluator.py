from dataclasses import dataclass
from hoopgn.evaluators.evaluator import EvaluatorConfig, Evaluator
from hoopgn.observation.td_parameters import TDParameters


@dataclass(kw_only=True)
class SetEvaluatorConfig(EvaluatorConfig):
    set: str = "train"  # "train", "val", or "test"


class SetEvaluator(Evaluator):
    def __init__(self, config: SetEvaluatorConfig):
        super().__init__(config)
        self.config = config

    def step(self, current: TDParameters, goal: TDParameters) -> tuple[float, bool]:
        raise NotImplementedError("This should never be called.")
