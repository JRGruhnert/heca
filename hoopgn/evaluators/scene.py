from abc import abstractmethod
from dataclasses import dataclass
from hoopgn.agents.agent import AgentFeedback
from hoopgn.evaluators.evaluator import Evaluator
from hoopgn.properties.property import Property
from hoopgn.misc.td import TDScene


class SceneEvaluator(Evaluator):
    @dataclass(kw_only=True)
    class Config(Evaluator.Config):
        success_reward: float

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # State
        self.progress: float = 0.0
        self.max: float = 0.0
        self.y: TDScene | None = None

    def reset(self, x: TDScene, y: TDScene):
        self.progress = 0.0
        self.max = x.v1_length
        self.y = y

    def v1_equal(self, x: TDScene) -> bool:
        assert self.y is not None
        assert x.v1 == self.y.v1
        for k in x.keys():
            if x[k] != self.y[k]:
                return False
        return True
    
    def percentage(self, x: TDScene, y: TDScene) -> float:
        assert self.y is not None, "Goal must be set before calling is_equal"
        Property.Query(parent=label="dummy")
        results: list[bool] = []
        for p in self.properties:
            l = p.cfg.label
            result = p.evaluate(x[l], self.y[l])
            results.append(result)
        dones = results.count(True)
        self.progress = dones / self.max
        return all(results)

    def is_sample(self, x: TDScene, y: TDScene) -> bool:
        self.reset(x, y)
        return not self.percentage(x)

    @abstractmethod
    def step(self, x: TDScene, feedback: AgentFeedback) -> tuple[float, bool]:
        raise NotImplementedError()

    def get_feedback(self, x: TDScene) -> AgentFeedback:
        raise NotImplementedError()
