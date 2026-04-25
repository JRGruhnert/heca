from abc import abstractmethod
from dataclasses import dataclass
from hoopgn.agents.agent import AgentFeedback
from hoopgn.entities.properties.property import Property
from hoopgn.misc.classes import ConfigurableClass
from hoopgn.misc.td import TDScene


class Evaluator(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
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

    def is_equal(self, x: TDScene) -> bool:
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
        return not self.is_equal(x)

    @abstractmethod
    def step(self, x: TDScene, feedback: AgentFeedback) -> tuple[float, bool]:
        raise NotImplementedError()

    def get_feedback(self, x: TDScene) -> AgentFeedback:
        raise NotImplementedError()
