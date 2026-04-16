from abc import ABC, abstractmethod
from dataclasses import dataclass
from hoopgn.environments.entities.entity import Entity
from hoopgn.environments.properties.property import Property
from hoopgn.observation.td_scene import TDScene


@dataclass(kw_only=True)
class EvaluatorConfig:
    success_reward: float


class Evaluator(ABC):
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.properties: list[Property] = []
        self.entities: list[Entity] = []
        self.max = max(len(self.properties), 1)

        # State
        self.progress: float = 0.0
        self.goal: TDScene | None = None

    def reset(self, goal: TDScene):
        self.progress = 0.0
        self.goal = goal

    def is_equal(self, x: TDScene) -> bool:
        assert self.goal is not None, "Goal must be set before calling is_equal"
        results: list[bool] = []
        for p in self.properties:
            l = p.cfg.label
            result = p.evaluate(x[l], self.goal[l])
            results.append(result)
        dones = results.count(True)
        self.progress = dones / self.max
        return all(results)

    def is_valid(self, x: TDScene) -> bool:
        assert self.goal is not None, "Goal must be set before calling is_valid"
        for p in self.properties:
            l = p.cfg.label
            if not p.validate(x[l], self.goal[l]):
                return False
        return True

    def is_sample(self, x: TDScene) -> bool:
        done = self.is_equal(x)
        valid = self.is_valid(x)
        return not done and valid

    @abstractmethod
    def step(self, x: TDScene) -> tuple[float, bool]:
        "Returns the step reward and wether the step is a terminal step, cause some ending condition was met."
        raise NotImplementedError()
