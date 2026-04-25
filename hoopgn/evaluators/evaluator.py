from abc import abstractmethod
from dataclasses import dataclass
from hoopgn.classes import ConfigurableClass
from hoopgn.entities.entity import Entity
from hoopgn.properties.property import Property
from hoopgn.observation.td_scene import TDScene


class Evaluator(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        success_reward: float

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # State
        self.progress: float = 0.0
        self.max: float = 0.0
        self.goal: TDScene | None = None

    def reset(self, start: TDScene, goal: TDScene):
        self.progress = 0.0
        self.max = len(goal)
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

    def check_sample(self, x: TDScene, y: TDScene) -> bool:
        self.reset(x, y)
        done = self.is_equal(x)
        valid = self.is_valid(x)
        return not done and valid

    @abstractmethod
    def step(self, x: TDScene) -> tuple[float, bool]:
        raise NotImplementedError()
