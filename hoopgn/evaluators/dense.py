from dataclasses import dataclass
from hoopgn.evaluators.evaluator import SceneEvaluator
from hoopgn.misc.td import TDProperties


class DenseEvaluator(SceneEvaluator):
    @dataclass(kw_only=True)
    class Config(SceneEvaluator.Config):
        max_progress_reward: float = 1.0
        # Small step penalty to encourage efficiency
        step_penalty: float = -0.002
        add_monotonic_reward: bool = True
        success_reward: float = 25.0

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.max_percentage_done: float = 0.0

    def reset(self, start: TDProperties, goal: TDProperties):
        super().reset(start, goal)
        self.max_percentage_done = 0.0

    def check_sample(self, x: TDProperties, y: TDProperties) -> bool:
        valid = super().check_sample(x, y)
        self.max_percentage_done = max(self.max_percentage_done, self.progress)
        return valid

    def step(self, x: TDProperties) -> tuple[float, bool]:
        prev_percentage_done = self.progress

        if self.is_equal(x):
            return self.cfg.success_reward, True

        improvement = self.progress - prev_percentage_done
        reward = max(0.0, improvement * self.cfg.max_progress_reward)
        if self.cfg.add_monotonic_reward:
            if self.progress > self.max_percentage_done:
                reward += (
                    self.progress - self.max_percentage_done
                ) * self.cfg.max_progress_reward
                self.max_percentage_done = self.progress
        reward += self.cfg.step_penalty
        return reward, False
