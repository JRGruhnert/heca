from dataclasses import dataclass
from hoopgn.evaluators.evaluator import Evaluator
from hoopgn.observation.td_properties import TDProperties


class DenseEvaluator(Evaluator):
    @dataclass(kw_only=True)
    class Config(Evaluator.Config):
        max_progress_reward: float = 1.0
        # Small step penalty to encourage efficiency
        step_penalty: float = -0.002
        add_monotonic_reward: bool = True
        success_reward: float = 25.0

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.max_percentage_done: float = 0.0

    def reset(self, goal: TDProperties):
        super().reset(goal)
        self.max_percentage_done = 0.0

    def check_sample(self, x: TDProperties) -> bool:
        valid = super().check_sample(x)
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
