from abc import abstractmethod
from dataclasses import dataclass
from heca.agents.agent import AgentFeedback
from heca.entities.entity import Entity
from heca.entities.meta import MetaEntity
from heca.evaluators.evaluator import Evaluator
from heca.misc.td import TDWorld


class HecaEvaluator(Evaluator):
    @dataclass(kw_only=True)
    class Config(Evaluator.Config):
        success_reward: float = 25.0
        max_steps: int = 10
        max_progress_reward: float = 1.0
        # Small step penalty to encourage efficiency
        step_penalty: float = -0.002
        add_monotonic_reward: bool = True

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # State
        self.highscore: float = 0.0
        self.progress: float = 0.0
        self.current_step: int = 0

    def reset(self, x: TDWorld, y: TDWorld, e: Entity):
        self.y = y
        self.e = e
        self.highscore = MetaEntity.distance(x, y, e)
        self.progress = 0.0
        self.current_step = 0

    def is_sample(self, x: TDWorld, y: TDWorld, e: Entity) -> bool:
        self.reset(x, y, e)
        return not MetaEntity.evaluate(x, y, e)

    @abstractmethod
    def step(self, x: TDWorld, fb: AgentFeedback) -> tuple[float, bool, bool]:
        reward = 0.0 + self.cfg.step_penalty
        alltime = self.highscore
        last = self.progress
        self.progress = MetaEntity.distance(x, self.y, self.e)
        self.highscore = max(self.highscore, self.progress)
        done = MetaEntity.evaluate(x, self.y, self.e)

        if done:
            reward += self.cfg.success_reward
            return reward, done, True

        if self.current_step >= self.cfg.max_steps:
            return reward, done, True

        improvement = self.progress - last
        reward += max(0.0, improvement * self.cfg.max_progress_reward)
        if self.cfg.add_monotonic_reward:
            if self.highscore > alltime:
                reward += (self.highscore - alltime) * self.cfg.max_progress_reward

        self.current_step += 1
        return reward, done, False
