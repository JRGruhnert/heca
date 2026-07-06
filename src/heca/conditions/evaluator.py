from dataclasses import dataclass
from heca.agents.agent import AgentFeedback
from heca.conditions.pair import ConditionPair
from heca.misc.base import Configurable
from heca.misc.dc import DCScene


class Evaluator(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
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
        self.conditions: list[ConditionPair] = []

    def reset(self, y: DCScene):
        self.y = y
        self.highscore = 0.0
        self.progress = 0.0
        self.current_step = 0

    def step(self, x: DCScene) -> AgentFeedback:
        reward = 0.0 + self.cfg.step_penalty
        alltime = self.highscore
        last = self.progress
        self.progress = self.distance(x, self.y)
        self.highscore = max(self.highscore, self.progress)
        done = self.evaluate(x, self.y)

        if done:
            reward += self.cfg.success_reward
            return AgentFeedback(reward=reward, done=done, terminal=True)

        if self.current_step >= self.cfg.max_steps:
            return AgentFeedback(reward=reward, done=done, terminal=True)

        improvement = self.progress - last
        reward += max(0.0, improvement * self.cfg.max_progress_reward)
        if self.cfg.add_monotonic_reward:
            if self.highscore > alltime:
                reward += (self.highscore - alltime) * self.cfg.max_progress_reward

        self.current_step += 1
        return AgentFeedback(reward=reward, done=done, terminal=False)

    def setup(self, conditions: list[ConditionPair]) -> "Evaluator":
        self.conditions = conditions
        return self

    def distance(self, x: DCScene, y: DCScene) -> float:
        return 0.0

    def evaluate(self, x: DCScene, y: DCScene) -> bool:
        return False

    def task_score(self, x: DCScene, y: DCScene) -> float:
        # filter out ones that are already solved
        #
        return 0.0
