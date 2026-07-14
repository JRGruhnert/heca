from dataclasses import dataclass

from heca.agents.agent import AgentFeedback
from heca.conditions.pair import ConPair
from heca.misc.base import Configurable
from heca.misc.data import DCScene
from heca.misc.entity import Entity


class Evaluator(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        success_reward: float = 25.0
        max_steps: int = 10
        # Small step penalty to encourage efficiency
        step_penalty: float = -0.002
        sample_threshold: float = 0.75

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.current_step: int = 0
        self.conditions: list[ConPair] = []
        self.entities: set[Entity] = set()

    def reset(self, y: DCScene):
        self.y = y
        self.current_step = 0

    def step(self, x: DCScene) -> AgentFeedback:
        reward = 0.0 + self.cfg.step_penalty
        done = self.evaluate(x, self.y)

        if done:
            reward += self.cfg.success_reward
            return AgentFeedback(reward=reward, done=done, terminal=True)

        if self.current_step >= self.cfg.max_steps:
            return AgentFeedback(reward=reward, done=done, terminal=True)

        self.current_step += 1
        return AgentFeedback(reward=reward, done=done, terminal=False)

    def setup(
        self,
        conditions: list[ConPair],
        entities: set[Entity],
    ) -> "Evaluator":
        self.conditions = conditions
        self.entities = entities
        return self

    def evaluate(self, x: DCScene, y: DCScene) -> bool:
        for e in self.entities:
            if not e.evaluate(x.get(e.cfg.label), y.get(e.cfg.label)):
                return False
        return True

    def test_task(self, x: DCScene, y: DCScene) -> bool:
        for pair in self.conditions:
            pair_match = True
            for label in pair.pre.elabels:
                _, valid = pair.pre.score_single(x.get(label), label)
                if not valid:
                    pair_match = valid
            for label in pair.post.elabels:
                _, valid = pair.pre.score_single(x.get(label), label)
                if not valid:
                    pair_match = valid
            if pair_match:
                return True
        return False
