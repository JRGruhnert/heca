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
        # Small step penalty to encourage efficiency
        step_penalty: float = -0.002

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.current_step: int = 0
        self.max_steps: int = 0
        self.conditions: list[ConPair] = []
        self.entities: set[Entity] = set()
        self.elabels: set[str] = set()

    def reset(self, y: DCScene):
        self.y = y
        self.current_step = 0

    def step(self, x: DCScene) -> AgentFeedback:
        success = self.evaluate(x, self.y)
        reward = self.cfg.step_penalty + self.cfg.success_reward * int(success)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return AgentFeedback(reward=reward, terminal=success, truncated=truncated)

    def evaluate(self, x: DCScene, y: DCScene) -> bool:
        for e in self.entities:
            if e.cfg.label in self.elabels:
                if not e.evaluate(x.get(e.cfg.label), y.get(e.cfg.label)):
                    return False
        return True

    def valid_task(self, x: DCScene, y: DCScene) -> bool:
        for pair in self.conditions:
            pair_matches = True
            for label in pair.pre.elabels:
                score, valid = pair.pre.score_single(x.get(label).value, label)
                pair_matches = pair_matches and valid
            for label in pair.post.elabels:
                score, valid = pair.post.score_single(y.get(label).value, label)
                pair_matches = pair_matches and valid
            if pair_matches:
                return True
        return False

    def setup(
        self,
        conditions: list[ConPair],
        entities: set[Entity],
        elabels: set[str],
        max_steps: int,
    ) -> "Evaluator":
        self.conditions = conditions
        self.entities = entities
        self.elabels = elabels
        self.max_steps = max_steps
        return self
