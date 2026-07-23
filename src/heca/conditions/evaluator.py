from dataclasses import dataclass

from heca.conditions.pair import ConPair
from heca.misc.base import Configurable
from heca.misc.data import DCScene
from heca.misc.entity import Entity


@dataclass(kw_only=True, slots=True)
class AgentFeedback:
    terminal: bool
    reward: float
    truncated: bool


class Evaluator(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        success_reward: float = 1.0
        step: float = -0.01

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.current_step: int = 0
        self.max_steps: int = 1
        self.conditions: list[ConPair] = []
        self.entities: dict[str, Entity] = {}
        self.elabels: set[str] = set()

    def reset(self, y: DCScene):
        self.y = y
        self.current_step = 0

    def step(self, x: DCScene, lfb: AgentFeedback) -> AgentFeedback:
        if lfb.terminal and lfb.reward < 0:
            success = lfb.terminal
        else:
            success = self.evaluate(x, self.y)
        reward = self.cfg.step + self.cfg.success_reward * int(success)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return AgentFeedback(reward=reward, terminal=success, truncated=truncated)

    def evaluate(self, x: DCScene, y: DCScene) -> bool:
        for key in self.elabels:
            e = self.entities[key]
            if not e.evaluate(x.get(key), y.get(key)):
                return False
        return True

    def valid_task(self, x: DCScene, y: DCScene) -> bool:
        for pair in self.conditions:
            pair_matches = True
            for key in pair.pre.elabels:
                score, valid = pair.pre.score_single(
                    x.get(key).value,
                    self.entities[key],
                    key,
                )
                pair_matches = pair_matches and valid
            for key in pair.post.elabels:
                score, valid = pair.post.score_single(
                    y.get(key).value,
                    self.entities[key],
                    key,
                )
                pair_matches = pair_matches and valid
            if pair_matches:
                return True
        return False

    def setup(
        self,
        conditions: list[ConPair],
        entities: dict[str, Entity],
        elabels: set[str],
    ) -> "Evaluator":
        self.conditions = conditions
        self.entities = entities
        self.elabels = elabels
        return self

    def set_max_steps(self, steps: int):
        self.max_steps = steps
