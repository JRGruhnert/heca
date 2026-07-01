from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Sequence

from heca.conditions.analyzer import ConditionAnalyzer
from heca.conditions.pair import ConditionPair
from heca.agents.agent import Agent, AgentFeedback

from heca.misc.td import TDScene


class HecaMode(Enum):
    TRAIN = "train"
    EXPLAIN = "explain"
    EVAL = "evaluate"


class HecaBase(Agent):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        agents: Sequence[Agent.Config]
        label: str = "heca"
        n_con_samples: int = 1000

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.analyzer = ConditionAnalyzer()

    def predict(self, data, options) -> tuple[Agent.Config, TDScene, TDScene]:
        raise NotImplementedError

    def step(self, data, options) -> tuple[TDScene, AgentFeedback]:
        raise NotImplementedError

    def act(self, x: TDScene, y: TDScene) -> TDScene:
        raise NotImplementedError

    def sample(self) -> tuple[TDScene, TDScene]:
        raise NotImplementedError

    @cached_property
    def conditions(self) -> list[ConditionPair]:
        return self.merge_lower_conditions()

    def merge_lower_conditions(self) -> list[ConditionPair]:
        path = Agent.resolve(self.cfg)
        cons = []
        for cfg in self.cfg.agents:
            cons.extend(Agent.get(cfg).conditions)

        sets = [{i} for i in range(len(cons))]
        while True:
            merged = False
            for i in range(len(cons)):
                for j in range(i + 1, len(cons)):
                    a = cons[i]
                    b = cons[j]
                    if self.should_merge(a, b):
                        a_set = sets[i]
                        b_set = sets[j]
                        new_set = a_set | b_set
                        new_pair = ConditionPair.merge(
                            label=f"{self.cfg.folder}".join(map(str, sorted(new_set))),
                            a=a,
                            b=b,
                            n_samples=self.cfg.n_con_samples,
                            plot_path=path,
                        )
                        cons.pop(j)
                        cons.pop(i)
                        sets.pop(j)
                        sets.pop(i)
                        sets.append(new_set)
                        cons.append(new_pair)

                        merged = True
                        break

                if merged:
                    break

            if not merged:
                break
        return cons

    def should_merge(self, a: ConditionPair, b: ConditionPair) -> bool:
        path = Agent.resolve(self.cfg)
        result = self.analyzer.analyze(a, b, path)
        # for elabel in result.keys():
        #    forward = result[elabel]["forward"]
        #    backward = result[elabel]["backward"]

        return False

    def _load(self, path: Path):
        raise NotImplementedError

    def _save(self, path: Path):
        raise NotImplementedError
