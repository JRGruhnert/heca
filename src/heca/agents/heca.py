from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

from heca.conditions.pair import ConPair
from heca.agents.agent import Agent, AgentFeedback
from heca.conditions.evaluator import Evaluator
from heca.graphs.graph import Graph

from heca.misc.data import DCScene
from heca.misc.entity import Entity
from heca.learning.ppo import PPO
from heca.scenes.scene import Scene


class Heca(Agent):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        evaluator: Evaluator.Config = Evaluator.Config()
        agents: Sequence[Agent.Config]
        ppo: PPO.Config
        label: str = "heca"
        visualize: bool = True
        n_samples: int = 1000
        threshold: float = 0.75

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ppo = PPO.get(self.cfg.ppo)

        self.evaluator = Evaluator.get(cfg.evaluator).setup(
            self.conditions,
            self.entities,
            len(self.low_cons) * 2,
        )
        self.graph = Graph.generate(list(self.cfg.agents), self.entities)

    def step(self, x: DCScene) -> tuple[DCScene, AgentFeedback]:
        self.graph.set_start(x)
        data = self.graph.export()
        option = self.ppo.predict(data, self.cfg.tag)
        a, y = self.graph.select(option)
        z = Agent.get(a).act(x, y)
        fb = self.evaluator.step(z)
        self.ppo.update(fb.reward, fb.terminal, fb.truncated, self.cfg.tag)
        return z, fb

    def act(self, x: DCScene, y: DCScene) -> DCScene:
        self.graph.set_goal(y)
        self.evaluator.reset(y)
        z, fb = self.step(x)
        while not fb.truncated:
            z, fb = self.step(z)
        return z

    def sample(self, cfg: Scene.Config) -> tuple[DCScene, DCScene]:
        scene = Scene.get(cfg)
        (x, ix), (y, iy) = scene.sample_task()
        while not self.evaluator.test_task(x, y):
            (x, ix), (y, iy) = scene.sample_task()
        return x, y

    def train(self, max_episodes: int, cfg: Scene.Config):
        """Train the network with PPO for a given number of episodes."""

        for ep in range(max_episodes):
            x, y = self.sample(cfg)
            z = self.act(x, y)  # runs a full episode to terminal, accumulates PPO data

    @cached_property
    def elabels(self) -> set[str]:
        values = set()
        for cfg in self.cfg.agents:
            for con in Agent.get(cfg).conditions:
                values.union(con.elabels)
        return values

    @cached_property
    def entities(self) -> set[Entity]:
        values = set()
        for cfg in self.cfg.agents:
            values.union(Agent.get(cfg).entities)
        return values

    @cached_property
    def low_cons(self) -> list[ConPair]:
        cons: list[ConPair] = []
        for cfg in self.cfg.agents:
            cons.extend(Agent.get(cfg).conditions)
        return cons

    @cached_property
    def conditions(self) -> list[ConPair]:
        path = Agent.load_dir(self.cfg)
        cons: list[ConPair] = self.low_cons
        sets = [{i} for i in range(len(cons))]
        while True:
            merged = False
            for i in range(len(cons)):
                for j in range(i + 1, len(cons)):
                    a = cons[i]
                    b = cons[j]
                    if a.can_merge(b, path):
                        a_set = sets[i]
                        b_set = sets[j]
                        new_set = a_set | b_set
                        ids = map(str, sorted(new_set))
                        label = f"{self.cfg.tag}_{''.join(ids)}"
                        new_pair = ConPair.merge(
                            label=label,
                            a=a,
                            b=b,
                            n_samples=self.cfg.n_samples,
                            threshold=self.cfg.threshold,
                        )
                        new_pair.plot(path)
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
