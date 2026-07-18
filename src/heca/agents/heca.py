from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Sequence

from heca.agents.experts.tapas import TapasAgent
from heca.conditions.pair import ConPair
from heca.agents.agent import Agent, AgentFeedback
from heca.conditions.evaluator import Evaluator
from heca.graphs.graph import Graph
from heca.learning.learner import Learner
from heca.misc.data import DCScene
from heca.misc.entity import Entity
from heca.scenes.ogbench.scene import OGBenchScene
from heca.scenes.scene import Scene


class Heca(Agent):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        evaluator: Evaluator.Config = Evaluator.Config()
        agents: Sequence[Agent.Config]
        learner: Learner.Config
        label: str = "heca"
        visualize: bool = True
        n_samples: int = 1000
        threshold: float = 0.5
        downstream_virtual: bool = False
        upstream_noise: bool = True
        inference: bool = False
        ee_agent: Agent.Config = TapasAgent.Config(
            tag="move_ee", scene=OGBenchScene.Config()
        )

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.learner = Learner.get(self.cfg.learner).register(self.cfg.tag)
        self.end_flag = False
        if self.cfg.inference:
            self.learner.eval()

        self.evaluator = Evaluator.get(cfg.evaluator).setup(
            self.conditions,
            self.entities,
            len(self.low_cons) * 2,
        )
        self.graph = Graph.generate(list(self.cfg.agents), self.entities)

    def step(self, x: DCScene) -> tuple[DCScene, AgentFeedback]:
        self.graph.set_start(x)
        data = self.graph.export()
        option = self.learner.predict(data, self.cfg.tag)
        a, y = self.graph.select(option)
        x = self.adjust_ee(a, x, y)
        z = Agent.get(a).act(x, y)
        if self.cfg.downstream_virtual:
            z = y  # pretend that downstream perfectly achieved the goal
        fb = self.evaluator.step(z)
        self.end_flag = self.learner.update(
            fb.reward, fb.terminal, fb.truncated, self.cfg.tag
        )
        return z, fb

    def adjust_ee(self, a: Agent.Config, x: DCScene, y: DCScene):
        # Basically adjusts the ee pose for tapas models
        #  to be in a good position for execution
        agent = Agent.get(a)
        if isinstance(agent, TapasAgent):
            y.ee.value = agent.start_ee
            return Agent.get(self.cfg.ee_agent).act(x, y)
        return x

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
        while not self.evaluator.valid_task(x, y):
            print("Starting Episode")
            (x, ix), (y, iy) = scene.sample_task()
        return x, y

    def train(self, cfg: Scene.Config):
        """Train the network with PPO for a given number of episodes."""
        while not self.end_flag:
            x, y = self.sample(cfg)
            z = self.act(x, y)  # runs a full episode to terminal, accumulates PPO data

    @cached_property
    def elabels(self) -> set[str]:
        values = set()
        for cfg in self.cfg.agents:
            for con in Agent.get(cfg).conditions:
                values.update(con.elabels)
        return values

    @cached_property
    def entities(self) -> set[Entity]:
        values = set()
        for cfg in self.cfg.agents:
            values.update(Agent.get(cfg).entities)
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
        print(len(cons))
        return cons

    def _load(self, path: Path):
        pass

    def _save(self, path: Path):
        pass
