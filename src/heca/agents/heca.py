from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Sequence

from heca.agents.experts.expert import ExpertAgent
from heca.agents.agent import Agent
from heca.conditions.evaluator import AgentFeedback
from heca.graphs.graph import Graph
from heca.learning.learner import Learner
from heca.misc import logger
from heca.data.data import DCScene
from heca.data.entity import Entity
from heca.scenes.scene import Scene


class Heca(Agent):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        agents: Sequence[ExpertAgent.Config]
        scene: Scene.Config
        learner: Learner.Config
        label: str = "heca"
        visualize: bool = True
        downstream_virtual: bool = False
        inference: bool = False
        step_multiplier: int = 2

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.learner = Learner.get(self.cfg.learner).register(self.cfg.tag)
        self.end_flag = False
        if self.cfg.inference:
            self.learner.eval()

        self.evaluator.set_max_steps(len(self.cfg.agents) * self.cfg.step_multiplier)
        self.graph = Graph.generate(list(self.cfg.agents), self.entities)
        self.graph.plot(Agent.load_dir(self.cfg))
        self.graph.log()

    def step(
        self, x: DCScene, new_episode: bool = False
    ) -> tuple[DCScene, AgentFeedback]:
        self.graph.set_start(x)
        data = self.graph.export()
        option = self.learner.predict(data, self.cfg.tag, new_episode)
        a, y = self.graph.select(option)
        if logger.DEBUG:
            logger.debug(f"Start:\n{str(x)}")
            logger.debug(f"Goal:\n{str(y)}")
            # logger.debug(str(self.graph.ns_entity))
            input("Press Enter to continue...")

        ds_agent = ExpertAgent.get(a)
        if ds_agent.evaluator.valid_task(x, y):
            if self.cfg.downstream_virtual:
                z = y.copy()  # pretend that downstream perfectly achieved the goal
                lfb = AgentFeedback(terminal=True, reward=1.0, truncated=False)

            else:
                z, lfb = ExpertAgent.get(a).act(x, y)
        else:
            # Sub Agent rejects goal
            z = x.copy()
            lfb = AgentFeedback(terminal=True, reward=0.0, truncated=False)

        fb = self.evaluator.step(z, lfb)
        self.end_flag = self.learner.update(
            fb.reward, fb.terminal, fb.truncated, self.cfg.tag
        )
        return z, fb

    def act(self, x: DCScene, y: DCScene) -> tuple[DCScene, AgentFeedback]:
        self.graph.set_goal(y)
        self.evaluator.reset(y)
        z, fb = self.step(x, True)
        while not (fb.truncated or fb.terminal):
            z, fb = self.step(z)
        return z, fb

    def sample(self) -> tuple[DCScene, DCScene]:
        scene = Scene.get(self.cfg.scene)
        (x, ix), (y, iy) = scene.sample_task()
        logger.debug("New Episode")
        return x, y

    @cached_property
    def entities(self) -> dict[str, Entity]:
        values = {}
        for cfg in self.cfg.agents:
            values.update(Agent.get(cfg).entities)
        return values

    def _load(self, path: Path):
        pass

    def _save(self, path: Path):
        pass
