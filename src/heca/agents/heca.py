from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Sequence

from heca.agents.experts.expert import ExpertAgent
from heca.agents.agent import Agent
from heca.graphs.graph import Graph
from heca.learning.learner import Learner
from heca.misc import logger
from heca.data.data import DCScene
from heca.data.entity import Entity
from heca.scenes.scene import Scene, SceneFeedback


class Heca(Agent):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        agents: Sequence[ExpertAgent.Config]
        learner: Learner.Config
        label: str = "heca"
        visualize: bool = True
        downstream_virtual: bool = False
        inference: bool = False
        step_multiplier: int = 2
        success_reward: float = 1.0
        step_reward: float = -0.01

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.learner = Learner.get(self.cfg.learner).register(self.cfg.tag)
        if self.cfg.inference:
            self.learner.eval()
        self.current_step = 0
        self.max_steps = len(self.cfg.agents) * self.cfg.step_multiplier

        self.graph = Graph.generate(list(self.cfg.agents), self.entities)
        self.graph.plot(Agent.load_dir(self.cfg))
        self.graph.log()

        if self.cfg.downstream_virtual:
            for a in self.cfg.agents:
                ExpertAgent.get(a).virtual()

    def step(
        self, x: DCScene, y: DCScene, new_ep: bool = False
    ) -> tuple[DCScene, SceneFeedback, bool]:
        self.graph.set_start(x)
        data = self.graph.export()
        option = self.learner.predict(data, self.cfg.tag, new_ep)
        a, s = self.graph.select(option)
        if logger.DEBUG:
            logger.debug(f"Start:\n{str(x)}")
            logger.debug(f"Goal:\n{str(y)}")
            # logger.debug(str(self.graph.ns_entity))
            input("Press Enter to continue...")

        z, lfb = ExpertAgent.get(a).act(x, s, y)

        fb = self.apply_truncation(lfb)
        lock = self.learner.update(fb, self.cfg.tag)
        self.learner.sync_inference()
        return z, fb, lock

    def act(self, x: DCScene, y: DCScene) -> tuple[DCScene, SceneFeedback]:
        self.graph.set_goal(y)
        self.current_step = 0
        z, fb, lock = self.step(x, y, True)
        while not (fb.truncated or fb.terminal or lock):
            z, fb, lock = self.step(z, y)
        return z, fb

    def sample(self) -> tuple[DCScene, DCScene]:
        scene = Scene.get(self.cfg.scene)
        (x, ix), (y, iy) = scene.sample_task()
        logger.debug("New Episode")
        return x, y

    def apply_truncation(self, lfb: SceneFeedback) -> SceneFeedback:
        if lfb.terminal and lfb.reward < 0:
            success = lfb.terminal
        else:
            success = False
        reward = self.cfg.step_reward + self.cfg.success_reward * int(success)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return SceneFeedback(reward=reward, terminal=success, truncated=truncated)

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
