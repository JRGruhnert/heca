from dataclasses import dataclass
import pathlib
from typing import Sequence

from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph
from heca.learning.learner import Learner
from heca.misc import logger
from heca.data.data import DCScene
from heca.misc.base import Configurable
from heca.scenes.scene import Scene, SceneFeedback


class Heca(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        agents: Sequence[ExpertModel.Config]
        learner: Learner.Config
        visualize: bool = False
        inference: bool = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.learner = Learner.get(self.cfg.learner)
        if self.cfg.inference:
            self.learner.eval()
        self.current_step = 0
        self.max_steps = len(self.cfg.agents) * self.cfg.learner.step_multiplier
        self._x: DCScene | None = None
        self._y: DCScene | None = None
        self.scene = Scene.get(self.cfg.agents[0].scene)

        for a in self.cfg.agents:
            expert = ExpertModel.get(a, auto_load=False)
            expert.use_gt(self.cfg.learner.use_gt)
            expert.load()

        if self.cfg.learner.reload:  # nees to be before Graph.generate
            for a in self.cfg.agents:
                ExpertModel.get(a).force_recompute()

        self.graph = Graph.generate(list(self.cfg.agents))
        self.graph.plot(path=self.scene.save_dir(self.scene.cfg))
        self.graph.plot_connections(path=self.scene.save_dir(self.scene.cfg))
        self.graph.log()

        if self.cfg.learner.virtual:
            for a in self.cfg.agents:
                ExpertModel.get(a).virtual()

    def step(
        self, x: DCScene, y: DCScene, new_ep: bool = False
    ) -> tuple[DCScene, SceneFeedback, bool]:
        self.graph.set_start(x)
        data = self.graph.export()
        option = self.learner.predict(data, new_ep)
        a, s = self.graph.select(option)
        if logger.DEBUG:
            logger.debug(f"Start:\n{str(x)}")
            logger.debug(f"Goal:\n{str(y)}")
            # logger.debug(str(self.graph.ns_entity))
            input("Press Enter to continue...")

        z, lfb = ExpertModel.get(a).act(x, s, y)

        fb = self.apply_truncation(lfb)
        lock = self.learner.update(fb)
        return z, fb, lock

    def act(self, x: DCScene, y: DCScene) -> tuple[DCScene, SceneFeedback]:
        self.graph.set_goal(y)
        self.current_step = 0
        z, fb, lock = self.step(x, y, True)
        while not (fb.truncated or fb.terminal or lock):
            z, fb, lock = self.step(z, y)
        return z, fb

    def sample(self) -> tuple[DCScene, DCScene]:
        (x, ix), (y, iy) = self.scene.sample_task()
        logger.debug("New Episode")
        return x, y

    def tick(self) -> bool:
        if self._x is None or self._y is None:
            self._x, self._y = self.sample()
            self.graph.set_goal(self._y)
            self.current_step = 0
            new_ep = True
        else:
            new_ep = False

        z, fb, lock = self.step(self._x, self._y, new_ep)
        self._x = z

        if fb.truncated or fb.terminal or lock:
            self._x = None
            self._y = None

        return lock

    def apply_truncation(self, lfb: SceneFeedback) -> SceneFeedback:
        if lfb.terminal and lfb.reward < 0:
            success = lfb.terminal
        else:
            success = False
        reward = self.cfg.learner.step_reward + self.cfg.learner.success_reward * int(
            success
        )

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return SceneFeedback(reward=reward, terminal=success, truncated=truncated)
