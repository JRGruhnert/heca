from dataclasses import dataclass
from typing import Sequence

from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.learning.learner import Learner
from heca.misc import logger
from heca.misc.interrupt import stop_requested
from heca.data.data import DCScene
from heca.misc.base import Configurable
from heca.scenes.scene import Scene, SceneFeedback


class Heca(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        agents: Sequence[ExpertModel.Config]
        learner: Learner.Config
        smode: SubgoalMode
        visualize: bool
        inference: bool
        virtual: bool
        reload: bool
        use_gt: bool
        gating: bool

        fit_rotation: bool = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.learner = Learner.get(self.cfg.learner)
        if self.cfg.inference:
            self.learner.eval()

        self._x: DCScene | None = None
        self._y: DCScene | None = None
        self.scene = Scene.get(self.cfg.agents[0].scene)

        for a in self.cfg.agents:
            expert = ExpertModel.get(a, auto_load=False)
            expert.use_gt(self.cfg.use_gt)
            expert.load()

            if self.cfg.reload:
                expert.force_recompute()

            if self.cfg.virtual:
                expert.virtual()

        self.graph = Graph.generate(
            list(self.cfg.agents), smode=cfg.smode, gating=cfg.gating
        )
        self.graph.plot(path=self.scene.save_dir(self.scene.cfg))
        self.graph.log()

    def step(
        self, x: DCScene, y: DCScene, budget: float = 0.0
    ) -> tuple[DCScene, SceneFeedback, bool]:
        data = self.graph.build(x, y, budget)
        option = self.learner.predict(data)
        a, s = self.graph.select(option)
        z, fb = ExpertModel.get(a).act(x, s)
        finished = self.learner.update(fb)
        return z, fb, finished

    def act(self, x: DCScene, y: DCScene) -> tuple[DCScene, SceneFeedback]:
        z, fb, finished = self.step(x, y, 0.0)
        while not (fb.end or finished):
            z, fb, finished = self.step(z, y, fb.budget)
        return z, fb

    def sample(self) -> tuple[DCScene, DCScene]:
        (x, ix), (y, iy) = self.scene.sample_task()
        logger.debug("New Episode")
        return x, y

    def tick(self) -> bool:
        if stop_requested():
            return True

        if self._x is None or self._y is None:
            self._x, self._y = self.sample()

        z, fb, finished = self.step(self._x, self._y)
        self._x = z

        if fb.end or finished:
            self._x = None
            self._y = None

        return finished
