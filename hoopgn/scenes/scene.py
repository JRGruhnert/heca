from dataclasses import dataclass

from hoopgn.evaluators.evaluator import SceneEvaluator
from hoopgn.misc.classes import ConfigurableClass


class Scene(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        evaluator: SceneEvaluator.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # TODO this is better sceneevaluator directly
        # Entity Evaluator between property and scene
        # (World Evaluator)

    @classmethod
    def is_equal(cls, x: TDScene, y: "Scene") -> bool:
        return x.cfg == y.cfg
