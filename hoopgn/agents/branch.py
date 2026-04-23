from dataclasses import dataclass
from functools import cached_property

from hoopgn.domains.domain import Domain
from hoopgn.observation.td_entity import TDEntity
from hoopgn.observation.td_scene import TDScene
from hoopgn.agents.agent import Agent


class BranchAgent(Agent):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        domains: set[Domain.Signature]
        agents: set[Agent.Signature]

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def act(self, x: TDScene, y: TDScene) -> tuple[float, bool, bool]:
        raise NotImplementedError()

    def predict(self, x: TDScene) -> tuple[Agent.Signature, TDScene, TDScene]:
        raise NotImplementedError()

    def sample_task(self) -> tuple[TDScene, TDScene]:
        raise NotImplementedError()

    @cached_property
    def precons(self) -> dict[str, TDEntity]:
        raise NotImplementedError()
        # return self.policy.load_precons()

    @cached_property
    def postcons(self) -> dict[str, TDEntity]:
        raise NotImplementedError()
        # return self.policy.load_postcons()
