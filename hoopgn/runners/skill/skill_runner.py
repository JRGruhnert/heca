from abc import abstractmethod
from dataclasses import dataclass

from hoopgn.misc import logger
from hoopgn.agents.agent import Agent
from hoopgn.runners.runner import HoopGNRunner
from hoopgn.policies.leafs.tapas_policy import TapasPolicy


class SkillRunner(HoopGNRunner):
    @dataclass(kw_only=True)
    class Config(HoopGNRunner.Config):
        agent: Agent.Config | None

        def __post_init__(self):
            logger.warning(
                f"SkillRunner multiple version support should be removed in the future."
            )
            super().__post_init__()
            if self.agent and isinstance(self.agent.policy, TapasPolicy):
                self.agent.policy.cfg.properties = self.properties

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.agents = Agent.build_registry(cfg.agents)
        self.agent = Agent.from_config(cfg.agent) if cfg.agent else None

    def run(self):
        if self.cfg.agent:
            if self.agent:
                self.skill_run(self.agent)
            else:
                raise ValueError(f"Agent '{self.cfg.agent}' not found in agents.")
        else:
            for agent in self.agents.values():
                self.skill_run(agent)

    @abstractmethod
    def skill_run(self, agent: Agent):
        raise NotImplementedError()
