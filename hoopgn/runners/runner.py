from abc import abstractmethod
from dataclasses import dataclass, field

from hoopgn.entities.entities import properties_to_entities
from hoopgn import logger
from hoopgn.base import ConfigurableClass
from hoopgn.entities.entity import Entity
from hoopgn.environments.environment import Environment
from hoopgn.properties.property import Property
from hoopgn.agents.agent import Agent
from hoopgn.policies.tapas_policy import TapasPolicy


class HoopGNRunner(ConfigurableClass):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        agents: list[Agent.Config]
        environments: list[Environment.Config]
        properties: list[Property.Config]
        entities: list[Entity.Config] = field(init=False)

        def __post_init__(self):
            assert len(self.agents) > 0, "At least one skill must be provided."
            assert len(self.properties) > 0, "At least one property must be provided."
            logger.warning(
                f"HoopGN multiple version support should be removed in the future."
            )
            self.entities = properties_to_entities(properties=self.properties)
            for agent in self.agents:
                if isinstance(agent.policy, TapasPolicy):
                    logger.warning(
                        f"Agent '{agent.ident.label}' is a Tapas agent. Performing automated property assignment."
                    )
                    agent.policy.cfg.properties = self.properties

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def run(self):
        raise NotImplementedError()
