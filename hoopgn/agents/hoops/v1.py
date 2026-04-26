from dataclasses import dataclass, field

from hoopgn.agents.agent import Agent
from hoopgn.properties.property import Property
from hoopgn.evaluators.dense import DenseEvaluator
from hoopgn.misc.ppo import PPO
from hoopgn.agents.hoops.hoop import HoopAgent
from hoopgn.evaluators.evaluator import SceneEvaluator
from hoopgn.generators.mp import MPGenerator
from hoopgn.generators.generator import Generator
from hoopgn.networks.hoops.bases.v1 import V1Network
from hoopgn.networks.hoops.hoop import HoopNetwork


class HoopV1Agent(HoopAgent):
    @dataclass(kw_only=True)
    class Query(HoopAgent.Query):
        label: str = "v1"

    @dataclass(kw_only=True)
    class Config(HoopAgent.Config):
        agents: list[Agent.Config]
        properties: list[Property.Config]
        evaluator: SceneEvaluator.Config = DenseEvaluator.Config()
        reinforcement: PPO.Config = PPO.Config()
        hoop: HoopNetwork.Config = field(init=False)
        generator: Generator.Config = field(init=False)
        training: bool = False
        max_steps: int = 10

        def __post_init__(self):
            self.hoop = HoopNetwork.Config(
                query=HoopNetwork.Query(label="v1"),
                base=V1Network.Config(),
                encoder=set(p.encoder for p in self.properties),
            )
            self.generator = MPGenerator.Config(
                agents=self.agents,
                properties=self.properties,
            )
