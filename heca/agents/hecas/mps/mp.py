from dataclasses import dataclass, field

from heca.agents.agent import Agent
from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.encoders.v1.area import AreaEncoder
from heca.properties.encoders.v1.rotation import QuaternionEncoder
from heca.properties.encoders.v1.bool import BoolEncoder
from heca.properties.encoders.v1.range import RangeEncoder
from heca.properties.encoders.v1.flip import FlipEncoder
from heca.misc.ppo import PPO
from heca.agents.hecas.heca import Heca
from heca.evaluators.agent_hoop import HoopEvaluator
from heca.generators.mp import MPGenerator
from heca.generators.generator import HoopGenerator
from heca.networks.bases.v1 import V1Network
from heca.networks.heca.hoop import HoopNetwork


class MPAgent(Heca):
    @dataclass(kw_only=True)
    class Query(Heca.Query):
        label: str

    @dataclass(kw_only=True)
    class Config(Heca.Config):
        agents: set[Agent.Query]
        encoders: set[PropertyEncoder.Query] = set(
            [
                AreaEncoder.Query(),
                QuaternionEncoder.Query(),
                BoolEncoder.Query(),
                RangeEncoder.Query(),
                FlipEncoder.Query(),
            ]
        )
        evaluator: HoopEvaluator.Config = HoopEvaluator.Config()
        ppo: PPO.Config = PPO.Config()
        hoop: HoopNetwork.Config = field(init=False)
        generator: HoopGenerator.Config = field(init=False)
        training: bool = False
        max_steps: int = 10

        def __post_init__(self):
            self.hoop = HoopNetwork.Config(
                query=HoopNetwork.Query(label="mp"),
                base=V1Network.Config(),
                encoders=self.encoders,
            )
            self.generator = MPGenerator.Config(
                agents=self.agents,
                meta=self.query.label,
            )
