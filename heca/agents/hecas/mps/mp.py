from dataclasses import dataclass, field

from heca.agents.leafs.leaf import LeafAgent
from heca.entities.real import RealEntity
from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.encoders.v1.area import AreaEncoder
from heca.properties.encoders.v1.rotation import QuaternionEncoder
from heca.properties.encoders.v1.bool import BoolEncoder
from heca.properties.encoders.v1.range import RangeEncoder
from heca.properties.encoders.v1.flip import FlipEncoder
from heca.agents.hecas.heca import Heca
from heca.evaluators.heca import HecaEvaluator
from heca.generators.mp import MPGenerator
from heca.generators.generator import HecaGenerator
from heca.heca_gnn.bases.v1 import V1Network
from heca.heca_gnn.hecagn import HecaGN
from heca.misc.ppo import PPO


class MPHeca(Heca):
    @dataclass(kw_only=True)
    class Query(Heca.Query):
        version: str = "v1"

    @dataclass(kw_only=True)
    class Config(Heca.Config):
        entity: RealEntity.Config
        agents: set[LeafAgent.Query]
        encoders: set[PropertyEncoder.Query] = set(
            [
                AreaEncoder.Query(),
                QuaternionEncoder.Query(),
                BoolEncoder.Query(),
                RangeEncoder.Query(),
                FlipEncoder.Query(),
            ]
        )
        evaluator: HecaEvaluator.Config = HecaEvaluator.Config()
        ppo: PPO.Config = PPO.Config()
        heca: HecaGN.Config = field(init=False)
        generator: HecaGenerator.Config = field(init=False)
        training: bool = False
        max_steps: int = 10

        def __post_init__(self):
            self.heca = HecaGN.Config(
                query=HecaGN.Query(label="mp"),
                base=V1Network.Config(),
                encoders=self.encoders,
            )
            self.generator = MPGenerator.Config(
                agents=self.agents,
                entity=self.entity,
            )
