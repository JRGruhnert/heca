import asyncio

from heca.agents.heca import Heca
from heca.experts.expert import ExpertModel
from heca.heca_gnn.network import Network
from heca.heca_gnn.network2 import Network2
from heca.learning.fppo import FPPO
from heca.learning.ppo import PPO
from heca.learning.server import FLServer
from conf.experts.sceneog import agents


async def train(clients: list[Heca.Config], n_batch: int = 1000):
    async def run(client: Heca.Config, n_batch: int):
        agent = Heca.get(client)
        n = 0
        while n < n_batch:
            if agent.tick():
                n += 1
                await agent.learner.sync()

    await asyncio.gather(*[run(c, n_batch) for c in clients])


def generate_fl_clients(
    tag: str, network: Network.Config, aaa: list[list[ExpertModel.Config]]
) -> list[Heca.Config]:
    server = FLServer.Config(tag=tag, network=network)
    hecas = []
    for idx, aa in enumerate(aaa):
        heca = Heca.Config(
            agents=aa,
            learner=FPPO.Config(
                tag=f"{tag}_heca{idx}",
                network=network,
                server=server,
            ),
        )
        hecas.append(heca)
    return hecas


heca_cfg = Heca.Config(
    agents=agents,
    learner=PPO.Config(
        tag="sceneog",
        network=Network2.Config(),
    ),
)

exp1 = [heca_cfg]


asyncio.run(train(exp1))
