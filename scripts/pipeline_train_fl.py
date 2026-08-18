"""Train Heca clients on the configured Tapas experts.

Each scene module (``conf.experts.sceneX``) becomes one client whose agents are
the TapasExpert configs defined in that module. Can run federated (FPPO) or
plain (PPO) training.
"""

import argparse
import asyncio

import matplotlib

matplotlib.use("Agg")  # headless plotting (graph plot in Heca.__init__)

from heca.agents.heca import Heca
from heca.experts.expert import ExpertModel
from heca.learning.fppo import FPPO
from heca.learning.learner import WandBConfig
from heca.learning.ppo import PPO
from heca.learning.server import FLServer

import conf.networks
import conf.experts.scene1
import conf.experts.scene2
import conf.experts.scene3
import conf.experts.scene4
import conf.experts.scene5

SCENE_MODULES = (
    conf.experts.scene1,
    conf.experts.scene2,
    conf.experts.scene3,
    conf.experts.scene4,
    conf.experts.scene5,
)

NETWORK_NAMES = [name for name in vars(conf.networks) if not name.startswith("_")]


def collect_clients():
    """Yield ``(scene_name, agents)`` pairs, one client per scene module."""
    for mod in SCENE_MODULES:
        scene_name = mod.__name__.split(".")[-1]
        yield scene_name, list(mod.agents)


def generate_clients(
    tag: str,
    network,
    clients: list[list[ExpertModel.Config]],
    learner: str,
    virtual: bool = False,
    wandb_enabled: bool = False,
):
    """Create one Heca client per agent list."""
    wandb = WandBConfig(enabled=wandb_enabled)
    if learner == "fppo":
        server = FLServer.Config(tag=tag, network=network)
        hecas = []
        for idx, agents in enumerate(clients):
            heca = Heca.Config(
                agents=agents,
                learner=FPPO.Config(
                    tag=f"{tag}_heca{idx}",
                    network=network,
                    server=server,
                    virtual=virtual,
                    wandb=wandb,
                ),
            )
            hecas.append(heca)
        return hecas

    # Plain PPO, no server.
    hecas = []
    for idx, agents in enumerate(clients):
        heca = Heca.Config(
            agents=agents,
            learner=PPO.Config(
                tag=f"{tag}_heca{idx}",
                network=network,
                virtual=virtual,
                wandb=wandb,
            ),
        )
        hecas.append(heca)
    return hecas


async def train(clients: list[Heca.Config], n_batch: int = 1000):
    # Instantiate all clients up front so every scene builds its graph and
    # writes its plots immediately, rather than only when the first client
    # yields control back to the event loop.
    agents = [Heca.get(client) for client in clients]

    async def run(agent: Heca, n_batch: int):
        n = 0
        while n < n_batch:
            if agent.tick():
                n += 1
                await agent.learner.sync()
            # Yield to the event loop so the clients interleave step-by-step
            # instead of one client hogging the CPU for a whole batch.
            await asyncio.sleep(0)

    await asyncio.gather(*[run(a, n_batch) for a in agents])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        required=True,
        help="Run tag for the training.",
    )
    parser.add_argument(
        "--network",
        choices=NETWORK_NAMES,
        default="default",
        help="Network config name from conf.networks.",
    )
    parser.add_argument(
        "--learner",
        choices=["fppo", "ppo"],
        default="fppo",
        help="Federated (FPPO) or plain (PPO) training.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1000,
        help="Number of training batches per client.",
    )
    parser.add_argument(
        "--virtual",
        action="store_true",
        help="Initialize agents in virtual mode.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging. Disabled by default for multi-client runs.",
    )
    parser.add_argument(
        "--scene",
        help="Only train this scene module (e.g. sceneog). Defaults to all scenes.",
    )
    args = parser.parse_args()

    network = getattr(conf.networks, args.network)

    clients = []
    for scene_name, agents in collect_clients():
        if args.scene and scene_name != args.scene:
            continue
        clients.append(agents)

    exp = generate_clients(
        args.tag, network, clients, args.learner, args.virtual, args.wandb
    )
    asyncio.run(train(exp, args.batch))


if __name__ == "__main__":
    main()
