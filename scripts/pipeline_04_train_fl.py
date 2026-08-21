import argparse
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib

from heca.heca_gnn.network import Network

matplotlib.use("Agg")

from heca.agents.heca import Heca
from heca.experts.expert import ExpertModel
from heca.learning.fppo import FPPO
from heca.learning.learner import WandBConfig
from heca.learning.ppo import PPO
from heca.learning.server import FLServer

from scripts.common.args import add_scene_argument
from scripts.common.scenes import agents_by_scene

import conf.networks
from conf.networks import NETWORK_NAMES


def generate_clients(
    tag: str,
    network: Network.Config,
    clients: list[list[ExpertModel.Config]],
    virtual: bool = False,
    federated: bool = True,
    wandb_enabled: bool = False,
    reload: bool = False,
    use_gt: bool = True,
):
    wandb = WandBConfig(enabled=wandb_enabled)
    hecas = []
    server = None
    if federated:
        server_cfg = FLServer.Config(tag=tag, network=network)
        server = FLServer.get(server_cfg)
        for idx, agents in enumerate(clients):
            heca = Heca.Config(
                agents=agents,
                learner=FPPO.Config(
                    tag=f"{tag}_heca{idx}",
                    network=network,
                    server=server_cfg,
                    virtual=virtual,
                    wandb=wandb,
                    reload=reload,
                    use_gt=use_gt,
                ),
            )
            hecas.append(heca)
    else:
        for idx, agents in enumerate(clients):
            heca = Heca.Config(
                agents=agents,
                learner=PPO.Config(
                    tag=f"{tag}_heca{idx}",
                    network=network,
                    virtual=virtual,
                    wandb=wandb,
                    reload=reload,
                    use_gt=use_gt,
                ),
            )
            hecas.append(heca)
    return hecas, server


def train(
    client_cfgs: list[Heca.Config],
    server: FLServer | None,
    n_batch: int = 1000,
):
    clients = [Heca.get(client) for client in client_cfgs]
    stop = threading.Event()

    def run(agent: Heca):
        n = 0
        while n < n_batch and not stop.is_set():
            if agent.tick():
                n += 1
                agent.learner.sync()

    def shutdown():
        stop.set()
        if server is not None:
            server.stop()

    try:
        with ThreadPoolExecutor(max_workers=len(clients)) as pool:
            futures = [pool.submit(run, a) for a in clients]
            for future in as_completed(futures):
                future.result()
    except KeyboardInterrupt:
        shutdown()
        raise
    except Exception:
        shutdown()
        raise


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
        "--federated",
        action="store_true",
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
        "--reload",
        action="store_true",
        help="Reload expert conditions instead of loading conditions.joblib.",
    )
    parser.add_argument(
        "--use-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ground-truth observations (default: true).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging. Disabled by default for multi-client runs.",
    )
    add_scene_argument(parser)
    args = parser.parse_args()

    def _handle_stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_stop)

    network = getattr(conf.networks, args.network)

    clients = []
    for scene_tag, models in agents_by_scene().items():
        if args.scene and scene_tag != args.scene:
            continue
        clients.append(models)

    exp, server = generate_clients(
        args.tag,
        network,
        clients,
        federated=args.federated,
        virtual=args.virtual,
        wandb_enabled=args.wandb,
        reload=args.reload,
        use_gt=args.use_gt,
    )
    train(exp, server, args.batch)


if __name__ == "__main__":
    main()
