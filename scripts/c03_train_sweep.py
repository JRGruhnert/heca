import argparse
import random
import signal

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

from heca.agents.heca import Heca
from heca.learning import dist as pdist
from heca.misc import logger

from scripts.common.args import add_heca_arguments, generate_tag, generate_group
from scripts.common.helper import generate_clients
from scripts.common.scenes import scene_tags

import conf.networks


def selected_scenes(args) -> list[str]:
    """The scene tags (one client each) this run trains on; imports nothing."""
    return [tag for tag in scene_tags() if not args.scene or tag == args.scene]


def worker(rank: int, world_size: int, args) -> None:
    """One client, one process: the same run as before, minus the shared server."""
    pdist.pin_intra_op_threads()
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    hecas = generate_clients(
        generate_tag(args),
        generate_group(args),
        conf.networks.get(args.network),
        selected_scenes(args),
        inference=args.inference,
        federated=args.federated,
        virtual=args.virtual,
        use_wandb=args.wandb,
        reload=args.reload,
        use_gt=args.gt,
        smode=args.smode,
        n_batch=args.batch,
        method=args.method,
        personal_coef=args.personal_coef,
        mu=args.mu,
        lr_annealing=True,  # this script anneals, the sequential one does not
        rank=rank,
        world_size=world_size,
    )
    agent = Heca.get(hecas[0])

    n = 0
    while n < args.batch:
        if agent.tick():
            n += 1
            agent.learner.sync()
    logger.info(f"[{generate_tag(args)}] rank {rank}/{world_size} finished {n} episodes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_heca_arguments(parser)
    args = parser.parse_args()

    def _handle_stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_stop)

    args.seed = args.seed if args.seed is not None else random.randrange(2**31)

    scenes = selected_scenes(args)
    n_ranks = args.ranks or len(scenes)
    if n_ranks != len(scenes):
        raise SystemExit(
            f"--ranks {n_ranks} but {len(scenes)} clients; one process per client "
            "is required. Pass --scene to train a single client."
        )
    logger.info(f"Training {len(scenes)} clients in {n_ranks} process(es).")
    pdist.spawn(worker, n_ranks, args=(args,))


if __name__ == "__main__":
    main()

# good scenes: 0, 1, 8
# bad scenes: 2, 4
# fixable: 3, 5, 6, 7, 9, 10
