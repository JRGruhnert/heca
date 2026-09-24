import argparse
import random
import signal

import numpy as np
import torch
import matplotlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heca.agents.heca import Heca
from heca.learning import dist as pdist
from heca.misc import logger

from scripts.common.args import add_heca_arguments, generate_tag, generate_group
from scripts.common.helper import generate_clients
from scripts.common.scenes import selected_scenes

matplotlib.use("Agg")

import conf.networks  # noqa: E402


def worker(rank: int, world_size: int, args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
        rank=rank,
        world_size=world_size,
    )
    agent = Heca.get(hecas[0])

    n = 0
    while n < args.batch:
        if agent.tick():
            n += 1
            agent.learner.sync()
    logger.info(f"[rank {rank}/{world_size}] finished {n} episodes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_heca_arguments(parser)
    args = parser.parse_args()

    def _handle_stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_stop)

    seed = args.seed if args.seed is not None else random.randrange(2**31)
    args.seed = seed  # so generate_tag picks it up

    scenes = selected_scenes(args)
    n_ranks = args.ranks or len(scenes)
    if n_ranks != len(scenes):
        raise SystemExit(
            f"--ranks {n_ranks} but {len(scenes)} clients; one process per client is "
            "required. Pass --scene to train a single client."
        )
    logger.info(f"Training {len(scenes)} clients in {n_ranks} process(es).")
    pdist.spawn(worker, n_ranks, args=(args,), threads=args.threads)


if __name__ == "__main__":
    main()

# good scenes: 0, 1, 8
# bad scenes: 2, 4
