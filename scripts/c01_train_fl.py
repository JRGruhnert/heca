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

from scripts.common.args import (
    add_fed_arguments,
    add_heca_arguments,
    generate_group,
    generate_tag,
)
from scripts.common.helper import generate_fed_client
from scripts.common.scenes import selected_scenes

matplotlib.use("Agg")

import conf.networks  # noqa: E402


def worker(rank: int, world_size: int, args) -> None:
    scenes = selected_scenes(args)
    assert world_size == len(scenes), f"{world_size} ranks but {len(scenes)} scenes"
    scene = scenes[rank]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    heca_cfg = generate_fed_client(
        generate_tag(args, federated=True),
        generate_group(args, federated=True),
        conf.networks.get(args.network),
        scene,
        inference=False,
        virtual=args.virtual,
        use_wandb=args.wandb,
        reload=args.reload,
        use_gt=args.gt,
        smode=args.smode,
        n_batch=args.batch,
        mu=args.mu,
        k=args.k,
        fedadamw_alpha=args.fedadamw_alpha,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_annealing=args.lr_annealing,
    )
    agent = Heca.get(heca_cfg)

    n = 0
    while n < args.batch:
        if agent.tick():
            n += 1
            agent.learner.sync()
    logger.info(f"[Heca on {scene}] finished {n} episodes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_heca_arguments(parser)
    add_fed_arguments(parser)
    args = parser.parse_args()

    def _handle_stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_stop)

    scenes = selected_scenes(args)
    if args.seed is None:
        args.seed = random.randrange(2**31)

    logger.info(
        f"Training {len(scenes)} clients in {len(scenes)} process(es), "
        f"{args.threads} thread(s) each, seed {args.seed}."
    )
    pdist.spawn(worker, len(scenes), args=(args,), threads=args.threads)


if __name__ == "__main__":
    main()
