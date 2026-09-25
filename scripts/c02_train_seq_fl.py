import numpy as np
import torch

from heca.graphs.graph import SubgoalMode
from heca.learning import dist as pdist
from heca.agents.heca import Heca
from heca.misc import logger

from scripts.common.args import generate_tag, generate_group
from scripts.common.scenes import selected_scenes
from scripts.common.helper import generate_fed_client, get_sim_args

import conf.networks

DEFAULTS: dict[str, object] = {
    "tag": "final",
    "scenes": [
        "scene1",
        "scene3",
        "scene5",
        "scene6",
        "scene7",
        "scene8",
        "scene9",
        "scene10",
    ],
    "smode": SubgoalMode.BOTH,
    "wandb": True,
    "batch": 1000,
    "repeats": 3,
    "gt": True,
    "virtual": True,
    "mu": 0.01,
    "k": 1,
}

RUNS: list[dict[str, object]] = [
    {"network": "a0"},
]


def worker(rank: int, world_size: int, args, tag: str, group: str) -> None:
    scenes = selected_scenes(args)
    assert world_size == len(scenes), f"{world_size} ranks but {len(scenes)} scenes"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    heca_cfg = generate_fed_client(
        tag,
        group,
        conf.networks.get(args.network),
        selected_scenes(args)[rank],
        mu=args.mu,
        k=args.k,
        server_lr=args.server_lr,
        fedavgm_beta=args.fedavgm_beta,
        inference=args.inference,
        virtual=args.virtual,
        use_wandb=args.wandb,
        reload=args.reload,
        use_gt=args.gt,
        smode=args.smode,
        n_batch=args.batch,
        lr_annealing=args.lr_annealing,
    )
    heca = Heca.get(heca_cfg)
    heca.train(args.batch)


def main():
    planned = list(get_sim_args(DEFAULTS, RUNS))
    logger.info(f"{len(planned)} run(s) planned:")

    for args in planned:
        scenes = selected_scenes(args)
        n_ranks = len(scenes)
        pdist.spawn(
            worker,
            n_ranks,
            args=(args, generate_tag(args, True), generate_group(args, True)),
            threads=args.threads,
        )


if __name__ == "__main__":
    main()
