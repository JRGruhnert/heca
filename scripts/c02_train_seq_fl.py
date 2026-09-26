import os

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
}

RUNS: list[dict[str, object]] = [
    {"network": "a0", "k": 1},  # FedAvg
    {"network": "a0", "k": 5},  # FedAvg
    {"network": "a0", "k": 1, "mu": 0.01},  # FedProx
    {"network": "a0", "k": 5, "mu": 0.01},  # FedProx
    {"network": "a0", "k": 1, "mu": 0.1},  # FedProx
    {"network": "a0", "k": 5, "mu": 0.1},  # FedProx
    {"network": "a0", "k": 1, "mu": 1.0},  # FedProx
    {"network": "a0", "k": 5, "mu": 1.0},  # FedProx
    {"network": "a0", "k": 1, "fedadamw_alpha": 0.5},  # FedAdamW
    {"network": "a0", "k": 5, "fedadamw_alpha": 0.5},  # FedAdamW
    {"network": "a0", "k": 1, "fedadamw_alpha": 0.5, "mu": 0.01},  # FedAdamW + FedProx
    {"network": "a0", "k": 5, "fedadamw_alpha": 0.5, "mu": 0.01},  # FedAdamW + FedProx
    {"network": "a0", "k": 1, "fedadamw_alpha": 0.5, "mu": 0.1},  # FedAdamW + FedProx
    {"network": "a0", "k": 5, "fedadamw_alpha": 0.5, "mu": 0.1},  # FedAdamW + FedProx
    {"network": "a0", "k": 1, "fedadamw_alpha": 0.5, "mu": 1.0},  # FedAdamW + FedProx
    {"network": "a0", "k": 5, "fedadamw_alpha": 0.5, "mu": 1.0},  # FedAdamW + FedProx
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
        fedadamw_alpha=args.fedadamw_alpha,
        lr=args.lr,
        weight_decay=args.weight_decay,
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
    shard = os.environ.get("PLAN_SHARD")
    if shard:
        index, total = (int(part) for part in shard.split("/"))
        planned = planned[index::total]
    logger.info(
        f"{len(planned)} run(s) planned" + (f" (shard {shard})" if shard else ":")
    )

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
