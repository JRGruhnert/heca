import os

import numpy as np
import torch

from heca.graphs.graph import SubgoalMode
from heca.learning import dist as pdist
from heca.learning.learner import Learner
from heca.agents.heca import Heca
from heca.misc import logger
from heca.misc.base import latest_checkpoint

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
# mean of the second-moment
# proximal term
# alpha, mu: Clients follow the global direction more closely.
# RUNS: list[dict[str, object]] = [
#     #    {"network": "a0", "k": 1},  # FedAvg
#     #    {"network": "a0", "k": 5},  # FedAvg
#     {"network": "a0", "k": 1, "mu": 0.01},  # FedProx #proximal term
#     {"network": "a0", "k": 5, "mu": 0.01},  # FedProx
#     {"network": "a0", "k": 1, "mu": 0.1},  # FedProx
#     {"network": "a0", "k": 5, "mu": 0.1},  # FedProx
#     {"network": "a0", "k": 1, "mu": 1.0},  # FedProx
#     {"network": "a0", "k": 5, "mu": 1.0},  # FedProx
#     #    {"network": "a0", "k": 1, "fedadamw_alpha": 0.5},  # FedAdamW
#     #    {"network": "a0", "k": 5, "fedadamw_alpha": 0.5},  # FedAdamW
#     {"network": "a0", "k": 1, "fedadamw_alpha": 0.5, "mu": 0.01},  # FedAdamW + FedProx
#     {"network": "a0", "k": 5, "fedadamw_alpha": 0.5, "mu": 0.01},  # FedAdamW + FedProx
#     {"network": "a0", "k": 1, "fedadamw_alpha": 0.5, "mu": 0.1},  # FedAdamW + FedProx
#     {"network": "a0", "k": 5, "fedadamw_alpha": 0.5, "mu": 0.1},  # FedAdamW + FedProx
#     {"network": "a0", "k": 1, "fedadamw_alpha": 0.5, "mu": 1.0},  # FedAdamW + FedProx
#     {"network": "a0", "k": 5, "fedadamw_alpha": 0.5, "mu": 1.0},  # FedAdamW + FedProx
# ]
RUNS: list[dict[str, object]] = [
    {"network": "a0", "k": 1, "fedadamw_alpha": 0.5},  # FedAdamW
    {"network": "a0", "k": 5, "fedadamw_alpha": 0.5},  # FedAdamW
    {"network": "a0", "k": 1, "fedadamw_alpha": 0.5, "mu": 0.01},  # FedAdamW + FedProx
    {"network": "a0", "k": 5, "fedadamw_alpha": 0.5, "mu": 0.01},  # FedAdamW + FedProx
    {"network": "a0", "k": 1, "fedadamw_alpha": 0.5, "mu": 0.1},  # FedAdamW + FedProx
    {"network": "a0", "k": 5, "fedadamw_alpha": 0.5, "mu": 0.1},  # FedAdamW + FedProx
    {"network": "a0", "k": 1, "fedadamw_alpha": 0.5, "mu": 1.0},  # FedAdamW + FedProx
    {"network": "a0", "k": 5, "fedadamw_alpha": 0.5, "mu": 1.0},  # FedAdamW + FedProx
]


def client_cfg(args, tag: str, group: str, scene: str):
    """The client config of one scene, as the worker builds it.

    Split out so the completion check in ``main`` can ask for the same run
    directory without duplicating this argument list.
    """
    return generate_fed_client(
        tag,
        group,
        conf.networks.get(args.network),
        scene,
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


def completed_update(args, tag: str, group: str) -> int:
    """Latest update a namespace already reached, 0 when it never ran.

    Rank 0 writes the shared federated checkpoint every ``save_interval`` rounds,
    and that is the file every client resumes from, so its update count is the
    marker for "this namespace is done".
    """
    cfg = client_cfg(args, tag, group, selected_scenes(args)[0]).learner
    directory = Learner.shared_dir(cfg)
    checkpoint = latest_checkpoint(directory, "ckp") if directory else None
    if checkpoint is None:
        return 0
    number = checkpoint.stem.rsplit("_", 1)[-1]
    return int(number) if number.isdigit() else 0


def worker(rank: int, world_size: int, args, tag: str, group: str) -> None:
    scenes = selected_scenes(args)
    assert world_size == len(scenes), f"{world_size} ranks but {len(scenes)} scenes"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    heca = Heca.get(client_cfg(args, tag, group, scenes[rank]))
    heca.train(args.batch)


def main():
    planned = list(get_sim_args(DEFAULTS, RUNS))
    shard = os.environ.get("PLAN_SHARD")
    if shard:
        index, total = (int(part) for part in shard.split("/"))
        planned = planned[index::total]
    force = bool(os.environ.get("PLAN_FORCE"))
    logger.info(
        f"{len(planned)} run(s) planned" + (f" (shard {shard})" if shard else ":")
    )

    for args in planned:
        tag = generate_tag(args, True)
        group = generate_group(args, True)
        done = completed_update(args, tag, group)
        if done >= args.batch and not force:
            logger.info(
                f"[{tag}] already at update {done}/{args.batch}, skipping "
                f"(PLAN_FORCE=1 to run it again)"
            )
            continue
        pdist.spawn(
            worker,
            len(selected_scenes(args)),
            args=(args, tag, group),
            threads=args.threads,
        )


if __name__ == "__main__":
    main()
