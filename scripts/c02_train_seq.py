import os

import numpy as np
import torch

from heca.graphs.graph import SubgoalMode
from heca.agents.heca import Heca
from heca.learning.learner import Learner
from heca.misc import logger
from scripts.common.args import generate_tag, generate_group
from scripts.common.helper import generate_client, get_sim_args, latest_update

import conf.networks

DEFAULTS: dict[str, object] = {
    "tag": "final",
    "scene": "scene0",
    "smode": SubgoalMode.BOTH,
    "wandb": True,
    "batch": 1000,
    "repeats": 3,
    "gt": True,
    "virtual": False,
}

RUNS: list[dict[str, object]] = [
    # {"network": "a0"},
    # {"network": "a1"},
    # {"network": "a2"},
    # {"network": "a3"},
    # {"network": "a4"},
    # {"network": "a5"},
    # {"network": "a6"},
    # {"network": "a7"},
    # {"network": "a8"},
    # {"network": "a9"},
    {"network": "a10"},
    # {"network": "a11"},
    # {"network": "a12"},
    # {"network": "a13"},
]


def client_cfg(args, tag: str, group: str):
    """The single-agent config, as ``main`` builds it.

    Split out so the completion check can ask for the same run directory.
    """
    return generate_client(
        tag,
        group,
        conf.networks.get(args.network),
        scene=args.scene,
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
    """Latest update this namespace already reached, 0 when it never ran.

    Not federated, so the learner's own directory is the marker: every
    ``save_interval`` updates it writes ``ckp_<n>.pt`` there, and that is the
    file a restart resumes from.
    """
    cfg = client_cfg(args, tag, group).learner
    return latest_update(Learner.save_dir(cfg))


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
        tag = generate_tag(args, False)
        group = generate_group(args, False)
        done = completed_update(args, tag, group)
        if done >= args.batch and not force:
            logger.info(
                f"[{tag}] already at update {done}/{args.batch}, skipping "
                f"(PLAN_FORCE=1 to run it again)"
            )
            continue

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        heca = Heca.get(client_cfg(args, tag, group))
        heca.train(args.batch)


if __name__ == "__main__":
    main()
