import numpy as np
import torch

from heca.graphs.graph import SubgoalMode
from heca.agents.heca import Heca
from heca.misc import logger
from scripts.common.args import generate_tag, generate_group
from scripts.common.helper import generate_client, get_sim_args

import conf.networks

DEFAULTS: dict[str, object] = {
    "tag": "final",
    "scene": "scene0",
    "smode": SubgoalMode.BOTH,
    "wandb": True,
    "batch": 1000,
    "repeats": 3,
    "gt": True,
    "virtual": True,
}

RUNS: list[dict[str, object]] = [
    # {"network": "a0"},
    # {"network": "a1"},
    # {"network": "a2"},
    # {"network": "a3"},
    # {"network": "a4"},
    {"network": "a5"},
    {"network": "a6"},
    {"network": "a7"},
    {"network": "a8"},
    {"network": "a9"},
]


def main():
    planned = list(get_sim_args(DEFAULTS, RUNS))
    logger.info(f"{len(planned)} run(s) planned:")
    for args in planned:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        heca_cfg = generate_client(
            generate_tag(args, False),
            generate_group(args, False),
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

        heca = Heca.get(heca_cfg)
        heca.train(args.batch)


if __name__ == "__main__":
    main()
