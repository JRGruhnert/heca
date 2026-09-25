import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Iterator

from heca.agents.heca import Heca
from heca.graphs.graph import SubgoalMode
from heca.heca_gnn.network import Network
from heca.learning.fedprox import FedProxPPO
from heca.learning.ppo import PPO
from heca.misc import logger
from scripts.common.args import (
    add_eval_arguments,
    add_fed_arguments,
    add_heca_arguments,
)
from scripts.common.scenes import experts_for_scene


def fmt_duration(seconds: float) -> str:
    total = int(round(seconds))
    if total < 1:
        return "<1s"
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def find_checkpoint(run_dir: Path, name: str) -> Path:
    """A named checkpoint in a run dir, or the highest-numbered ``ckp_*.pt``."""
    if name != "latest":
        return run_dir / name
    checkpoints = sorted(
        run_dir.glob("ckp_*.pt"), key=lambda p: int(p.stem.split("_")[1])
    )
    if not checkpoints:
        raise FileNotFoundError(f"no ckp_*.pt in {run_dir}")
    return checkpoints[-1]


def get_sim_args(
    defaults: dict[str, object], runs: list[dict[str, object]]
) -> Iterator[argparse.Namespace]:
    """Yield one seeded Namespace per run (and per repeat) of the table."""
    parser = argparse.ArgumentParser(add_help=False)
    add_heca_arguments(parser)
    add_fed_arguments(parser)
    add_eval_arguments(parser)
    base = {action.dest: action.default for action in parser._actions}
    for row in runs:
        fields = {**base, **defaults, **row}
        repeats = int(fields.pop("repeats", 1))  # type: ignore

        for i in range(repeats):
            run_fields = dict(fields)
            run_fields["seed"] = i
            yield argparse.Namespace(**run_fields)


def generate_client(
    tag: str,
    group: str,
    network: Network.Config,
    scene: str,
    smode: SubgoalMode,
    inference: bool,
    use_wandb: bool,
    virtual: bool,
    reload: bool,
    use_gt: bool,
    n_batch: int,
    lr_annealing: bool,
) -> Heca.Config:
    wandb = logger.WandBConfig(enabled=use_wandb)
    experts = experts_for_scene(scene)
    return Heca.Config(
        experts=experts,
        learner=PPO.Config(
            tag=f"{scene}_{tag}",
            group=group,
            network=network,
            wandb=wandb,
            max_update=n_batch,
            lr_annealing=lr_annealing,
        ),
        inference=inference,
        virtual=virtual,
        reload=reload,
        use_gt=use_gt,
        smode=smode,
    )


def generate_fed_client(
    tag: str,
    group: str,
    network: Network.Config,
    scene: str,
    smode: SubgoalMode,
    inference: bool,
    use_wandb: bool,
    virtual: bool,
    reload: bool,
    use_gt: bool,
    n_batch: int,
    mu: float,
    k: int,
    lr_annealing: bool,
    server_lr: float | None = None,
    fedavgm_beta: float = 0.9,
) -> Heca.Config:
    wandb = logger.WandBConfig(enabled=use_wandb)
    experts = experts_for_scene(scene)
    return Heca.Config(
        experts=experts,
        learner=FedProxPPO.Config(
            tag=f"{scene}_{tag}",
            group=group,
            network=network,
            wandb=wandb,
            max_update=n_batch,
            lr_annealing=lr_annealing,
            label="federated",
            subdir=f"{tag}/clients/{scene}",
            k=k,
            server_lr=server_lr,
            fedavgm_beta=fedavgm_beta,
            mu=mu,
        ),
        inference=inference,
        virtual=virtual,
        reload=reload,
        use_gt=use_gt,
        smode=smode,
    )
