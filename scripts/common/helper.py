from collections.abc import Sequence
from pathlib import Path

from heca.agents.heca import Heca
from heca.graphs.graph import SubgoalMode
from heca.heca_gnn.network import Network
from heca.learning.ditto import DittoPPO
from heca.learning.fedprox import FedProxPPO
from heca.learning.fppo import FPPO
from heca.learning.ppo import PPO
from heca.misc import logger
from scripts.common.scenes import agents_for_scene


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


def generate_clients(
    tag: str,
    group: str,
    network: Network.Config,
    scenes: Sequence[str],
    smode: SubgoalMode,
    inference: bool,
    federated: bool,
    use_wandb: bool,
    virtual: bool,
    reload: bool,
    use_gt: bool,
    n_batch: int,
    method: str = "fedprox",
    personal_coef: float = 0.1,
    mu: float = 0.01,
    lr_annealing: bool = False,
    rank: int | None = None,
    world_size: int | None = None,
) -> list[Heca.Config]:
    if rank is not None and world_size != len(scenes):
        raise ValueError(
            f"{world_size} ranks but {len(scenes)} clients; one process per client "
            "is required (use --ranks with the number of scenes)."
        )
    wandb = logger.WandBConfig(enabled=use_wandb)
    hecas = []
    mine = [scenes[rank]] if rank is not None else list(scenes)
    if federated:
        learner_cfg = {
            "fedavg": FPPO.Config,
            "fedprox": FedProxPPO.Config,
            "ditto": DittoPPO.Config,
        }[method]
        for scene in mine:
            agents = agents_for_scene(scene)
            learner_kwargs = dict(
                tag=f"{scene}_{tag}",
                group=group,
                network=network,
                wandb=wandb,
                max_update=n_batch,
                lr_annealing=lr_annealing,
                label="federated",
                subdir=f"{tag}/clients/{scene}",
            )
            if method == "ditto":
                learner_kwargs["personal_coef"] = personal_coef
            if method != "fedavg":
                learner_kwargs["mu"] = mu
            heca = Heca.Config(
                agents=agents,
                learner=learner_cfg(**learner_kwargs),
                visualize=False,
                inference=inference,
                virtual=virtual,
                reload=reload,
                use_gt=use_gt,
                smode=smode,
            )
            hecas.append(heca)
    else:
        for scene in mine:
            agents = agents_for_scene(scene)
            heca = Heca.Config(
                agents=agents,
                learner=PPO.Config(
                    tag=f"{scene}_{tag}",
                    group=group,
                    network=network,
                    wandb=wandb,
                    max_update=n_batch,
                    lr_annealing=lr_annealing,
                ),
                visualize=False,
                inference=inference,
                virtual=virtual,
                reload=reload,
                use_gt=use_gt,
                smode=smode,
            )
            hecas.append(heca)
    return hecas
