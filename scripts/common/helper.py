from pathlib import Path

from heca.agents.heca import Heca
from heca.experts.expert import ExpertModel
from heca.graphs.graph import SubgoalMode
from heca.heca_gnn.network import Network
from heca.learning.ditto import DittoPPO
from heca.learning.fedprox import FedProxPPO
from heca.learning.fppo import FPPO
from heca.learning.ppo import PPO
from heca.learning.server import FLServer
from heca.misc import logger


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
    clients: dict[str, list[ExpertModel.Config]],
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
):
    wandb = logger.WandBConfig(enabled=use_wandb)
    hecas = []
    server = None
    if federated:
        server_cfg = FLServer.Config(tag=tag, network=network)
        server = FLServer.get(server_cfg)
        # the learner class is resolved from the Config type by
        # Learner.get(cfg), so choosing the method means choosing the class here:
        # FPPO is FedAvg (no penalty), FedProxPPO adds mu, DittoPPO adds lambda
        learner_cfg = {
            "fedavg": FPPO.Config,
            "fedprox": FedProxPPO.Config,
            "ditto": DittoPPO.Config,
        }[method]
        for scene, agents in clients.items():
            learner_kwargs = dict(
                tag=f"{scene}_{tag}",
                group=group,
                network=network,
                server=server_cfg,
                wandb=wandb,
                max_update=n_batch,
                lr_annealing=lr_annealing,
                # same label as the server, so the run dirs nest instead of
                # landing next to plain runs in data/network/standard/
                label="federated",
                subdir=f"{tag}/clients/{scene}",
            )
            if method == "ditto":
                learner_kwargs["personal_coef"] = personal_coef
            if method != "fedavg":
                learner_kwargs["mu"] = mu
            heca = Heca.Config(
                agents=agents,
                # every artifact of the federated run lives under
                # data/network/federated/<tag>/: the server checkpoint in the
                # run dir itself, each client in clients/<scene>/
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
        for scene, agents in clients.items():
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
    return hecas, server
