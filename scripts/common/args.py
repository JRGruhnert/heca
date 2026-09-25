import argparse
from conf.networks import NETWORK_NAMES
from heca.graphs.graph import SubgoalMode


def add_tag_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--tag",
        required=True,
        help="Run tag for identification.",
    )


def add_smode_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--smode",
        type=SubgoalMode,
        choices=list(SubgoalMode),
        default=SubgoalMode.BOTH,
        help="Steers option generation.",
    )


def add_viewer_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Enable the passive viewer.",
    )


def add_scene_argument(parser: argparse.ArgumentParser, default=None):
    parser.add_argument(
        "--scene",
        default=default,
        help="Scene module tag (e.g. scene0, scene1).",
    )


def add_scenes_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Scene tags to train, one client each (default: --scene's tag, else "
        "every tag in conf/scenes.SCENE_TAGS).",
    )


def add_model_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model",
        default=None,
        help="Agent tag within the selected scene.",
    )


def add_virtual_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--virtual",
        action="store_true",
        help="Initialize agents in virtual mode.",
    )


def add_federated_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--federated",
        action="store_true",
        help="Federated (FPPO/FedProxPPO/DittoPPO) or plain (PPO) training.",
    )


def add_method_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--method",
        choices=("fedavg", "fedprox", "ditto"),
        default="fedprox",
        help="Federated method: fedavg (plain aggregation, no penalty), fedprox "
        "(proximal term mu on the shared model) or ditto (shared model plus a "
        "personalized local model, lambda). Only applies with --federated.",
    )


def add_personal_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--personal-coef",
        type=float,
        default=0.1,
        help="Ditto's lambda: strength of the pull of the personal model "
        "towards the shared model (1e-2..1 is the paper's range, negative or 0 "
        "disables personalization).",
    )


def add_mu_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--mu",
        type=float,
        default=0.01,
        help="FedProx's proximal coefficient for the shared model (ignored by "
        "--method fedavg; ditto defaults to 0 unless you pass it).",
    )


def add_sync_every_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Federated aggregation period, in local updates. 1 (default) ",
    )


def add_fedavgm_beta_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--fedavgm-beta",
        type=float,
        default=0.9,
        help="Server momentum of FedAvgM. 0 disables it, which makes the server "
        "take the plain FedAvg step (the average itself); with --mu 0 that gives "
        "FedAvg, with mu > 0 it gives FedProx as published (plain-average server "
        "+ proximal client term).",
    )


def add_server_lr_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--server-lr",
        type=float,
        default=None,
        help="Server learning rate of the federated update. Unset means 1 - beta, "
        "the gain-neutral value: the steady-state server step then equals plain "
        "FedAvg's, so beta only smooths and nothing has to be compensated on the "
        "client side. 1.0 is the unscaled form of FedAvgM (1/(1-beta) times "
        "larger, i.e. 10x at beta=0.9); 0 freezes the global model, a no-sharing "
        "control.",
    )


def add_use_gt_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--gt",
        action="store_true",
        help="Use ground-truth observations if selected.",
    )


def add_ranks_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--ranks",
        type=int,
        default=0,
        help="Processes to train in (0 = one per client/scene).",
    )


def add_threads_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="BLAS/OpenMP threads per rank, so one rank occupies that many cores. "
        "1 = one core per client, 0 = auto (cores available to this launch divided "
        "by the number of ranks).",
    )


def add_wandb_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging. Disabled by default for multi-client runs.",
    )


def add_batch_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--batch",
        type=int,
        default=1000,
        help="Number of training batches per client.",
    )


def add_reload_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload expert conditions instead of loading the condition cache.",
    )


def add_inference_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Sets the network in inference mode.",
    )


def add_network_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--network",
        choices=NETWORK_NAMES,
        default="default",
        help="Network config name from conf.networks.",
    )


def add_seed_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Defaults to a fresh random seed each launch (pass an "
        "int for a reproducible repeat).",
    )


def add_checkpoint_path(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--ckp_path",
        type=str,
        help="Relative checkpoint path from data folder.",
    )


def add_fed_arguments(parser: argparse.ArgumentParser):
    add_federated_argument(parser)
    add_method_argument(parser)
    add_personal_argument(parser)
    add_mu_argument(parser)
    add_ranks_argument(parser)
    add_sync_every_argument(parser)
    add_server_lr_argument(parser)
    add_fedavgm_beta_argument(parser)
    add_threads_argument(parser)


def add_heca_arguments(parser: argparse.ArgumentParser):
    add_network_argument(parser)
    add_wandb_argument(parser)
    add_use_gt_argument(parser)
    add_batch_argument(parser)
    add_virtual_argument(parser)
    add_scenes_argument(parser)
    add_tag_argument(parser)
    add_smode_argument(parser)
    add_reload_argument(parser)
    add_seed_argument(parser)


def add_eval_arguments(parser: argparse.ArgumentParser):
    add_inference_argument(parser)
    add_checkpoint_path(parser)


def _base_tag(args: argparse.Namespace, federated: bool) -> str:
    final_tag = ""
    final_tag += args.tag
    final_tag += "-"
    final_tag += args.network
    final_tag += "-"
    final_tag += "gt-" if args.gt else ""
    final_tag += "virt-" if args.virtual else ""
    final_tag += "fed-" if federated else ""
    if federated:
        final_tag += f"k{args.k}-"
        final_tag += f"mu{args.mu}"
        beta = getattr(args, "fedavgm_beta", 0.9)
        eta = getattr(args, "server_lr", None)
        eta = 1.0 - beta if eta is None else eta
        final_tag += f"-b{beta:g}-eta{eta:g}"
    return final_tag


def generate_tag(args: argparse.Namespace, federated: bool) -> str:
    """Full run tag, unique per seed (also the checkpoint directory name)."""
    return f"{_base_tag(args, federated)}_s{args.seed}"


def generate_group(args: argparse.Namespace, federated: bool) -> str:
    """Wandb group: the seed-independent name shared by every repeat."""
    return _base_tag(args, federated)
