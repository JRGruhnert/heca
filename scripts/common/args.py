import argparse
from conf.networks import NETWORK_NAMES
from heca.graphs.graph import SubgoalMode


def add_smode_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--smode",
        type=SubgoalMode,
        choices=list(SubgoalMode),
        default=SubgoalMode.BOTH,
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
        help="Scene module tag (e.g. scene1, sceneog).",
    )


def add_tag_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--tag",
        required=True,
        help="Run tag for identification.",
    )


def add_model_argument(parser: argparse.ArgumentParser, default=None):
    parser.add_argument(
        "--model",
        default=default,
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


def add_use_gt_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--gt",
        action="store_true",
        help="Use ground-truth observations. Off unless passed, which selects the "
        "image variant (tapas_img.pt / conditions-vis.joblib).",
    )


def add_ranks_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--ranks",
        type=int,
        default=0,
        help="Processes to train in (0 = one per client/scene). Ranks synchronize "
        "with torch.distributed collectives.",
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
        default=750,
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


def add_heca_arguments(parser: argparse.ArgumentParser):
    add_network_argument(parser)
    add_federated_argument(parser)
    add_method_argument(parser)
    add_personal_argument(parser)
    add_mu_argument(parser)
    add_wandb_argument(parser)
    add_use_gt_argument(parser)
    add_batch_argument(parser)
    add_virtual_argument(parser)
    add_scene_argument(parser)
    add_tag_argument(parser)
    add_smode_argument(parser)
    add_inference_argument(parser)
    add_reload_argument(parser)
    add_seed_argument(parser)
    add_ranks_argument(parser)


def subgoal_tag(smode: SubgoalMode) -> str:
    if smode == SubgoalMode.GOAL:
        return "g"
    elif smode == SubgoalMode.CHAIN:
        return "c"
    elif smode == SubgoalMode.BOTH:
        return "b"
    raise ValueError


def _base_tag(args: argparse.Namespace) -> str:
    final_tag = ""
    final_tag += "fed-" if args.federated else ""
    final_tag += args.tag
    final_tag += "-"
    final_tag += args.network
    final_tag += "-"
    final_tag += "gt" if args.gt else ""
    final_tag += "-"
    final_tag += "virt" if args.virtual else ""
    if args.federated and getattr(args, "method", "fedprox") == "ditto":
        final_tag += f"-ditto{args.personal_coef:g}"
    return final_tag


def generate_tag(args: argparse.Namespace) -> str:
    """Full run tag, unique per seed (also the checkpoint directory name)."""
    return f"{_base_tag(args)}_s{args.seed}"


def generate_group(args: argparse.Namespace) -> str:
    """Wandb group: the seed-independent name shared by every repeat."""
    return _base_tag(args)
